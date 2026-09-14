"""Fail-closed inference runtime for a validated ARIA IQL checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
import torch

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.calibration_protocol import file_sha256
from modules.module_07_rl.calibration_protocol_v7 import (
    validate_calibration_protocol_v7,
    validate_development_bundle_v7,
    validate_protocol_state_v7,
)
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_07_rl.state_builder import (
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
)
from modules.module_07_rl.train import CHECKPOINT_SCHEMA_VERSION, IQLNetworks


IQL_POLICY_RUNTIME_VERSION = "aria-iql-policy-runtime-v1"


class IQLPolicy:
    """A deterministic-by-default, action-mask-aware IQL policy."""

    def __init__(self, networks: IQLNetworks, *, checkpoint_sha256: str, metadata: dict):
        self.networks = networks.eval()
        self.checkpoint_sha256 = checkpoint_sha256
        self.metadata = dict(metadata)

    @classmethod
    def load(
        cls,
        checkpoint_file: str | Path,
        *,
        protocol_file: str | Path | None = None,
        protocol_state_file: str | Path | None = None,
        development_bundle_file: str | Path | None = None,
        split_manifest_file: str | Path | None = None,
        belief_config_file: str | Path | None = None,
        require_validated: bool = True,
    ) -> "IQLPolicy":
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("IQL checkpoint must be an object")
        if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported IQL checkpoint schema")
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("IQL checkpoint schema fields disagree")
        if CHECKPOINT_SCHEMA_VERSION not in checkpoint.get("supported_consumer_versions", []):
            raise ValueError("IQL checkpoint does not support this consumer")
        if checkpoint.get("state_schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("IQL checkpoint state schema mismatch")
        if checkpoint.get("state_feature_names") != list(STATE_FEATURE_NAMES):
            raise ValueError("IQL checkpoint state features mismatch")

        provenance_paths = (
            protocol_file,
            protocol_state_file,
            development_bundle_file,
            split_manifest_file,
            belief_config_file,
        )
        if any(path is not None for path in provenance_paths) and not all(
            path is not None for path in provenance_paths
        ):
            raise ValueError("IQL provenance validation requires all five artifact paths")
        if require_validated and not all(path is not None for path in provenance_paths):
            raise ValueError("validated IQL inference requires complete v7 provenance")

        metadata = {
            key: checkpoint.get(key)
            for key in (
                "protocol_hash",
                "belief_config_hash",
                "development_bundle_hash",
                "split_manifest_hash",
                "raw_file_sha256",
                "raw_dataset_hash",
                "git_commit",
                "environment_fingerprint_hash",
            )
        }
        if all(path is not None for path in provenance_paths):
            protocol = validate_calibration_protocol_v7(
                protocol_file, split_manifest=split_manifest_file
            )
            state = validate_protocol_state_v7(protocol_state_file, protocol=protocol)
            allowed = {"VALIDATED_SYNTHETIC"} if require_validated else {
                "PROVISIONAL_TRAINING_CV", "VALIDATED_SYNTHETIC"
            }
            if state["current_status"] not in allowed:
                raise ValueError("protocol state does not permit IQL inference")
            config = BeliefModelConfig.load(belief_config_file)
            bundle = validate_development_bundle_v7(
                development_bundle_file,
                protocol=protocol,
                state=state,
                belief_config_hash=config.config_hash,
                split_manifest=split_manifest_file,
            )
            expected = {
                "protocol_hash": protocol["protocol_hash"],
                "belief_config_hash": config.config_hash,
                "development_bundle_hash": bundle["bundle_hash"],
                "split_manifest_hash": bundle["split_manifest_hash"],
                "raw_file_sha256": protocol["raw_file_sha256"],
                "raw_dataset_hash": protocol["raw_dataset_hash"],
                "git_commit": protocol["code_commit"],
                "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
            }
            for field, value in expected.items():
                if checkpoint.get(field) != value:
                    raise ValueError(f"IQL checkpoint {field} mismatch")
            if checkpoint.get("belief_config") != config.to_dict():
                raise ValueError("IQL checkpoint embedded belief configuration mismatch")
            metadata.update(expected)
            metadata["protocol_status"] = state["current_status"]

        networks = IQLNetworks(STATE_DIM, len(RL_ACTION_SPACE))
        try:
            networks.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise ValueError("IQL checkpoint model architecture mismatch") from exc
        return cls(
            networks,
            checkpoint_sha256=file_sha256(checkpoint_file),
            metadata=metadata,
        )

    @staticmethod
    def _validated_inputs(observation: Any, action_mask: Any) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(action_mask)
        if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
            raise ValueError(f"observation must contain {STATE_DIM} finite values")
        if mask.shape != (len(RL_ACTION_SPACE),):
            raise ValueError("action mask shape mismatch")
        if not np.all(np.isin(mask, (0, 1, False, True))):
            raise ValueError("action mask must be binary")
        legal = mask.astype(bool)
        if not np.any(legal):
            raise ValueError("action mask contains no legal action")
        return state, legal

    def action_distribution(
        self, observation: Any, action_mask: Any, *, temperature: float = 1.0
    ) -> np.ndarray:
        state, legal = self._validated_inputs(observation, action_mask)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        with torch.no_grad():
            logits = self.networks.policy_net(
                torch.as_tensor(state[None, :], dtype=torch.float32)
            ).cpu().numpy()[0].astype(np.float64)
        if logits.shape != (len(RL_ACTION_SPACE),) or not np.all(np.isfinite(logits)):
            raise ValueError("policy produced invalid logits")
        scaled = logits[legal] / float(temperature)
        scaled -= np.max(scaled)
        legal_probabilities = np.exp(scaled)
        legal_probabilities /= legal_probabilities.sum()
        probabilities = np.zeros(len(RL_ACTION_SPACE), dtype=np.float64)
        probabilities[legal] = legal_probabilities
        if not np.all(np.isfinite(probabilities)) or not np.isclose(
            probabilities.sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("policy produced invalid probabilities")
        return probabilities

    def select_action(
        self,
        observation: Any,
        action_mask: Any,
        *,
        deterministic: bool = True,
        rng: np.random.Generator | None = None,
        temperature: float = 1.0,
    ) -> dict:
        _, legal = self._validated_inputs(observation, action_mask)
        probabilities = self.action_distribution(
            observation, action_mask, temperature=temperature
        )
        if deterministic:
            action_idx = int(np.argmax(probabilities))
        else:
            if rng is None:
                raise ValueError("stochastic action selection requires an explicit RNG")
            action_idx = int(rng.choice(len(RL_ACTION_SPACE), p=probabilities))
        if not legal[action_idx]:
            raise RuntimeError("IQL policy selected an illegal action")
        return {
            "schema_version": IQL_POLICY_RUNTIME_VERSION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "action_idx": action_idx,
            "action_name": RL_ACTION_SPACE[action_idx],
            "action_mask": legal.astype(int).tolist(),
            "action_probabilities": probabilities.tolist(),
            "selected_action_probability": float(probabilities[action_idx]),
            "deterministic": bool(deterministic),
            "temperature": float(temperature),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate an ARIA IQL checkpoint for inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-state", required=True)
    parser.add_argument("--development-bundle", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--belief-config", required=True)
    args = parser.parse_args(argv)
    policy = IQLPolicy.load(
        args.checkpoint,
        protocol_file=args.protocol,
        protocol_state_file=args.protocol_state,
        development_bundle_file=args.development_bundle,
        split_manifest_file=args.split_manifest,
        belief_config_file=args.belief_config,
        require_validated=True,
    )
    print(json.dumps({
        "runtime_version": IQL_POLICY_RUNTIME_VERSION,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "protocol_hash": policy.metadata["protocol_hash"],
        "belief_config_hash": policy.metadata["belief_config_hash"],
        "protocol_status": policy.metadata["protocol_status"],
        "ready_for_inference": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
