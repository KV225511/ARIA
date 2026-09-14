"""Frozen v7 Low-class decision-bias protocol.

V7 intentionally consumes no new calibration validation attempt: v6 validation
has already been observed. Selection is grouped-training OOF only and the
locked test is the sole clean final evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

from modules.module_07_rl.calibration_protocol import atomic_json_write, canonical_json_hash
from modules.module_07_rl.calibration_protocol_v6 import (
    validate_calibration_protocol_v6, validate_protocol_state,
)

CALIBRATION_PROTOCOL_VERSION = "aria-calibration-protocol-v7"
CALIBRATION_ALGORITHM_VERSION = "aria-belief-calibration-v7"
CALIBRATION_CV_REPORT_VERSION = "aria-calibration-cv-report-v4"
DEVELOPMENT_BUNDLE_VERSION = "aria-development-bundle-v4"
PROTOCOL_STATE_VERSION = "aria-calibration-protocol-state-v2"
LOW_CLASS_LOGIT_BIAS_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MAX_CALIBRATION_CANDIDATES = len(LOW_CLASS_LOGIT_BIAS_VALUES)


def _load(value):
    return value if isinstance(value, dict) else json.loads(Path(value).read_text(encoding="utf-8"))


def _validate_hash(value, name):
    unsigned = dict(value)
    stored = unsigned.pop(name, None)
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError(f"invalid {name}")
    return stored


def freeze_calibration_protocol_v7(parent_protocol, parent_state, parent_config, output_dir):
    """Freeze a v7 protocol from a valid, unreleased v6 lineage."""
    v6 = validate_calibration_protocol_v6(parent_protocol)
    state = validate_protocol_state(parent_state, protocol=v6)
    if state["current_status"] != "PROVISIONAL_SYNTHETIC" or state["validation_executions"] != 1:
        raise ValueError("v7 requires a once-validated provisional v6 parent")
    parent_config_hash = getattr(parent_config, "config_hash", None)
    if parent_config_hash is None and isinstance(parent_config, dict):
        parent_config_hash = parent_config.get("config_hash")
    if not parent_config_hash:
        raise ValueError("parent v6 config hash is required")
    output = Path(output_dir)
    path = output / "protocol" / "calibration_protocol_v7.json"
    state_path = output / "protocol" / "calibration_protocol_state_v2.json"
    if path.exists() or state_path.exists():
        raise FileExistsError("v7 protocol already exists")
    protocol = {
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION,
        "initial_protocol_status": "FROZEN_FOR_DEVELOPMENT",
        "parent_v6_protocol_hash": v6["protocol_hash"],
        "parent_v6_state_hash": state["state_hash"],
        "parent_v6_belief_config_hash": parent_config_hash,
        "raw_file_sha256": v6["raw_file_sha256"],
        "raw_dataset_hash": v6["raw_dataset_hash"],
        "split_assignment_hash": v6["split_assignment_hash"],
        "locked_test_assignment_hash": v6["locked_test_assignment_hash"],
        "candidate_values": {"low_class_logit_bias": LOW_CLASS_LOGIT_BIAS_VALUES},
        "maximum_calibration_candidates": MAX_CALIBRATION_CANDIDATES,
        "candidate_selection_source": "original-training-components-only",
        "independent_development_validation_available": False,
        "clean_final_evaluation_source": "locked-test-only",
    }
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    status = {
        "schema_version": PROTOCOL_STATE_VERSION,
        "protocol_hash": protocol["protocol_hash"],
        "current_status": "FROZEN_FOR_DEVELOPMENT",
        "revision": 1,
        "locked_test_attempts": 0,
    }
    status["state_hash"] = canonical_json_hash(status)
    atomic_json_write(path, protocol)
    atomic_json_write(state_path, status)
    return protocol


def validate_calibration_protocol_v7(value):
    protocol = _load(value)
    _validate_hash(protocol, "protocol_hash")
    expected = {
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION,
        "initial_protocol_status": "FROZEN_FOR_DEVELOPMENT",
        "candidate_values": {"low_class_logit_bias": LOW_CLASS_LOGIT_BIAS_VALUES},
        "maximum_calibration_candidates": MAX_CALIBRATION_CANDIDATES,
        "candidate_selection_source": "original-training-components-only",
        "independent_development_validation_available": False,
        "clean_final_evaluation_source": "locked-test-only",
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise ValueError(f"v7 protocol {key} changed")
    return protocol
