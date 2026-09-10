"""One-command, test-locked calibration and replay preparation."""

from __future__ import annotations

import argparse
import json

from modules.module_07_rl.replay_dataset import (
    freeze_development_splits,
    prepare_development_calibration,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--raw-file")
    parser.add_argument("--base-split-manifest")
    parser.add_argument("--development-train")
    parser.add_argument("--development-validation")
    parser.add_argument("--split-manifest")
    parser.add_argument("--protocol")
    parser.add_argument("--dependency-lock")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.development_train or args.development_validation or args.split_manifest or args.protocol:
        if not (args.development_train and args.development_validation and args.split_manifest and args.protocol):
            parser.error("development calibration requires --development-train, --development-validation, --split-manifest, and --protocol")
        result = prepare_development_calibration(
            args.development_train, args.development_validation, args.split_manifest,
            args.protocol, args.output_dir, bootstrap_samples=args.bootstrap_samples,
        )
    else:
        if not args.raw_file or not args.base_split_manifest:
            parser.error("split freeze requires --raw-file and --base-split-manifest")
        result = freeze_development_splits(
            args.raw_file, args.base_split_manifest, args.output_dir, seed=args.split_seed,
            dependency_lock_path=args.dependency_lock,
        )
    print(json.dumps(result, indent=2))
