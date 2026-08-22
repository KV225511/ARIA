"""One-command, test-locked calibration and replay preparation."""

from __future__ import annotations

import argparse
import json

from modules.module_07_rl.replay_dataset import prepare_calibrate_replay


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("output_dir")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(prepare_calibrate_replay(
        args.raw_file,
        args.output_dir,
        split_seed=args.split_seed,
        bootstrap_samples=args.bootstrap_samples,
    ), indent=2))
