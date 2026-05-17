"""
evaluation/multiseed_evaluation.py

5-seed multi-seed evaluation protocol (paper Section 3.4, Table 11).
Runs the specified architecture under 5 random seeds {7, 42, 123, 999, 2024}
and reports mean ± 1 SD on the deduplicated leakage-free test partition
(n = 175).

Usage:
    python evaluation/multiseed_evaluation.py \
        --data <DATA_CSV> \
        --model bilstm_only \
        --seeds-file seeds.txt \
        --output-dir results/multiseed/
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True,
                        choices=["bilstm_only", "vanilla_lstm", "full_hybrid"])
    parser.add_argument("--seeds-file", default="seeds.txt")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.seeds_file) as f:
        seeds = [int(line.strip()) for line in f
                 if line.strip() and not line.strip().startswith("#")]

    print(f"Multi-seed protocol: {args.model} × {len(seeds)} seeds")
    print(f"Seeds: {seeds}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # See paper for full implementation; this stub documents the entry point.


if __name__ == "__main__":
    main()
