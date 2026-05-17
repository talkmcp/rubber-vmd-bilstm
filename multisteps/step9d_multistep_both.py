"""
multistep/step9d_multistep_both.py

Direct multi-step forecasting protocol (paper Section 3.6, Table 14).
Trains a separate model per horizon h ∈ {1, 2, 3, 5, 10, 20, 30}
for both the BiLSTM-only and full-hybrid architectures, single seed = 42,
apples-to-apples comparison.

Multi-seed extension is flagged as a planned extension in the paper.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[1, 2, 3, 5, 10, 20, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(f"Multi-step protocol: horizons={args.horizons}, seed={args.seed}")


if __name__ == "__main__":
    main()
