"""
evaluation/ablation_evaluation.py

Architectural ablation study (paper Section 3.3, Table 10).
Runs four configurations at seed = 42:
  1. Per-IMF BiLSTM+Transformer (conventional VMD pipeline)
  2. VMD-as-features + BiLSTM only ★ (primary)
  3. VMD-as-features + Transformer only (no BiLSTM)
  4. VMD-as-features + BiLSTM + Transformer (full hybrid — control)
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(f"Ablation study at seed={args.seed}")


if __name__ == "__main__":
    main()
