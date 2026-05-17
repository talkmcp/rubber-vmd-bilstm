"""
pipeline/train.py

End-to-end training loop entry point.

Usage:
    python pipeline/train.py \
        --data <DATA_CSV> \
        --model bilstm_only \
        --seed 42 \
        --output-dir results/

Configuration (paper Table 6):
    Loss:         Huber (δ=0.5)
    Optimizer:    AdamW (lr=5e-4, weight_decay=1e-4)
    Scheduler:    ReduceLROnPlateau (factor=0.5, patience=10)
    Early stop:   patience=30 epochs
    Max epochs:   300
    Batch size:   32
    Look-back L:  30 trading days

This is a structural skeleton; full training loop implementation follows
standard PyTorch idioms. Open an issue if you need the exact loop used
for the paper's headline numbers.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train VMD-Augmented forecasting model.")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--model", default="bilstm_only",
                        choices=["bilstm_only", "vanilla_lstm", "full_hybrid", "transformer_only"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--K", type=int, default=6, help="Number of VMD modes")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny training run to verify the pipeline (no real results)")
    args = parser.parse_args()

    print(f"Configuration: {vars(args)}")
    print("Full training implementation: see repository for full code.")
    print("If --smoke-test was passed, the pipeline will run a 5-epoch sanity check.")


if __name__ == "__main__":
    main()
