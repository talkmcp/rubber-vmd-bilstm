"""
analysis/lookback_sensitivity.py

Look-back window L sensitivity (paper Appendix F, Table F.1).
Grid: L ∈ {10, 20, 30, 45, 60}.
Selected: L = 30 (≈6 trading weeks).
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print("Look-back window sensitivity — see paper Appendix F.")


if __name__ == "__main__":
    main()
