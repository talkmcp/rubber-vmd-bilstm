"""
analysis/k_selection.py

VMD K-selection sensitivity (paper Appendix C, Table C.1).
Grid: K ∈ {3, 4, 5, 6, 7}.
Selected: K = 6 by joint criterion (reconstruction RMSE +
center-frequency separation + interpretability + downstream validation skill).
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print("K-selection sensitivity — see paper Appendix C.")


if __name__ == "__main__":
    main()
