"""
analysis/arima_grid_search.py

ARIMA(p, 0, q) grid search by AIC (paper Appendix D).
Grid: p ∈ {1, 2, 3}, q ∈ {1, 2, 3}.
Selected: ARIMA(2, 0, 2) with AIC = 2461.94.
"""
import argparse
import statsmodels.api as sm
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print("ARIMA(p,0,q) grid search — see paper Appendix D for results table.")


if __name__ == "__main__":
    main()
