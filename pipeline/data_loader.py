"""
pipeline/data_loader.py

CSV loader + chronological partitioner. Expected schema documented in
data_schema/feature_dictionary.md.

Implementation notes for paper users:
  1. Read CSV with Date as parse_dates index, sorted ascending
  2. Forward-fill missing (max 3 days) then linear interpolate (max 10 days)
  3. ADF tests on training partition only — flag I(1) features for differencing
  4. Apply first-differencing to 21 of 24 features (keep COVID_dummy, ENSO_ONI,
     Brent_return in levels)
  5. Chronological split:
       train: 2018-05-07 → 2023-08-27 (~80%)
       val:   2023-08-28 → 2025-09-17 (~10%)
       test:  2025-09-18 → 2026-02-27 (~9%)
  6. Fit MinMaxScaler on training partition ONLY (no leakage)
  7. Construct sliding-window samples with L=30 look-back
"""
import pandas as pd
import numpy as np
from pathlib import Path


# Placeholder — to be implemented by repository users following the schema
def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV, validate schema, return DataFrame indexed by Date."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df
