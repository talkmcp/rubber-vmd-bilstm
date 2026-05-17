"""
pipeline/normalization.py

MinMax scaler with explicit train-only fitting to prevent data leakage.
Implements Equations (2) and (3) of paper Section 2.2.3.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TrainOnlyMinMaxScaler:
    """Wrapper enforcing train-only fit. Range: [-1, 1]."""

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self._fitted = False

    def fit(self, X_train: np.ndarray) -> "TrainOnlyMinMaxScaler":
        self.scaler.fit(X_train)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler must be fit on training data first.")
        return self.scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X)
