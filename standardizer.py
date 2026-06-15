import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class ZScoreStandardizer(BaseEstimator, TransformerMixin):
    """Z-score standardization service (mean=0, std=1) powered by Scikit-learn."""

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler = StandardScaler(
            with_mean=self.with_mean, with_std=self.with_std
        )
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = self._validate_input(X)
        self._scaler.fit(X)
        self.mean_ = self._scaler.mean_
        self.scale_ = self._scaler.scale_
        return self

    def transform(self, X):
        X = self._validate_input(X)
        return self._scaler.transform(X)

    def fit_transform(self, X, y=None):
        X = self._validate_input(X)
        result = self._scaler.fit_transform(X)
        self.mean_ = self._scaler.mean_
        self.scale_ = self._scaler.scale_
        return result

    def inverse_transform(self, X):
        X = self._validate_input(X)
        return self._scaler.inverse_transform(X)

    def _validate_input(self, X):
        if isinstance(X, list):
            X = np.array(X, dtype=np.float64)
        elif isinstance(X, np.ndarray):
            X = X.astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if np.any(np.isnan(X)):
            raise ValueError("Input contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("Input contains infinite values")
        return X

    def get_params(self):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer has not been fitted yet")
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "with_mean": self.with_mean,
            "with_std": self.with_std,
        }
