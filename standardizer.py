import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class ZScoreStandardizer(BaseEstimator, TransformerMixin):
    """Data scaling service supporting Z-score standardization and Min-Max normalization.

    Parameters
    ----------
    method : str, default='zscore'
        Scaling method. 'zscore' for (mean=0, std=1), 'minmax' for [0, 1] range.
    with_mean : bool, default=True
        Only used with method='zscore'. If True, center the data before scaling.
    with_std : bool, default=True
        Only used with method='zscore'. If True, scale to unit variance.
    feature_range : tuple (min, max), default=(0, 1)
        Only used with method='minmax'. Desired range of transformed data.
    """

    VALID_METHODS = ("zscore", "minmax")

    def __init__(self, method="zscore", with_mean=True, with_std=True,
                 feature_range=(0, 1)):
        self.method = method
        self.with_mean = with_mean
        self.with_std = with_std
        self.feature_range = feature_range
        self.mean_ = None
        self.scale_ = None
        self.zero_std_features_ = None
        self.constant_features_ = None
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Valid options are: {self.VALID_METHODS}"
            )
        if method == "zscore":
            self._scaler = StandardScaler(
                with_mean=self.with_mean, with_std=self.with_std
            )
        else:
            if not (isinstance(feature_range, tuple) and len(feature_range) == 2):
                raise ValueError(
                    "feature_range must be a tuple of (min, max)"
                )
            if feature_range[0] >= feature_range[1]:
                raise ValueError(
                    "feature_range[0] must be less than feature_range[1]"
                )
            self._scaler = MinMaxScaler(feature_range=feature_range)

    def fit(self, X, y=None):
        X = self._validate_input(X)
        self._scaler.fit(X)
        if self.method == "zscore":
            self.mean_ = self._scaler.mean_
            self.scale_ = self._scaler.scale_
            self._check_zero_std(X)
        else:
            self.mean_ = self._scaler.data_min_
            self.scale_ = self._scaler.data_range_
            self._check_constant_range(X)
        return self

    def transform(self, X):
        X = self._validate_input(X)
        return self._scaler.transform(X)

    def fit_transform(self, X, y=None):
        X = self._validate_input(X)
        result = self._scaler.fit_transform(X)
        if self.method == "zscore":
            self.mean_ = self._scaler.mean_
            self.scale_ = self._scaler.scale_
            self._check_zero_std(X)
        else:
            self.mean_ = self._scaler.data_min_
            self.scale_ = self._scaler.data_range_
            self._check_constant_range(X)
        return result

    def inverse_transform(self, X):
        X = self._validate_input(X)
        return self._scaler.inverse_transform(X)

    def _check_zero_std(self, X):
        if not self.with_std:
            self.zero_std_features_ = np.array([], dtype=int)
            return
        stds = np.std(X, axis=0, ddof=0)
        zero_mask = stds == 0.0
        self.zero_std_features_ = np.where(zero_mask)[0]
        if len(self.zero_std_features_) > 0:
            cols = self.zero_std_features_.tolist()
            raise ValueError(
                f"Cannot standardize feature(s) with zero standard deviation "
                f"(constant column(s)): {cols}"
            )

    def _check_constant_range(self, X):
        ranges = np.ptp(X, axis=0)
        const_mask = ranges == 0.0
        self.constant_features_ = np.where(const_mask)[0]
        if len(self.constant_features_) > 0:
            cols = self.constant_features_.tolist()
            raise ValueError(
                f"Cannot normalize feature(s) with zero range "
                f"(constant column(s)): {cols}"
            )

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
            raise RuntimeError("Scaler has not been fitted yet")
        if self.method == "zscore":
            return {
                "method": self.method,
                "mean": self.mean_.tolist(),
                "scale": self.scale_.tolist(),
                "with_mean": self.with_mean,
                "with_std": self.with_std,
            }
        else:
            return {
                "method": self.method,
                "data_min": self.mean_.tolist(),
                "data_range": self.scale_.tolist(),
                "feature_range": self.feature_range,
            }
