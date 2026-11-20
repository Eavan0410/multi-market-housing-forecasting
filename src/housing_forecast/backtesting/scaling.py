"""Standardization fitted on the training window only."""

from __future__ import annotations

import numpy as np


class WindowStandardizer:
    """Feature-wise mean and standard deviation fitted separately on each training fold."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "WindowStandardizer":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        self.std_ = np.nanstd(flat, axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("fit() must be called on the training window first")
        return (X - self.mean_) / self.std_
