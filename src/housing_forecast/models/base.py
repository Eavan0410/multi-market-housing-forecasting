"""Forecaster interface and reproducible random seeds.

Models fit one training fold and predict cumulative real HPI log growth at each
horizon. Missing training labels are masked; predictions must be finite."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np

from housing_forecast.backtesting.infoset import SampleSet


class Forecaster(ABC):
    name: str = "base"
    stochastic: bool = False
    needs_scaling: bool = False

    @abstractmethod
    def fit(self, train: SampleSet, seed: int) -> None: ...

    @abstractmethod
    def predict(self, test: SampleSet) -> np.ndarray: ...


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
