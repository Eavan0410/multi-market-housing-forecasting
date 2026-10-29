"""Model interface every forecaster implements (baselines, GRU, transformers).

A model receives one fold's training SampleSet and predicts all horizons at
once for the fold's test SampleSet:

    model.fit(train, seed)                 # train.y may contain NaN (hidden labels) -> mask them
    y_pred = model.predict(test)           # shape (n_test, n_horizons)

Class attributes tell the runner how to treat the model:
    stochastic    : True -> run once per seed in the fixed seed list
    needs_scaling : True -> runner standardizes X with stats from the training fold
"""

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
