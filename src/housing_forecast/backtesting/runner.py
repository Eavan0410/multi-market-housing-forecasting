"""Backtest runner: same folds, same information set, same output format for every model."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from housing_forecast.backtesting.infoset import SampleSet
from housing_forecast.backtesting.scaling import WindowStandardizer
from housing_forecast.backtesting.splits import (
    Split, check_no_leakage, mask_unpublished_labels, eval_mask, train_mask,
)
from housing_forecast.data.contract import PRED_COLUMNS, validate_predictions
from housing_forecast.models.base import Forecaster, set_global_seed


# The pretraining pool is overseas only: no Taiwan national or city series.

def pool_target_only(target_series: str) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda meta: (meta["series_id"] == target_series).to_numpy()


def pool_overseas(home_country: str = "Taiwan") -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda meta: (meta["country"] != home_country).to_numpy()


def pool_overseas_plus_target(target_series: str, home_country: str = "Taiwan"):
    return lambda meta: ((meta["country"] != home_country) | (meta["series_id"] == target_series)).to_numpy()


def pool_all() -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda meta: np.ones(len(meta), dtype=bool)



def run_backtest(
    samples: SampleSet,
    splits: list[Split],
    model_factory: Callable[[], Forecaster],
    cfg: dict,
    train_pool: Callable[[pd.DataFrame], np.ndarray],
    model_name: str | None = None,
) -> pd.DataFrame:
    """Run one model over all folds and return predictions in the shared CSV format."""
    eval_series = cfg["target"]["eval_series"]
    seeds = cfg["run"]["seeds"]
    gap = cfg["periods"]["gap_quarters"]
    rows = []

    for split in splits:
        train = samples.subset(train_mask(samples, split) & train_pool(samples.meta))
        train = mask_unpublished_labels(train, split)
        test = samples.subset(eval_mask(samples, split, eval_series))
        if len(test) == 0:
            continue
        check_no_leakage(train, test, split, gap)

        probe = model_factory()
        if probe.needs_scaling:
            scaler = WindowStandardizer().fit(train.X)          # training window only
            train = train.replace(X=scaler.transform(train.X))
            test = test.replace(X=scaler.transform(test.X))

        for seed in (seeds if probe.stochastic else seeds[:1]):
            set_global_seed(seed)
            model = model_factory()
            model.fit(train, seed)
            y_pred = np.asarray(model.predict(test), dtype=float)
            if y_pred.shape != test.y.shape:
                raise ValueError(f"predict() returned {y_pred.shape}, expected {test.y.shape}")
            name = model_name or model.name
            for j, h in enumerate(test.horizons):
                ok = ~np.isnan(test.y[:, j])                     # only score labels that exist
                origins = pd.PeriodIndex(test.meta["origin"][ok], freq="Q")
                rows.append(pd.DataFrame({
                    "model": name,
                    "series_id": test.meta["series_id"][ok].to_numpy(),
                    "origin_quarter": origins.astype(str),
                    "as_of_date": test.meta["as_of_date"][ok].dt.strftime("%Y-%m-%d").to_numpy(),
                    "target_quarter": (origins + h).astype(str),
                    "horizon": h,
                    "y_true": test.y[ok, j],
                    "y_pred": y_pred[ok, j],
                    "target_type": cfg["target"]["target_type"],
                    "seed": seed,
                    "split_id": split.split_id,
                }))

    preds = pd.concat(rows, ignore_index=True)[PRED_COLUMNS] if rows else pd.DataFrame(columns=PRED_COLUMNS)
    validate_predictions(preds)
    return preds
