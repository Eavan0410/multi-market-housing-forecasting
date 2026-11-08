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

def pool_target_only(target_series: str) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda meta: (meta["series_id"] == target_series).to_numpy()


def pool_overseas(home_country: str = "Taiwan") -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda meta: (meta["country"] != home_country).to_numpy()


def pool_overseas_plus_target(target_series: str, home_country: str = "Taiwan"):
    return lambda meta: ((meta["country"] != home_country) | (meta["series_id"] == target_series)).to_numpy()


def pool_national() -> Callable[[pd.DataFrame], np.ndarray]:
    """Every national series, Taiwan included; city series left out (multi-market runs)."""
    return lambda meta: (meta["geography_level"] == "national").to_numpy()


def pool_all() -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda meta: np.ones(len(meta), dtype=bool)



def training_samples(samples: SampleSet, split: Split, train_pool: Callable[[pd.DataFrame], np.ndarray]) -> SampleSet:
    train = samples.subset(train_mask(samples, split) & train_pool(samples.meta))
    return mask_unpublished_labels(train, split)


def _prediction_rows(test, y_pred, name, seed, split, target_type):
    if y_pred.shape != test.y.shape:
        raise ValueError(f"predict() returned {y_pred.shape}, expected {test.y.shape}")
    rows = []
    for j, h in enumerate(test.horizons):
        ok = ~np.isnan(test.y[:, j])
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
            "target_type": target_type,
            "seed": seed,
            "split_id": split.split_id,
        }))
    return rows


def run_backtest(
    samples: SampleSet,
    splits: list[Split],
    model_factory: Callable[[], Forecaster],
    cfg: dict,
    train_pool: Callable[[pd.DataFrame], np.ndarray],
    model_name: str | None = None,
    reference_model_name: str | None = None,
) -> pd.DataFrame:
    """Run one model over all folds and return predictions in the shared CSV format.

    A model that has a diagnostics() method (e.g. ARIMA fit / fallback status per market)
    gets it collected per fold and seed into preds.attrs["diagnostics"].
    """
    eval_series = cfg["target"]["eval_series"]
    seeds = cfg["run"]["seeds"]
    gap = cfg["periods"]["gap_quarters"]
    rows, diag_rows, reference_rows = [], [], []

    for split in splits:
        train = training_samples(samples, split, train_pool)
        test = samples.subset(eval_mask(samples, split, eval_series))
        if len(test) == 0:
            continue
        check_no_leakage(train, test, split, gap)

        probe = model_factory()
        if reference_model_name is not None and not callable(getattr(probe, "predict_shared", None)):
            raise ValueError("A shared reference requires a model with predict_shared().")
        if probe.needs_scaling:
            scaler = WindowStandardizer().fit(train.X)          # training window only
            train = train.replace(X=scaler.transform(train.X))
            test = test.replace(X=scaler.transform(test.X))

        for seed in (seeds if probe.stochastic else seeds[:1]):
            set_global_seed(seed)
            model = model_factory()
            model.fit(train, seed)
            y_pred = np.asarray(model.predict(test), dtype=float)
            name = model_name or model.name
            diag = model.diagnostics() if callable(getattr(model, "diagnostics", None)) else None
            if diag is not None and len(diag):
                diag_rows.append(diag.assign(model=name, split_id=split.split_id, seed=seed))
            rows.extend(_prediction_rows(test, y_pred, name, seed, split, cfg["target"]["target_type"]))
            if reference_model_name is not None:
                shared = np.asarray(model.predict_shared(test), dtype=float)
                reference_rows.extend(_prediction_rows(test, shared, reference_model_name, seed, split,
                                                       cfg["target"]["target_type"]))

    preds = pd.concat(rows, ignore_index=True)[PRED_COLUMNS] if rows else pd.DataFrame(columns=PRED_COLUMNS)
    validate_predictions(preds)
    preds.attrs["diagnostics"] = pd.concat(diag_rows, ignore_index=True) if diag_rows else None
    if reference_model_name is not None:
        reference = (pd.concat(reference_rows, ignore_index=True)[PRED_COLUMNS] if reference_rows
                     else pd.DataFrame(columns=PRED_COLUMNS))
        validate_predictions(reference)
        preds.attrs["reference_predictions"] = reference
    return preds
