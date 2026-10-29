"""Circular block bootstrap over forecast origins, with seeds held fixed.

Resample whole origins to keep overlapping horizons and series together.
Score each seed separately, then average its metric; paired comparisons use
identical draws for both models. Negative RMSE/MAE differences favor model A.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_forecast.data.contract import check_alignment
from housing_forecast.evaluation.metrics import direction_accuracy, mae, rmse

METRICS = {
    "rmse": lambda yt, yp: rmse(yp - yt),
    "mae": lambda yt, yp: mae(yp - yt),
    "dir_acc": direction_accuracy,
}
FORECAST_KEY = ["series_id", "origin_quarter", "horizon"]


def circular_block_indices(n_units: int, block_length: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Return n_boot draws, each containing n_units circular-block positions."""
    n_blocks = int(np.ceil(n_units / block_length))
    starts = rng.integers(0, n_units, size=(n_boot, n_blocks))
    indices = (starts[:, :, None] + np.arange(block_length)) % n_units
    return indices.reshape(n_boot, -1)[:, :n_units]


def _model_arrays(preds, model):
    """Align truth and per-seed predictions on sorted forecast keys."""
    selected = preds[preds["model"] == model]
    if selected.empty:
        raise ValueError(f"model {model} is not present")
    wide = selected.pivot(index=FORECAST_KEY, columns="seed", values="y_pred").sort_index()
    truth = selected.drop_duplicates(FORECAST_KEY).set_index(FORECAST_KEY)["y_true"]
    return wide.index, truth.reindex(wide.index).to_numpy(), wide.to_numpy()


def _mean_seed_metric(metric, truth, predictions):
    return float(np.nanmean([
        metric(truth, predictions[:, seed]) for seed in range(predictions.shape[1])
    ]))


def _resampled_rows(index, settings):
    """Yield row positions, keeping every forecast of an origin together."""
    origins = index.get_level_values("origin_quarter").to_numpy()
    unique = np.unique(origins)
    rows = [np.flatnonzero(origins == origin) for origin in unique]
    draws = circular_block_indices(
        len(unique), settings["block_length"], settings["n_boot"],
        np.random.default_rng(settings["seed"]),
    )
    for draw in draws:
        yield np.concatenate([rows[i] for i in draw])


def _interval(values, confidence):
    tail = (1 - confidence) / 2
    return np.nanquantile(values, [tail, 1 - tail])


def paired_bootstrap(preds: pd.DataFrame, model_a: str, model_b: str, cfg: dict,
                     metric: str = "rmse", by_horizon: bool = True) -> pd.DataFrame:
    """CI of mean metric(A) minus mean metric(B), averaged over fitted seeds."""
    check_alignment(preds)
    index, truth, a = _model_arrays(preds, model_a)
    _, _, b = _model_arrays(preds, model_b)
    settings, score = cfg["bootstrap"], METRICS[metric]
    horizons = index.get_level_values("horizon")
    groups = [("all", np.arange(len(index)))]
    if by_horizon:
        groups += [(h, np.flatnonzero(horizons == h)) for h in np.unique(horizons)]

    results = []
    for horizon, positions in groups:
        keys = index[positions]
        yt, ya, yb = truth[positions], a[positions], b[positions]

        def difference(rows):
            return (_mean_seed_metric(score, yt[rows], ya[rows])
                    - _mean_seed_metric(score, yt[rows], yb[rows]))

        diffs = np.array([difference(rows) for rows in _resampled_rows(keys, settings)])
        low, high = _interval(diffs, settings["ci"])
        better = diffs > 0 if metric == "dir_acc" else diffs < 0
        results.append({
            "model_a": model_a, "model_b": model_b, "metric": metric, "horizon": horizon,
            "n_origins": keys.get_level_values("origin_quarter").nunique(),
            "n_rows": len(keys), "diff": difference(slice(None)),
            "ci_low": low, "ci_high": high, "share_a_better": float(better.mean()),
        })
    return pd.DataFrame(results)


def metric_ci(preds: pd.DataFrame, model: str, cfg: dict, metric: str = "rmse") -> pd.DataFrame:
    """CI of a model's mean per-seed metric, separately for each horizon."""
    check_alignment(preds)
    index, truth, predictions = _model_arrays(preds, model)
    settings, score = cfg["bootstrap"], METRICS[metric]
    horizons = index.get_level_values("horizon")
    results = []
    for horizon in np.unique(horizons):
        mask = horizons == horizon
        yt, yp = truth[mask], predictions[mask]
        values = np.array([
            _mean_seed_metric(score, yt[rows], yp[rows])
            for rows in _resampled_rows(index[mask], settings)
        ])
        low, high = _interval(values, settings["ci"])
        results.append({
            "model": model, "metric": metric, "horizon": horizon,
            "value": _mean_seed_metric(score, yt, yp), "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(results)
