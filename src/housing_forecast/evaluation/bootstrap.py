"""Block bootstrap confidence intervals (bootstrap 置信区间).

Why blocks: labels for h = 1..4 overlap (origin t and t+1 share 3 quarters of
the 4-quarter label), so errors at nearby origins are correlated. Resampling
single rows would make the intervals too narrow.

Unit of resampling = ORIGIN QUARTER. Every row with that origin (all horizons,
all evaluated series — e.g. Taiwan national and its cities, which are not
independent) is kept together. We draw circular blocks of `block_length`
consecutive origins.

Paired comparison: model A and model B are scored on exactly the same
resampled rows each time, and we report the CI of metric(A) - metric(B).
Negative RMSE difference = A is better.
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


def circular_block_indices(n_units: int, block_length: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_units) array of resampled unit positions."""
    n_blocks = int(np.ceil(n_units / block_length))
    starts = rng.integers(0, n_units, size=(n_boot, n_blocks))
    offs = np.arange(block_length)
    idx = (starts[:, :, None] + offs[None, None, :]) % n_units
    return idx.reshape(n_boot, -1)[:, :n_units]


def _rows_by_origin(df: pd.DataFrame, origins: np.ndarray) -> list[np.ndarray]:
    pos = {o: i for i, o in enumerate(origins)}
    groups = [[] for _ in origins]
    for r, o in enumerate(df["origin_quarter"].to_numpy()):
        groups[pos[o]].append(r)
    return [np.asarray(g, dtype=int) for g in groups]


def paired_bootstrap(preds: pd.DataFrame, model_a: str, model_b: str, cfg: dict,
                     metric: str = "rmse", by_horizon: bool = True) -> pd.DataFrame:
    """CI of the difference of mean per-seed metrics (not ensemble metrics).

    Seeds are held fixed; uncertainty reflects resampled origin quarters.
    """
    bs = cfg["bootstrap"]
    check_alignment(preds)
    if model_a not in set(preds["model"]) or model_b not in set(preds["model"]):
        raise ValueError("both comparison models must be present")
    keys = ["series_id", "origin_quarter", "horizon"]
    a_rows = preds[preds["model"] == model_a]
    b_rows = preds[preds["model"] == model_b]
    a = a_rows.pivot(index=keys, columns="seed", values="y_pred").sort_index()
    b = b_rows.pivot(index=keys, columns="seed", values="y_pred").reindex(a.index)
    truth = a_rows.drop_duplicates(keys).set_index(keys)["y_true"].reindex(a.index)
    joined = a.index.to_frame(index=False)
    joined["y_true"] = truth.to_numpy()
    for j in range(a.shape[1]):
        joined[f"a_{j}"] = a.iloc[:, j].to_numpy()
    for j in range(b.shape[1]):
        joined[f"b_{j}"] = b.iloc[:, j].to_numpy()
    a_cols = [f"a_{j}" for j in range(a.shape[1])]
    b_cols = [f"b_{j}" for j in range(b.shape[1])]

    groups = [("all", joined)]
    if by_horizon:
        groups += [(h, g) for h, g in joined.groupby("horizon")]

    f = METRICS[metric]
    out = []
    for label, g in groups:
        g = g.reset_index(drop=True)
        origins = np.sort(g["origin_quarter"].unique())
        rows = _rows_by_origin(g, origins)
        yt, ya, yb = g["y_true"].to_numpy(), g[a_cols].to_numpy(), g[b_cols].to_numpy()
        point = _mean_seed_metric(f, yt, ya) - _mean_seed_metric(f, yt, yb)
        rng = np.random.default_rng(bs["seed"])
        draws = circular_block_indices(len(origins), bs["block_length"], bs["n_boot"], rng)
        diffs = np.empty(len(draws))
        for i, d in enumerate(draws):
            r = np.concatenate([rows[k] for k in d])
            diffs[i] = _mean_seed_metric(f, yt[r], ya[r]) - _mean_seed_metric(f, yt[r], yb[r])
        alpha = 1 - bs["ci"]
        lo, hi = np.nanquantile(diffs, [alpha / 2, 1 - alpha / 2])
        out.append({
            "model_a": model_a, "model_b": model_b, "metric": metric, "horizon": label,
            "n_origins": len(origins), "n_rows": len(g),
            "diff": point, "ci_low": lo, "ci_high": hi,
            "share_a_better": float(np.mean(diffs < 0)) if metric != "dir_acc" else float(np.mean(diffs > 0)),
        })
    return pd.DataFrame(out)


def _mean_seed_metric(f, truth, predictions):
    return float(np.nanmean([f(truth, predictions[:, j]) for j in range(predictions.shape[1])]))


def metric_ci(preds: pd.DataFrame, model: str, cfg: dict, metric: str = "rmse") -> pd.DataFrame:
    """CI of a model's mean per-seed metric, using identical draws for every seed."""
    bs = cfg["bootstrap"]
    check_alignment(preds)
    selected = preds[preds["model"] == model]
    if selected.empty:
        raise ValueError(f"model {model} is not present")
    f = METRICS[metric]
    keys = ["series_id", "origin_quarter", "horizon"]
    out = []
    for h, g in selected.groupby("horizon"):
        wide = g.pivot(index=keys, columns="seed", values="y_pred").sort_index()
        truth = g.drop_duplicates(keys).set_index(keys)["y_true"].reindex(wide.index)
        origins = np.sort(g["origin_quarter"].unique())
        rows = _rows_by_origin(wide.index.to_frame(index=False), origins)
        yt, yp = truth.to_numpy(), wide.to_numpy()
        rng = np.random.default_rng(bs["seed"])
        draws = circular_block_indices(len(origins), bs["block_length"], bs["n_boot"], rng)
        vals = np.array([_mean_seed_metric(f, yt[r], yp[r])
                         for r in (np.concatenate([rows[k] for k in d]) for d in draws)])
        alpha = 1 - bs["ci"]
        lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
        out.append({"model": model, "metric": metric, "horizon": h,
                    "value": _mean_seed_metric(f, yt, yp), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)
