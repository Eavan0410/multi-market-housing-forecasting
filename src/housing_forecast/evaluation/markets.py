"""Multi-market evaluation: how accurate is each model in each market and at each horizon?

Pooled RMSE over many markets is dominated by the most volatile ones, so the main
summary is RELATIVE: in every market and horizon, a model's RMSE divided by the
reference model's RMSE (normally no-change) in that same market. Below 1 = better.
Across markets we report the median and the geometric mean of that ratio and the
share of markets where the model beats the reference.

Seeds are handled as elsewhere: metrics per seed, then averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_forecast.data.contract import check_alignment


def metrics_by_market(preds: pd.DataFrame) -> pd.DataFrame:
    """Per model x market x horizon: metrics per seed, then the mean over seeds."""
    check_alignment(preds)
    yt, yp = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    signed = yt != 0                                   # same rule as metrics.direction_accuracy
    d = preds.assign(se=(yp - yt) ** 2, ae=np.abs(yp - yt),
                     dir_hit=signed & (np.sign(yp) == np.sign(yt)), dir_n=signed)
    per_seed = d.groupby(["model", "series_id", "horizon", "seed"]).agg(
        n=("se", "size"), mse=("se", "mean"), mae=("ae", "mean"),
        dir_hit=("dir_hit", "sum"), dir_n=("dir_n", "sum")).reset_index()
    per_seed["rmse"] = np.sqrt(per_seed["mse"])
    per_seed["dir_acc"] = per_seed["dir_hit"] / per_seed["dir_n"].where(per_seed["dir_n"] > 0)
    out = per_seed.groupby(["model", "series_id", "horizon"]).agg(
        n_origins=("n", "first"), n_seeds=("seed", "nunique"),
        rmse=("rmse", "mean"), mae=("mae", "mean"), dir_acc=("dir_acc", "mean"),
    ).reset_index()
    return out


def relative_to_reference(by_market: pd.DataFrame, reference: str, min_origins: int = 8) -> pd.DataFrame:
    """Add rel_rmse / rel_mae = model metric / reference metric in the same market and horizon.

    Market-horizon cells with fewer than `min_origins` forecasts are dropped: a ratio
    from a handful of points is mostly noise.
    """
    if reference not in set(by_market["model"]):
        raise ValueError(f"reference model {reference!r} is not among the predictions")
    ref = by_market[by_market["model"] == reference][["series_id", "horizon", "rmse", "mae"]]
    rel = by_market.merge(ref, on=["series_id", "horizon"], suffixes=("", "_ref"))
    rel = rel[rel["n_origins"] >= min_origins].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel["rel_rmse"] = rel["rmse"] / rel["rmse_ref"]
        rel["rel_mae"] = rel["mae"] / rel["mae_ref"]
    rel = rel[np.isfinite(rel["rel_rmse"])]
    return rel.drop(columns=["rmse_ref", "mae_ref"])


def _summary(rel: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = rel.groupby(keys)
    return g.agg(
        n_markets=("series_id", "nunique"),
        median_rel_rmse=("rel_rmse", "median"),
        gmean_rel_rmse=("rel_rmse", lambda x: float(np.exp(np.log(x).mean()))),
        share_beats_reference=("rel_rmse", lambda x: float((x < 1).mean())),
        median_rel_mae=("rel_mae", "median"),
    ).reset_index()


def summary_by_horizon(rel: pd.DataFrame) -> pd.DataFrame:
    """One row per model and horizon, summarised across markets (the error-by-horizon curve)."""
    return _summary(rel, ["model", "horizon"])


def history_groups(panel: pd.DataFrame, value_col: str = "hpi",
                   bins=("1979Q4", "1999Q4")) -> pd.DataFrame:
    """HPI start quarter per series and a coarse history-length group."""
    start = (panel.dropna(subset=[value_col]).assign(q=lambda d: pd.PeriodIndex(d["quarter"], freq="Q"))
             .groupby("series_id")["q"].min())
    b0, b1 = pd.Period(bins[0], "Q"), pd.Period(bins[1], "Q")
    group = np.where(start <= b0, f"long (by {b0.year})",
                     np.where(start <= b1, f"medium (by {b1.year})", f"short (after {b1.year})"))
    return pd.DataFrame({"series_id": start.index, "hpi_start": start.astype(str).to_numpy(),
                         "history": group})


def summary_by_history(rel: pd.DataFrame, groups: pd.DataFrame,
                       horizon_bands=((1, 2), (3, 6), (7, 15), (16, 20))) -> pd.DataFrame:
    """Does the model do better in markets with longer histories? Summarised per horizon band."""
    r = rel.merge(groups, on="series_id", how="left")
    bands = pd.Series(pd.NA, index=r.index, dtype="object")
    for lo, hi in horizon_bands:
        bands[(r["horizon"] >= lo) & (r["horizon"] <= hi)] = f"h{lo}-{hi}"
    r = r.assign(horizon_band=bands).dropna(subset=["horizon_band"])
    return _summary(r, ["model", "history", "horizon_band"])
