"""Evaluate each market and horizon, then summarize equally weighted market ratios.

Average metrics over seeds before taking market medians. Use paired forecast-date
blocks for uncertainty intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_forecast.data.contract import check_alignment
from housing_forecast.evaluation.bootstrap import circular_block_indices


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


def _squared_errors(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = preds[preds["model"] == model]
    keys = ["series_id", "origin_quarter", "horizon"]
    return (rows.assign(se=(rows["y_pred"] - rows["y_true"]) ** 2)
            .pivot(index=keys, columns="seed", values="se").sort_index())


def _sum_by_origin(errors: pd.DataFrame, markets: pd.Index, calendar: pd.PeriodIndex):
    market = markets.get_indexer(errors.index.get_level_values("series_id"))
    origin = calendar.get_indexer(pd.PeriodIndex(errors.index.get_level_values("origin_quarter"), freq="Q"))
    totals = np.zeros((len(markets), len(calendar), errors.shape[1]))
    np.add.at(totals, (market, origin), errors.to_numpy())
    return totals, market, origin


def _ratios(sse_a, sse_b, counts, weights):
    n = (weights @ counts.T)[:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rmse_a = np.sqrt(np.einsum("do,mos->dms", weights, sse_a) / n).mean(axis=2)
        rmse_b = np.sqrt(np.einsum("do,mos->dms", weights, sse_b) / n).mean(axis=2)
        ratios = rmse_a / rmse_b
    return np.where(np.isfinite(ratios), ratios, np.nan)


def paired_ratio_bootstrap(preds: pd.DataFrame, model_a: str, model_b: str, cfg: dict,
                           min_origins: int = 8, block_length: int | None = None,
                           horizon_bands=((3, 6), (16, 20))) -> pd.DataFrame:
    """Bootstrap the median market RMSE ratio, with RMSE averaged over seeds.

    Resample the same blocks of calendar quarters for both models and all markets.
    Within a band, pool the squared errors across its horizons before taking RMSE.
    The interval describes variation over forecast dates for these markets and seeds.
    """
    compared = preds[preds["model"].isin([model_a, model_b])]
    if not {model_a, model_b} <= set(compared["model"]):
        raise ValueError(f"both {model_a!r} and {model_b!r} must be among the predictions")
    check_alignment(compared)
    settings = cfg["bootstrap"]
    block = settings["block_length"] if block_length is None else block_length
    if block < 1 or min_origins < 1 or settings["n_boot"] < 1:
        raise ValueError("block_length, min_origins and n_boot must be positive")

    a = _squared_errors(compared, model_a)
    b = _squared_errors(compared, model_b)
    horizons = a.index.get_level_values("horizon")
    quarters = pd.PeriodIndex(a.index.get_level_values("origin_quarter"), freq="Q")
    calendar = pd.period_range(quarters.min(), quarters.max(), freq="Q")
    if block >= len(calendar):
        raise ValueError("block_length must be shorter than the forecast calendar")
    groups = [(str(h), (h, h)) for h in sorted(horizons.unique())]
    groups += [(f"h{lo}-{hi}", (lo, hi)) for lo, hi in horizon_bands]
    rng = np.random.default_rng(settings["seed"])
    draws = circular_block_indices(len(calendar), block, settings["n_boot"], rng)
    weights = np.stack([np.bincount(draw, minlength=len(calendar)) for draw in draws])
    tail = (1 - settings["ci"]) / 2

    results = []
    for label, (lo, hi) in groups:
        in_band = (horizons >= lo) & (horizons <= hi)
        keys = a.index[in_band].to_frame(index=False)
        coverage = keys.groupby("series_id")["origin_quarter"].nunique()
        markets = coverage.index[coverage >= min_origins]
        if markets.empty:
            continue
        keep = in_band & a.index.get_level_values("series_id").isin(markets)
        sse_a, market, origin = _sum_by_origin(a.loc[keep], markets, calendar)
        sse_b, _, _ = _sum_by_origin(b.loc[keep], markets, calendar)
        counts = np.zeros((len(markets), len(calendar)))
        np.add.at(counts, (market, origin), 1)

        point = _ratios(sse_a, sse_b, counts, np.ones((1, len(calendar))))[0]
        valid_markets = np.isfinite(point)
        if not valid_markets.any():
            continue
        boot_ratios = _ratios(sse_a[valid_markets], sse_b[valid_markets], counts[valid_markets], weights)
        valid_draws = np.isfinite(boot_ratios).any(axis=1)
        boot = np.nanmedian(boot_ratios[valid_draws], axis=1)
        low, high = np.quantile(boot, [tail, 1 - tail]) if boot.size else (np.nan, np.nan)
        results.append({
            "model_a": model_a, "model_b": model_b, "horizon": label,
            "n_markets": int(valid_markets.sum()),
            "n_origins": int(np.count_nonzero(counts[valid_markets].sum(axis=0))),
            "median_ratio": float(np.median(point[valid_markets])), "ci_low": float(low), "ci_high": float(high),
            "share_draws_below_1": float((boot < 1).mean()) if boot.size else np.nan,
            "share_markets_a_better": float((point[valid_markets] < 1).mean()),
            "block_length": block, "n_boot": int(settings["n_boot"]), "n_valid_draws": int(boot.size),
        })
    return pd.DataFrame(results, columns=[
        "model_a", "model_b", "horizon", "n_markets", "n_origins", "median_ratio", "ci_low", "ci_high",
        "share_draws_below_1", "share_markets_a_better", "block_length", "n_boot", "n_valid_draws",
    ])
