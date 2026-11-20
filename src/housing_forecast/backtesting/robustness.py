"""Input-noise robustness: train each model once per fold and seed, as in run_backtest,
then forecast the same test rows again with Gaussian noise added to the inputs.

Noise design
  - One perturbation per market x input quarter x draw: the same historical value is
    "measured wrong" once. Input windows overlap (the window for origin 2004Q4 repeats seven
    quarters of the 2004Q3 window), and the same quarter can be an input in two folds; it gets
    the same error in every window, fold and model within a draw.
  - perturbation = level x sd x z, where z ~ N(0, 1) is fixed by (noise seed, draw, market,
    quarter) and sd is one fixed scale: the standard deviation of the feature in the FIRST
    fold's training inputs (all training markets pooled; data before the validation period
    only). The same z is used at every level, so the curves are smooth in the level.
  - Noise is added to the raw (unscaled) feature before the model's own scaling. Training
    data are never perturbed.

The output is a compact error table, enough to rebuild the main experiment's RMSE (pooled
squared errors, square root per seed, then the mean over seeds) and the mean absolute change
of the forecasts relative to the same model's clean forecast.
"""

from __future__ import annotations

import zlib
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

from housing_forecast.backtesting.infoset import SampleSet
from housing_forecast.backtesting.runner import training_samples
from housing_forecast.backtesting.scaling import WindowStandardizer
from housing_forecast.backtesting.splits import Split, check_no_leakage, eval_mask
from housing_forecast.models.base import Forecaster, set_global_seed

BASE_QUARTER = pd.Period("1900Q1", "Q").ordinal
N_QUARTERS = 800                                     # 1900Q1 to 2099Q4
ERROR_COLUMNS = ["model", "noise_level", "draw", "seed", "series_id", "horizon", "n", "sse", "sum_abs_change"]


@lru_cache(maxsize=None)
def _z(noise_seed: int, draw: int, series_id: str) -> np.ndarray:
    """Standard normal draws for one market, indexed by quarter ordinal - BASE_QUARTER."""
    rng = np.random.default_rng([noise_seed, draw, zlib.crc32(series_id.encode("utf-8"))])
    return rng.standard_normal(N_QUARTERS)


def noisy_inputs(test: SampleSet, feature: int, sd: float, level: float, draw: int,
                 noise_seed: int) -> np.ndarray:
    """Copy of test.X with level * sd * z added to one feature; z depends on market and quarter."""
    X = test.X.copy()
    if level == 0:
        return X
    L = X.shape[1]
    origins = pd.PeriodIndex(test.meta["origin"], freq="Q").asi8
    offsets = np.arange(-L + 1, 1)                               # window position i holds quarter t-L+1+i
    for r, (sid, o) in enumerate(zip(test.meta["series_id"], origins)):
        X[r, :, feature] += level * sd * _z(noise_seed, draw, sid)[o + offsets - BASE_QUARTER]
    return X


def _errors(test: SampleSet, y_pred: np.ndarray, y_clean: np.ndarray) -> pd.DataFrame:
    """Per market and horizon: label count, sum of squared errors, sum of |noisy - clean|."""
    rows = []
    sid = test.meta["series_id"].to_numpy()
    for j, h in enumerate(test.horizons):
        ok = ~np.isnan(test.y[:, j])
        d = pd.DataFrame({"series_id": sid[ok], "se": (y_pred[ok, j] - test.y[ok, j]) ** 2,
                          "change": np.abs(y_pred[ok, j] - y_clean[ok, j])})
        g = d.groupby("series_id").agg(n=("se", "size"), sse=("se", "sum"), sum_abs_change=("change", "sum"))
        rows.append(g.reset_index().assign(horizon=h))
    return pd.concat(rows, ignore_index=True)


def noise_backtest(
    samples: SampleSet,
    splits: list[Split],
    model_factory: Callable[[], Forecaster],
    cfg: dict,
    train_pool: Callable[[pd.DataFrame], np.ndarray],
    levels: list[float],
    draws: int,
    feature: str = "hpi_qoq",
    noise_seed: int = 2026,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Error table summed over folds: one row per noise level, draw, seed, market and horizon.
    The returned frame's attrs["noise_sd"] holds the fixed noise scale.

    Level 0 (no noise) is always included, with draw 0; it reproduces run_backtest.
    """
    j = samples.feature_names.index(feature)
    if not splits:
        raise ValueError("no folds")
    sd = float(np.nanstd(training_samples(samples, splits[0], train_pool).X[:, :, j]))
    eval_series = cfg["target"]["eval_series"]
    seeds = cfg["run"]["seeds"]
    gap = cfg["periods"]["gap_quarters"]
    variants = [(float(k), d) for k in levels if k > 0 for d in range(draws)]
    frames = []

    for split in splits:
        train = training_samples(samples, split, train_pool)
        test = samples.subset(eval_mask(samples, split, eval_series))
        if len(test) == 0:
            continue
        check_no_leakage(train, test, split, gap)
        test_X = {(0.0, 0): test.X} | {v: noisy_inputs(test, j, sd, v[0], v[1], noise_seed) for v in variants}

        probe = model_factory()
        if probe.needs_scaling:
            scaler = WindowStandardizer().fit(train.X)
            train = train.replace(X=scaler.transform(train.X))
            test_X = {v: scaler.transform(X) for v, X in test_X.items()}

        for seed in (seeds if probe.stochastic else seeds[:1]):
            set_global_seed(seed)
            model = model_factory()
            model.fit(train, seed)
            name = model_name or model.name
            clean = np.asarray(model.predict(test.replace(X=test_X[(0.0, 0)])), dtype=float)
            for (level, draw), X in test_X.items():
                y_pred = clean if level == 0 else np.asarray(model.predict(test.replace(X=X)), dtype=float)
                frames.append(_errors(test, y_pred, clean).assign(model=name, noise_level=level, draw=draw, seed=seed))

    out = pd.concat(frames, ignore_index=True)
    keys = ["model", "noise_level", "draw", "seed", "series_id", "horizon"]
    out = out.groupby(keys, as_index=False)[["n", "sse", "sum_abs_change"]].sum()[ERROR_COLUMNS]
    out.attrs["noise_sd"] = sd
    return out


def band_metrics(errors: pd.DataFrame, horizons, min_origins: int = 8) -> pd.DataFrame:
    """RMSE and mean |change| per model, level, draw and market for one horizon range.

    Same rule as the main experiment: pool squared errors across the range's horizons and
    forecast dates, take the square root per seed, then average over seeds. A market needs
    at least `min_origins` forecast dates at the range's shortest horizon.
    """
    hs = list(horizons)
    e = errors[errors["horizon"].isin(hs)]
    coverage = e[e["horizon"].eq(min(hs))].groupby(["model", "series_id"])["n"].max()   # rows are per seed
    per_seed = e.groupby(["model", "noise_level", "draw", "seed", "series_id"])[["n", "sse", "sum_abs_change"]].sum()
    per_seed["rmse"] = np.sqrt(per_seed["sse"] / per_seed["n"])
    m = per_seed.reset_index().groupby(["model", "noise_level", "draw", "series_id"]).agg(
        rmse=("rmse", "mean"), abs_change=("sum_abs_change", "sum"), n=("n", "sum")).reset_index()
    m["mean_abs_change"] = m["abs_change"] / m["n"]
    m = m.merge(coverage.rename("n_origins").reset_index(), on=["model", "series_id"])
    return m[m["n_origins"].ge(min_origins)].drop(columns=["abs_change", "n"])
