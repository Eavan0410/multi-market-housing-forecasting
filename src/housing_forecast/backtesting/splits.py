"""Expanding-window folds with a shared calendar cutoff and a locked test period.

Training origins respect the configured gap; labels must be public by the
fold cutoff. Scaling and model fitting are handled inside each fold by runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from housing_forecast.backtesting.infoset import SampleSet
from housing_forecast.config import lag_days
from housing_forecast.quarters import quarter_end


class PeriodLockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Split:
    split_id: str
    phase: str                      # "validation" or "test"
    test_origins: tuple             # tuple of pd.Period
    train_origin_end: pd.Period     # last origin allowed in training
    cutoff_date: pd.Timestamp       # as_of_date of the first test origin (information cutoff)

    @property
    def first_test_origin(self) -> pd.Period:
        return self.test_origins[0]


def last_label_origin(samples: SampleSet, series_id: str) -> pd.Period:
    m = (samples.meta["series_id"] == series_id).to_numpy() & ~np.isnan(samples.y).all(axis=1)
    if not m.any():
        raise ValueError(f"no labelled origins for {series_id}")
    return max(samples.meta.loc[m, "origin"])


def make_splits(cfg: dict, phase: str, last_origin: pd.Period, allow_test: bool = False) -> list[Split]:
    p = cfg["periods"]
    if phase == "validation":
        start, end = pd.Period(p["validation_start"], "Q"), pd.Period(p["validation_end"], "Q")
    elif phase == "test":
        if not allow_test:
            raise PeriodLockedError(
                "The test period is locked until the configuration is frozen (11/19). "
                "Pass allow_test=True / --unlock-test only for the final run."
            )
        start = pd.Period(p["test_start"], "Q")
        end = last_origin if p["test_end"] == "auto" else pd.Period(p["test_end"], "Q")
    else:
        raise ValueError("phase must be 'validation' or 'test'")
    end = min(end, last_origin)

    hpi_lag = lag_days(cfg, "hpi", cfg["target"]["country"])
    gap, step = p["gap_quarters"], p["refit_every"]

    origins = list(pd.period_range(start, end, freq="Q"))
    splits = []
    for i in range(0, len(origins), step):
        block = tuple(origins[i: i + step])
        first = block[0]
        cutoff = quarter_end(pd.PeriodIndex([first], freq="Q"))[0] + pd.Timedelta(days=hpi_lag)
        splits.append(Split(
            split_id=f"{phase[:3]}_{first}",
            phase=phase,
            test_origins=block,
            train_origin_end=first - gap,
            cutoff_date=cutoff,
        ))
    return splits


def train_mask(samples: SampleSet, split: Split) -> np.ndarray:
    """Rows allowed in training: origin <= train_origin_end (all series, same calendar date)."""
    return (samples.meta["origin"] <= split.train_origin_end).to_numpy()


def mask_unpublished_labels(train: SampleSet, split: Split) -> SampleSet:
    """Hide any label not yet published at the cutoff date; drop rows left with no label."""
    y = train.y.copy()
    unpublished = ~(train.y_avail <= np.datetime64(split.cutoff_date))   # NaT compares False -> hidden
    y[unpublished] = np.nan
    keep = ~np.isnan(y).all(axis=1)
    return train.replace(y=y).subset(keep)


def eval_mask(samples: SampleSet, split: Split, eval_series) -> np.ndarray:
    in_block = samples.meta["origin"].isin(split.test_origins).to_numpy()
    in_series = samples.meta["series_id"].isin(eval_series).to_numpy()
    return in_block & in_series


def check_no_leakage(train: SampleSet, test: SampleSet, split: Split, gap: int) -> None:
    """Hard assertions run on every fold. Any failure stops the backtest."""
    if len(train) == 0:
        raise AssertionError(f"{split.split_id}: empty training set")
    tr_origin = pd.PeriodIndex(train.meta["origin"], freq="Q")
    te_origin = pd.PeriodIndex(test.meta["origin"], freq="Q")
    if (tr_origin > split.train_origin_end).any():
        raise AssertionError(f"{split.split_id}: training origin after train_origin_end")
    if len(test) and (te_origin.min() - tr_origin.max()).n < gap:
        raise AssertionError(f"{split.split_id}: gap between train and test origins < {gap}")
    # Every visible training label must have been published by the cutoff date.
    visible = ~np.isnan(train.y)
    if (train.y_avail[visible] > np.datetime64(split.cutoff_date)).any():
        raise AssertionError(f"{split.split_id}: training label published after the cutoff date")
    # And no visible training target may fall after the first test origin.
    for j, h in enumerate(train.horizons):
        tq = tr_origin[visible[:, j]] + h
        if len(tq) and tq.max() > split.first_test_origin:
            raise AssertionError(f"{split.split_id}: training target {tq.max()} overlaps test period")
    if len(test) and not set(te_origin).issubset(set(split.test_origins)):
        raise AssertionError(f"{split.split_id}: test origin outside the fold")
