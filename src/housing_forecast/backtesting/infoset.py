"""Build input windows and labels using only observations available at each origin.

Inputs follow the forecast date. Label availability follows each market's HPI
release lag, even when all markets share a forecast date."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from housing_forecast.backtesting.labels import make_labels
from housing_forecast.config import forecast_lag_days, lag_days
from housing_forecast.quarters import quarter_end, to_period


@dataclass(frozen=True)
class FeatureSpec:
    name: str        # column in the prepared panel, e.g. "hpi_qoq"
    source: str      # variable whose release lag applies, e.g. "hpi", "cpi"


DEFAULT_FEATURES = (FeatureSpec("hpi_qoq", "hpi"),)


def features_from_config(cfg: dict) -> tuple[FeatureSpec, ...]:
    """configs [features].use -> specs. hpi_* features take the HPI release lag;
    a derived feature "<var>_d4" (4-quarter change) takes the lag of <var>;
    any other feature takes the lag listed under its own name."""
    names = cfg["features"]["use"]
    if "hpi_qoq" not in names:
        raise ValueError("hpi_qoq (the target series' own past prices) must stay in the features")

    def source(n: str) -> str:
        if n.startswith("hpi"):
            return "hpi"
        return n[:-3] if n.endswith("_d4") else n

    return tuple(FeatureSpec(n, source(n)) for n in names)


@dataclass
class SampleSet:
    X: np.ndarray
    y: np.ndarray
    y_avail: np.ndarray            # availability date of each label (datetime64), NaT if no label
    meta: pd.DataFrame             # series_id, country, geography_level, origin (Period), as_of_date
    feature_names: list[str]
    horizons: tuple[int, ...]
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, mask) -> "SampleSet":
        mask = np.asarray(mask)
        return SampleSet(
            X=self.X[mask], y=self.y[mask], y_avail=self.y_avail[mask],
            meta=self.meta[mask].reset_index(drop=True),
            feature_names=list(self.feature_names), horizons=self.horizons, extra=dict(self.extra),
        )

    def replace(self, **kw) -> "SampleSet":
        return replace(self, **kw)


def delay_quarters(var_lag_days: int, ref_lag_days: int, years=range(1960, 2041)) -> int:
    """Smallest k >= 0 with  end(q-k) + var_lag <= end(q) + ref_lag  for every quarter q."""
    qs = pd.period_range(f"{years.start}Q1", f"{years.stop - 1}Q4", freq="Q")
    ref = quarter_end(qs) + pd.Timedelta(days=ref_lag_days)
    k = 0
    while True:
        avail = quarter_end(qs - k) + pd.Timedelta(days=var_lag_days)
        if (avail <= ref).all():
            return k
        k += 1


def build_samples(panel: pd.DataFrame, cfg: dict, features=DEFAULT_FEATURES) -> SampleSet:
    """Prepared panel (with hpi_real and feature columns) -> SampleSet for all series and origins."""
    lookback = cfg["features"]["lookback"]
    ffill_limit = cfg["features"]["ffill_limit"]
    horizons = tuple(cfg["target"]["horizons"])
    value_col = cfg["target"]["value_col"]
    names = [f.name for f in features]

    common = forecast_lag_days(cfg)
    labels = make_labels(panel, value_col=value_col, horizons=horizons)
    labels["q"] = to_period(labels["origin_quarter"])

    windows, metas = [], []
    for sid, g in panel.groupby("series_id", sort=True):
        country, level = g["country"].iloc[0], g["geography_level"].iloc[0]
        hpi_lag = lag_days(cfg, "hpi", country)
        ref_lag = hpi_lag if common is None else common

        g = g.assign(q=to_period(g["quarter"])).set_index("q").sort_index()
        full = pd.period_range(g.index.min(), g.index.max(), freq="Q")
        g = g.reindex(full)                                  # calendar-complete: gaps become NaN rows

        cols = []
        for f in features:
            k = delay_quarters(lag_days(cfg, f.source, country), ref_lag)
            s = g[f.name].ffill(limit=ffill_limit) if ffill_limit else g[f.name]
            cols.append(s.shift(k).to_numpy())
        F = np.column_stack(cols).astype(float)

        # Origins must have a published HPI_t (the label base).
        has_hpi = g[value_col].notna().to_numpy()
        T = len(full)
        for i in range(lookback - 1, T):
            if not has_hpi[i]:
                continue
            window = F[i - lookback + 1: i + 1]
            if cfg["features"]["drop_nan_windows"] and np.isnan(window).any():
                continue
            windows.append(window)
            metas.append((sid, country, level, full[i], ref_lag, hpi_lag))

    meta = pd.DataFrame(metas, columns=["series_id", "country", "geography_level", "origin", "_ref", "_hpi"])
    meta["as_of_date"] = quarter_end(pd.PeriodIndex(meta["origin"], freq="Q")) + pd.to_timedelta(meta["_ref"], unit="D")

    # Attach labels (NaN where missing) and the date each label becomes known.
    wide = labels.pivot_table(index=["series_id", "q"], columns="horizon", values="y_true")
    wide = wide.reindex(columns=list(horizons))
    key = pd.MultiIndex.from_arrays([meta["series_id"], pd.PeriodIndex(meta["origin"], freq="Q")])
    y = wide.reindex(key).to_numpy(dtype=float)
    y_avail = np.empty(y.shape, dtype="datetime64[ns]")
    origin_idx = pd.PeriodIndex(meta["origin"], freq="Q")
    for j, h in enumerate(horizons):
        dates = quarter_end(origin_idx + h) + pd.to_timedelta(meta["_hpi"], unit="D")
        y_avail[:, j] = np.where(np.isnan(y[:, j]), np.datetime64("NaT", "ns"), dates.to_numpy())

    meta = meta.drop(columns=["_ref", "_hpi"])
    X = np.stack(windows) if windows else np.empty((0, lookback, len(names)))
    return SampleSet(X=X, y=y, y_avail=y_avail, meta=meta, feature_names=names, horizons=horizons)
