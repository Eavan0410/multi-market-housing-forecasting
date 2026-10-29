"""Adapter: B's per-country processed files -> the shared panel contract.

B's files (data/processed/<ISO>_processed.csv) are stacked:
    country, feature, REF_AREA, FREQ, TIME_PERIOD, SOURCE_PERIOD, OBS_VALUE, ...
with FREQ "Q" (SOURCE_PERIOD "1970-Q1" or a quarter-start date "1977-10-01")
or "M" (SOURCE_PERIOD "1956-01").

Rules applied here (row-wise, no fitting, so safe outside the backtest window):
  - native quarterly rows are used as they are;
  - monthly rows -> quarterly mean, only for quarters with all 3 months;
  - if a quarter has both, the native quarterly value wins;
  - no filling, no interpolation, missing stays missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing_forecast.data.raw_dev import load_anahpi, load_taiwan_cpi_monthly, load_taiwan_hpi

B_FEATURES = [
    "hpi", "cpi_yoy", "gdp_growth", "household_credit_gdp",
    "interest_rate", "unemployment_rate",
]
# real_income_growth is dropped by team decision (starts 2007Q2, missing for Korea and Taiwan).
# One spelling per country across all sources.
COUNTRY_ALIASES = {"Korea": "South Korea"}


def _to_quarter(period: pd.Series) -> pd.Series:
    s = period.astype(str)
    is_q = s.str.contains("Q")
    out = pd.Series(index=s.index, dtype=object)
    out[is_q] = s[is_q].str.replace("-", "", regex=False)
    out[~is_q] = pd.PeriodIndex(pd.to_datetime(s[~is_q], format="mixed"), freq="Q").astype(str).to_numpy()
    return out


def load_b_processed(processed_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(processed_dir).glob("*_processed.csv"))
    if not files:
        raise FileNotFoundError(f"no *_processed.csv in {processed_dir}")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["quarter"] = _to_quarter(raw["SOURCE_PERIOD"])
    raw["country"] = raw["country"].replace(COUNTRY_ALIASES)

    q = (raw[raw["FREQ"] == "Q"]
         .groupby(["country", "feature", "quarter"])["OBS_VALUE"].mean().rename("value"))
    m = raw[raw["FREQ"] == "M"].groupby(["country", "feature", "quarter"])["OBS_VALUE"]
    m = m.mean().where(m.count() >= 3).dropna().rename("value")
    # Native quarterly first; monthly fills only quarters that have no quarterly value.
    both = pd.concat([q, m[~m.index.isin(q.index)]]).reset_index()

    panel = both.pivot_table(index=["country", "quarter"], columns="feature", values="value").reset_index()
    panel.columns.name = None
    panel.insert(0, "series_id", panel["country"])
    panel.insert(2, "geography_level", "national")
    for f in B_FEATURES:
        if f not in panel.columns:
            panel[f] = float("nan")
    return panel[["series_id", "country", "geography_level", "quarter"] + B_FEATURES]


def build_panel(b_processed_dir: str | Path, spring_raw_dir: str | Path,
                add_anahpi_countries: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined panel for the backtest.

    - B's countries: all their features (B's data takes priority)
    - other AnaHPI countries: real HPI only (larger HPI-only pretraining pool)
    - Taiwan national + 7 cities: nominal HPI from Spring 2026 Raw (until B delivers Taiwan)
    Returns (panel, taiwan_cpi_monthly).
    """
    spring = Path(spring_raw_dir)
    b = load_b_processed(b_processed_dir)
    parts = [b]
    if add_anahpi_countries:
        ana = load_anahpi(spring / "Global/Raw/AnaHPI_quarter_data.csv")
        ana["country"] = ana["country"].replace(COUNTRY_ALIASES)
        ana["series_id"] = ana["country"]
        parts.append(ana[~ana["country"].isin(set(b["country"]))])
    parts.append(load_taiwan_hpi(spring / "Taiwan/Raw/full_HPI_quarterly(city+taiwan).csv"))
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["series_id", "quarter"]).reset_index(drop=True)
    cpi = load_taiwan_cpi_monthly(spring / "Taiwan/Raw/CPI_monthly.csv")
    return panel, cpi
