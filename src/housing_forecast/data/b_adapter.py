"""Combine country data into the quarterly panel.

Use native quarterly observations first, then means of complete monthly
quarters. Preserve Taiwan macro history before HPI begins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing_forecast.data.raw_dev import load_anahpi, load_taiwan_cpi_monthly, load_taiwan_hpi
from housing_forecast.data.taiwan_raw import load_taiwan_raw

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


def attach_national_macro(hpi_rows: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Give every series of a country (national + cities) the country's macro features.

    Uses an OUTER join on quarter, so macro history from before the HPI starts is
    kept. Changes such as the 4-quarter change of credit / GDP are computed later
    on the full history; a left join on the HPI quarters would cut off the
    earlier quarters they need (Taiwan HPI starts 2001Q1, so 2001's
    4-quarter change needs 2000's ratio).
    """
    cols = [c for c in B_FEATURES if c != "hpi" and c in macro.columns]
    out = []
    for sid, g in hpi_rows.groupby("series_id"):
        country, level = g["country"].iloc[0], g["geography_level"].iloc[0]
        m = macro.loc[macro["country"] == country, ["quarter"] + cols]
        merged = g.drop(columns=[c for c in cols if c in g.columns]).merge(m, on="quarter", how="outer")
        merged[["series_id", "country", "geography_level"]] = [sid, country, level]
        out.append(merged)
    return pd.concat(out, ignore_index=True).sort_values(["series_id", "quarter"]).reset_index(drop=True)


TAIWAN_HPI_FILE = "taiwan_hpi_quarterly_raw.csv"     # Spring 2026 Raw full_HPI_quarterly(city+taiwan).csv
TAIWAN_CPI_FILE = "taiwan_cpi_monthly_raw.csv"       # Spring 2026 Raw CPI_monthly.csv


def build_panel(b_processed_dir: str | Path, anahpi_path: str | Path | None = "data/raw/global/anahpi_quarterly_raw.csv",
                add_anahpi_countries: bool = True,
                taiwan_dir: str | Path = "data/raw/taiwan") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined panel for the backtest.

    All inputs are in the repository:
    - B's 4 countries: all their features (data/processed/)
    - Taiwan national + 7 cities: nominal HPI and CPI index (copies of the Spring
      2026 Raw tables) and B's macro files, all in `taiwan_dir`. Macro series are
      national, so the 7 cities share them.
    - other AnaHPI countries: real HPI only (larger HPI-only pretraining pool),
      from `anahpi_path` (copy of Spring 2026 Raw AnaHPI_quarter_data.csv).
      Skipped with a message if the file is missing.
    Returns (panel, taiwan_cpi_monthly).
    """
    taiwan_dir = Path(taiwan_dir)
    b = load_b_processed(b_processed_dir)
    parts = [b]
    if add_anahpi_countries:
        ana_path = Path(anahpi_path) if anahpi_path else None
        if ana_path is not None and ana_path.exists():
            ana = load_anahpi(ana_path)
            ana["country"] = ana["country"].replace(COUNTRY_ALIASES)
            ana["series_id"] = ana["country"]
            parts.append(ana[~ana["country"].isin(set(b["country"]))])
        else:
            print(f"note: {ana_path} not found; extra AnaHPI countries skipped (HPI-only pool is smaller)")

    taiwan = load_taiwan_hpi(taiwan_dir / TAIWAN_HPI_FILE)
    taiwan = attach_national_macro(taiwan, load_taiwan_raw(taiwan_dir))
    parts.append(taiwan)

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["series_id", "quarter"]).reset_index(drop=True)
    cpi = load_taiwan_cpi_monthly(taiwan_dir / TAIWAN_CPI_FILE)
    return panel, cpi
