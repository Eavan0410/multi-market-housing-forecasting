"""Combine country data into the quarterly panel.

Use native quarterly observations first, then means of complete monthly
quarters. Preserve Taiwan macro history before HPI begins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing_forecast.data.readers import load_anahpi, load_taiwan_cpi_monthly, load_taiwan_hpi
from housing_forecast.data.taiwan_raw import load_taiwan_raw

B_FEATURES = [
    "hpi", "cpi_yoy", "gdp_growth", "household_credit_gdp",
    "interest_rate", "unemployment_rate",
]
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
    """Attach national macro observations to country series. Preserve macro history
    before HPI begins, including markets with no earlier price observations."""
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
    """Combine four-country observations, Taiwan raw data and optional OECD AnaHPI.
    Preserve missing values; transformations are applied by the backtest."""
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
