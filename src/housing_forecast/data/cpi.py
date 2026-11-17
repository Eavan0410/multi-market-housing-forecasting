"""National CPI inflation (year-on-year, percent) for the forecast markets.

Sources, in order of priority for each country:
  1. data/raw/global/cpi_gaps.csv        (optional; written by scripts/download_cpi_gaps.py)
  2. data/raw/global/CPI_48countries.csv (OECD DF_PRICES_ALL export plus BIS long CPI rows)
  3. data/raw/four_countries/cpi_yoy_raw.csv (only for countries missing from 1 and 2: Japan)
Taiwan comes from its monthly CPI index (data/raw/taiwan) and is handled in transforms.py.

Cleaning rules:
  - Rows copied from a secondary mirror of BIS data (STRUCTURE_ID contains "secondary")
    are dropped; the official BIS index covers the same months.
  - Published year-on-year rates (TRANSFORMATION "GY") are used first. Months or quarters
    without one are filled from the index (TRANSFORMATION "_Z"): 12-month change for
    monthly data, 4-quarter change for quarterly data.
  - Quarterly values: native quarterly rows first; otherwise the mean of the three months.
    Some OECD series (Australia, early Ireland) are quarterly but stored as one month per
    quarter (Feb, May, Aug, Nov). Such a middle month is used as the quarter's value when no
    quarter of that year has more than one observation. Other incomplete quarters are dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/global/CPI_48countries.csv")
GAPS = Path("data/raw/global/cpi_gaps.csv")
FALLBACK = Path("data/raw/four_countries/cpi_yoy_raw.csv")
NAME_MAP = {"Korea": "South Korea", "China (People’s Republic of)": "China"}


def _read_oecd_style(path: Path, source_label: str) -> pd.DataFrame:
    """Read an OECD/BIS-style export (Country or country; Period or TIME_PERIOD; Value or OBS_VALUE)."""
    d = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    d = d.rename(columns={"country": "Country", "TIME_PERIOD": "Period", "OBS_VALUE": "Value"})
    if "STRUCTURE_ID" in d:
        d = d[~d["STRUCTURE_ID"].fillna("").str.contains("secondary")]
    src = d["STRUCTURE_ID"].fillna("") if "STRUCTURE_ID" in d else pd.Series("", index=d.index)
    out = pd.DataFrame({
        "country": d["Country"].replace(NAME_MAP),
        "freq": np.where(d["Period"].str.contains("Q"), "Q", "M"),
        "period": d["Period"].str.strip(),
        "transformation": d["TRANSFORMATION"].fillna("GY"),
        "value": pd.to_numeric(d["Value"], errors="coerce"),
        "source": np.where(src.str.startswith("BIS"), "BIS long CPI",
                           np.where(src.str.startswith("FRED"), "FRED (OECD MEI archive)", source_label)),
    })
    return out.dropna(subset=["value"])


def load_cpi_raw(raw: Path = RAW, gaps: Path = GAPS, fallback: Path = FALLBACK) -> pd.DataFrame:
    """Long table: country, freq, period, transformation, value, source (one source per country)."""
    main = _read_oecd_style(raw, "OECD")
    parts = [main]
    if gaps.exists():
        g = _read_oecd_style(gaps, "OECD")
        parts = [main[~main["country"].isin(g["country"])], g]
    have = set(pd.concat(parts)["country"])
    if fallback.exists():
        f = _read_oecd_style(fallback, "OECD (four_countries file)")
        parts.append(f[~f["country"].isin(have)])
    return pd.concat(parts, ignore_index=True)


def native_yoy(raw: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year series at native frequency: country, freq, period (pd.Period), cpi_yoy, from_index, source."""
    rows = []
    for (country, freq), g in raw.groupby(["country", "freq"]):
        idx = pd.PeriodIndex(g["period"].str.replace("-Q", "Q"), freq=freq)
        g = g.assign(p=idx)
        gy = g[g["transformation"].eq("GY")].groupby("p")["value"].mean()
        lvl = g[g["transformation"].eq("_Z")].groupby("p")["value"].mean()
        lag = 12 if freq == "M" else 4
        if len(lvl):
            full = lvl.reindex(pd.period_range(lvl.index.min(), lvl.index.max(), freq=freq))
            from_idx = ((full / full.shift(lag) - 1.0) * 100.0).dropna()
        else:
            from_idx = pd.Series(dtype=float, index=pd.PeriodIndex([], freq=freq))
        s = gy.combine_first(from_idx)
        rows.append(pd.DataFrame({
            "country": country, "freq": freq, "period": s.index, "cpi_yoy": s.to_numpy(),
            "from_index": ~s.index.isin(gy.index), "source": g["source"].iloc[0],
        }))
    return pd.concat(rows, ignore_index=True)


def quarterly_yoy(native: pd.DataFrame) -> pd.DataFrame:
    """country, quarter (str like 2005Q1), cpi_yoy."""
    q = native[native["freq"].eq("Q")]
    q = pd.DataFrame({"country": q["country"], "quarter": q["period"].astype(str), "cpi_yoy": q["cpi_yoy"]})

    m = native[native["freq"].eq("M")].copy()
    m["quarter"] = m["period"].apply(lambda p: p.asfreq("Q"))
    m["mid"] = m["period"].apply(lambda p: p.month % 3 == 2)           # Feb, May, Aug, Nov
    agg = (m.groupby(["country", "quarter"])
             .agg(mean=("cpi_yoy", "mean"), count=("cpi_yoy", "size"), mid=("mid", "all")).reset_index())
    agg["year"] = agg["quarter"].apply(lambda p: p.year)
    one_per_q = agg.groupby(["country", "year"])["count"].transform("max").eq(1) & agg["mid"]
    agg = agg[agg["count"].eq(3) | one_per_q]
    mq = pd.DataFrame({"country": agg["country"], "quarter": agg["quarter"].astype(str), "cpi_yoy": agg["mean"]})

    mq = mq.merge(q[["country", "quarter"]], on=["country", "quarter"], how="left", indicator=True)
    mq = mq[mq["_merge"].eq("left_only")].drop(columns="_merge")
    return pd.concat([q, mq], ignore_index=True).sort_values(["country", "quarter"]).reset_index(drop=True)


def attach_cpi_yoy(panel: pd.DataFrame, quarterly: pd.DataFrame, index_countries=("Taiwan",)) -> pd.DataFrame:
    """Replace the panel's cpi_yoy with this module's quarterly series, matched on country and quarter.

    Countries in `index_countries` (Taiwan) are left empty here: prepare_panel fills them from
    their monthly CPI index with the same 4-quarter definition. Rows without a CPI value stay
    NaN, so input windows that need them are dropped (drop_nan_windows).
    """
    out = panel.drop(columns=["cpi_yoy"], errors="ignore")
    q = quarterly[~quarterly["country"].isin(index_countries)]
    out = out.merge(q[["country", "quarter", "cpi_yoy"]], on=["country", "quarter"], how="left")
    return out
