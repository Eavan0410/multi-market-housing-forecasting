"""Load quarterly CPI inflation from OECD, BIS, FRED and Taiwan.

Prefer gap-file observations, then the main export and four-country fallback.
Retain native quarterly data and require complete monthly quarters."""

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
    """Replace cpi_yoy by country and quarter; leave missing observations as NaN.

    Taiwan inflation is derived from CPI index levels in prepare_panel."""
    out = panel.drop(columns=["cpi_yoy"], errors="ignore")
    q = quarterly[~quarterly["country"].isin(index_countries)]
    out = out.merge(q[["country", "quarter", "cpi_yoy"]], on=["country", "quarter"], how="left")
    return out
