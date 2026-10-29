"""Readers for raw tables stored as copies of the Spring 2026 files.

load_anahpi             OECD AnaHPI (real house price index) -> long panel rows
load_taiwan_hpi         Taiwan national + 7 city HPI table -> long panel rows
load_taiwan_cpi_monthly Taiwan CPI table (Year x month) -> country, month, cpi

Raw levels only: no filling, scaling or growth rates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Regional aggregates in OECD files are not countries.
AGGREGATES = {
    "Euro area", "Euro area (17 countries)", "Euro area (19 countries)", "Euro area (20 countries)",
    "OECD", "OECD - Total", "European Union", "European Union (27 countries)",
    "APEC", "ASEAN", "Africa", "USMCA", "G7", "G20",
}
COUNTRY_RENAME = {
    "China (People’s Republic of)": "China",
    "China (People's Republic of)": "China",
    "People's Republic of China": "China",
}
TAIWAN_COLUMNS = {
    "台灣": ("Taiwan_national", "national"),
    "台北": ("Taiwan_Taipei", "city"),
    "新北": ("Taiwan_NewTaipei", "city"),
    "桃園": ("Taiwan_Taoyuan", "city"),
    "新竹": ("Taiwan_Hsinchu", "city"),
    "台中": ("Taiwan_Taichung", "city"),
    "台南": ("Taiwan_Tainan", "city"),
    "高雄": ("Taiwan_Kaohsiung", "city"),
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]


def load_anahpi(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw[raw["MEASURE"] == "RHP"]                       # real house price index
    country = raw["Country"].replace(COUNTRY_RENAME)
    out = pd.DataFrame({
        "series_id": country,
        "country": country,
        "geography_level": "national",
        "quarter": raw["Period"].str.replace("-", "", regex=False),   # 1960-Q1 -> 1960Q1
        "hpi": pd.to_numeric(raw["Value"], errors="coerce"),
    })
    out = out[~out["country"].isin(AGGREGATES)]
    # If two spellings of one country both exist, keep the first non-empty value.
    out = out.groupby(["series_id", "country", "geography_level", "quarter"], as_index=False)["hpi"].first()
    return out


def load_taiwan_hpi(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    quarter = raw["Period"].str.replace("/", "", regex=False)       # 2001/Q1 -> 2001Q1
    rows = []
    for col, (sid, level) in TAIWAN_COLUMNS.items():
        rows.append(pd.DataFrame({
            "series_id": sid, "country": "Taiwan", "geography_level": level,
            "quarter": quarter, "hpi": pd.to_numeric(raw[col], errors="coerce"),
        }))
    return pd.concat(rows, ignore_index=True)


def load_taiwan_cpi_monthly(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    long = raw.melt(id_vars="Year", value_vars=MONTHS, var_name="m", value_name="cpi")
    long["month"] = long["Year"].astype(str) + "-" + (long["m"].map(MONTHS.index) + 1).astype(str).str.zfill(2)
    long["country"] = "Taiwan"
    long = long.dropna(subset=["cpi"])
    return long[["country", "month", "cpi"]].sort_values("month").reset_index(drop=True)
