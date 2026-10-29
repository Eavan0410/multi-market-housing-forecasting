"""Reader for B's Taiwan raw files (data/raw/taiwan/*.csv).

Each file has the columns
    country, REF_AREA, feature, FREQ, TIME_PERIOD, OBS_VALUE, UNIT_MEASURE, ADJUSTMENT, OBS_STATUS
with TIME_PERIOD "1982Q2" (quarterly) or "Jul88" (monthly). Files are found by
their `feature` value, not by file name.

Output: one row per quarter for Taiwan with the shared feature names
    interest_rate         10-year government bond yield, quarterly mean of monthly values
    gdp_growth            real GDP growth, q/q, seasonally adjusted (same basis as OECD)
    household_credit_gdp  consumer loan balance (end of quarter) / sum of the last
                          4 quarters of nominal GDP * 100 (same basis as BIS % of GDP)
The two inputs of the credit ratio are kept as well, for checking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing_forecast.data.transforms import credit_to_gdp, monthly_to_quarterly

# raw `feature` value -> (output column, how monthly values become quarterly)
TAIWAN_FEATURES = {
    "interest_rate": ("interest_rate", "mean"),
    "gdp_growth_qoq": ("gdp_growth", None),
    "consumer_credit_balance": ("consumer_credit_balance", "last"),   # stock: end-of-quarter value
    "nominal_gdp": ("nominal_gdp", None),
}


def parse_period(s: pd.Series) -> pd.Series:
    """'1982Q2' -> quarterly Period, 'Jul88' -> monthly Period."""
    s = s.astype(str).str.strip()
    is_q = s.str.match(r"^\d{4}Q[1-4]$")
    out = pd.Series(index=s.index, dtype=object)
    out[is_q] = list(pd.PeriodIndex(s[is_q], freq="Q"))
    out[~is_q] = list(pd.PeriodIndex(pd.to_datetime(s[~is_q], format="%b%y"), freq="M"))
    return out


def load_taiwan_raw(taiwan_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(taiwan_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no csv files in {taiwan_dir}")
    # Only B's long-format files (with a `feature` column); the HPI and CPI tables
    # copied from Spring 2026 Raw sit in the same folder and are read elsewhere.
    tables = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    tables = [t for t in tables if "feature" in t.columns]
    if not tables:
        raise FileNotFoundError(f"no long-format files with a 'feature' column in {taiwan_dir}")
    raw = pd.concat(tables, ignore_index=True)
    raw = raw[raw["feature"].isin(TAIWAN_FEATURES)].dropna(subset=["OBS_VALUE"])

    series = []
    for feat, g in raw.groupby("feature"):
        col, how = TAIWAN_FEATURES[feat]
        periods = parse_period(g["TIME_PERIOD"])
        if (g["FREQ"] == "M").all():
            monthly = pd.DataFrame({"country": "Taiwan", "month": [str(p) for p in periods],
                                    col: g["OBS_VALUE"].to_numpy()})
            q = monthly_to_quarterly(monthly, col, how=how)
        else:
            q = pd.DataFrame({"country": "Taiwan", "quarter": [str(p) for p in periods],
                              col: g["OBS_VALUE"].to_numpy()})
        series.append(q.set_index(["country", "quarter"])[col])

    out = pd.concat(series, axis=1).reset_index()
    out = credit_to_gdp(out, credit_col="consumer_credit_balance", gdp_col="nominal_gdp",
                        out_col="household_credit_gdp")
    return out.sort_values("quarter").reset_index(drop=True)
