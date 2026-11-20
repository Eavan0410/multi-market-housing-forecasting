"""Quarterly policy actions from the IMF iMaPP MaPP sheet."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

import pandas as pd

ARCHIVE = Path("data/raw/imapp/iMaPP_database-2024-12-02.zip")
SOURCE_URL = "https://www.elibrary-areaer.imf.org/Macroprudential/Documents/iMaPP_database-2024-12-02.zip"
QUARTERLY = Path("data/interim/imapp_quarterly.csv")
INSTRUMENTS = ["LTV", "DSTI", "LoanR_HH"]
FEATURE = "policy_housing_flow"


def load_monthly(path: Path = ARCHIVE) -> pd.DataFrame:
    import openpyxl

    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [n for n in archive.namelist() if Path(n).name.startswith("iMaPP_database")
                     and n.endswith(".xlsx")]
            if len(names) != 1:
                raise ValueError("Expected one iMaPP database workbook in the archive")
            workbook = BytesIO(archive.read(names[0]))
    else:
        workbook = path
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        rows = book["MaPP"].iter_rows(values_only=True)
        columns = next(rows)
        selected = ["Country", "iso3", "Year", "Month", *INSTRUMENTS]
        positions = [columns.index(column) for column in selected]
        frame = pd.DataFrame(([row[i] for i in positions] for row in rows), columns=selected)
    finally:
        book.close()
    return frame.rename(columns={"Country": "source_country"})


def quarterly_actions(monthly: pd.DataFrame) -> pd.DataFrame:
    d = monthly.copy()
    dates = pd.to_datetime(dict(year=d["Year"], month=d["Month"], day=1), errors="raise")
    d["month"] = dates.dt.to_period("M")
    if d[["iso3", "source_country"]].isna().any().any():
        raise ValueError("iMaPP country identifiers are missing")
    if d.duplicated(["iso3", "month"]).any():
        raise ValueError("Duplicate iMaPP country-month rows")
    for name in INSTRUMENTS:
        d[name] = pd.to_numeric(d[name].replace({"NA": None, "": None}), errors="raise")
    d[FEATURE] = d[INSTRUMENTS].sum(axis=1, min_count=len(INSTRUMENTS))
    d["quarter"] = d["month"].dt.asfreq("Q")
    grouped = d.groupby(["iso3", "source_country", "quarter"], sort=True)
    values = grouped[[*INSTRUMENTS, FEATURE]].sum(min_count=3)
    out = values.join(grouped[FEATURE].count().rename("n_complete_months")).reset_index()
    out.loc[out.n_complete_months.ne(3), FEATURE] = float("nan")
    out["quarter"] = out["quarter"].astype(str)
    return out


def country_key(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z]", "", name)


def attach_policy(panel: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    names = {country_key(name): name for name in panel.country.unique()}
    aliases = {"KOR": "South Korea", "TWN": "Taiwan", "TUR": "Türkiye"}
    q = quarterly.copy()
    q["country"] = [names.get(country_key(aliases.get(iso, name)))
                    for iso, name in zip(q.iso3, q.source_country)]
    q = q.dropna(subset=["country"])
    if q.duplicated(["country", "quarter"]).any():
        raise ValueError("Multiple iMaPP series map to one country-quarter")
    return panel.merge(q[["country", "quarter", FEATURE]], on=["country", "quarter"],
                       how="left", validate="many_to_one")
