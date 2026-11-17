"""Download the CPI series still missing or short in data/raw/global/CPI_48countries.csv.

    pip install requests
    python scripts/download_cpi_gaps.py

Writes data/raw/global/cpi_gaps.csv. housing_forecast.data.cpi uses it in place of the
main file for these countries:
  Japan        BIS long CPI (OECD's own Japan series stops in 2021-06, so the main file and the
               four_countries copy both end there)
  New Zealand  FRED NZLCPIALLQINMEI, quarterly index, full history (main file starts 1990)
  Russia       BIS long CPI from 1995 (main file starts 2002; Russia's HPI starts 2001Q1)

Run it on a machine with internet access, then rebuild the CPI table:
    python scripts/build_cpi_assessment.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bis_missing_cpi  # noqa: E402
from bis_missing_cpi import as_project_row, fetch_one, make_session  # noqa: E402

bis_missing_cpi.COUNTRIES.setdefault("JPN", ("JP", "Japan"))

OUT = Path("data/raw/global/cpi_gaps.csv")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
COLUMNS = ["Country", "FREQ", "Period", "Value", "TRANSFORMATION", "STRUCTURE_ID"]


def fred_new_zealand(session: requests.Session) -> pd.DataFrame:
    series = "NZLCPIALLQINMEI"
    # Plain request without retries: FRED can stall; fail fast instead of hanging.
    r = requests.get(FRED.format(series=series), timeout=(10, 60), headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    date_col = d.columns[0]
    d = d[pd.to_numeric(d[series], errors="coerce").notna()]
    period = pd.PeriodIndex(pd.to_datetime(d[date_col]), freq="Q").astype(str).str.replace("Q", "-Q")
    return pd.DataFrame({"Country": "New Zealand", "FREQ": "Q", "Period": period, "Value": d[series].astype(str),
                         "TRANSFORMATION": "_Z", "STRUCTURE_ID": f"FRED: {series} (OECD archival)"})


def bis(session: requests.Session, iso3: str, start: str, end: str = "2026-10") -> pd.DataFrame:
    rows = []
    for measure in ("628", "771"):              # index (2010=100) and year-on-year percent
        got, _ = fetch_one(session, iso3, measure, start, end, 0.5)
        rows += [as_project_row(r) for r in got]
    return pd.DataFrame(rows)[COLUMNS]


def main() -> None:
    session = make_session()
    jobs = [("Japan (BIS)", lambda: bis(session, "JPN", "1955-01")),
            ("New Zealand (FRED)", lambda: fred_new_zealand(session)),
            ("Russia (BIS)", lambda: bis(session, "RUS", "1995-01"))]
    frames, failed = [], []
    for name, fetch in jobs:
        print(f"{name:20s} downloading ...", flush=True)
        try:
            d = fetch()
        except Exception as e:  # noqa: BLE001  keep the other countries
            print(f"{name:20s} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(name)
            continue
        print(f"{name:20s} {len(d):5d} rows, {d['Period'].min()} to {d['Period'].max()}", flush=True)
        frames.append(d[COLUMNS])
    if not frames:
        raise SystemExit("nothing downloaded")
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
    print(f"-> {OUT}" + (f"  (not included: {', '.join(failed)}; those keep the main file's data)" if failed else ""))


if __name__ == "__main__":
    main()
