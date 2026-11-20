#!/usr/bin/env python3
"""Download missing BIS CPI series and write CPI_48countries.csv.

Keep New Zealand’s interpolated monthly series separate unless explicitly enabled."""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_LONG_CPI/1.0"
COUNTRIES = {
    "NZL": ("NZ", "New Zealand"),
    "RUS": ("RU", "Russia"),
    "CHL": ("CL", "Chile"),
    "MEX": ("MX", "Mexico"),
    "ROU": ("RO", "Romania"),
}
MEASURES = {"628": ("_Z", "IX", "Index, 2010 = 100"),
            "771": ("GY", "PA", "Year-on-year change, percent")}
PROJECT_36 = [
    "STRUCTURE", "STRUCTURE_ID", "STRUCTURE_NAME", "ACTION", "REF_AREA",
    "Country", "FREQ", "Frequency of observation", "METHODOLOGY",
    "Methodology", "MEASURE", "Measure", "UNIT_MEASURE", "Unit of measure",
    "EXPENDITURE", "Expenditure", "ADJUSTMENT", "Adjustment",
    "TRANSFORMATION", "Transformation", "Period", "Time period", "Value",
    "Observation value", "OBS_STATUS", "Observation status", "UNIT_MULT",
    "Unit multiplier", "BASE_PER", "Base period", "DURABILITY", "Durability",
    "DECIMALS", "Decimals", "Year", "Month",
]
CAVEATS = {
    "NZL": "BIS monthly historical CPI constructed/interpolated from official quarterly CPI; use native Q for frequency-sensitive quality tests.",
    "RUS": "BIS monthly CPI begins about 2001; cannot cover full calendar 2000 from this dataset alone.",
    "CHL": "Historical BIS series before Dec 2009 measures Santiago rather than nationwide CPI; record coverage break.",
    "MEX": "Compare with INEGI national CPI and confirm base changes before mixing sources.",
    "ROU": "BIS national CPI differs from Eurostat HICP; do not mislabel it as HICP.",
}


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4, connect=4, read=3, status=4,
        backoff_factor=1.2, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "academic-cpi-research/1.0 (BIS SDMX public statistics)",
        "Accept": "text/csv, application/vnd.sdmx.data+csv;q=0.9, */*;q=0.5",
    })
    return s


def extract_rows(text: str, country: str, measure: str, requested_key: str) -> list[dict]:
    """Normalize actual BIS response column variants and ensure no wrong series leaks."""
    if text.lstrip().startswith(("<", "{", "[")):
        raise ValueError("Unexpected HTML/XML/JSON response; not an SDMX-CSV dataset")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    head = reader.fieldnames or []
    normalized = {h.strip().upper(): h for h in head if h}
    for req in ("REF_AREA", "TIME_PERIOD", "OBS_VALUE"):
        if req not in normalized:
            raise ValueError(f"BIS CSV missing required {req}; got columns: {head}")

    accepted = []
    for row in reader:
        if not row or all(not (v or "").strip() for v in row.values() if isinstance(v, str)):
            continue
        bis_area = (row.get(normalized["REF_AREA"]) or "").strip()
        if bis_area != COUNTRIES[country][0]:
            raise ValueError(f"Wrong BIS area {bis_area!r}: expected {COUNTRIES[country][0]!r}")
        freq = (row.get(normalized.get("FREQ", "")) or "M").strip()
        unit = (row.get(normalized.get("UNIT_MEASURE", "")) or measure).strip()
        if freq != "M" or unit != measure:
            raise ValueError(f"Unexpected series: freq={freq!r}, unit={unit!r} for {requested_key}")
        period = (row.get(normalized["TIME_PERIOD"]) or "").strip()
        if not re.fullmatch(r"\d{4}-\d{2}", period):
            raise ValueError(f"Unexpected period {period!r} for {requested_key}")
        raw_value = (row.get(normalized["OBS_VALUE"]) or "").strip()
        if not raw_value:
            continue  # No published value, don't make one up.
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Non-numeric CPI value {raw_value!r}: {requested_key}") from exc
        if not -1e100 < value < 1e100 or (measure == "628" and value <= 0):
            raise ValueError(f"Unexpected CPI value {value!r}: {requested_key}")
        row = {k: v for k, v in row.items() if k is not None}
        row.update({
            "REQUESTED_ISO3": country,
            "REQUESTED_BIS_SERIES": requested_key,
            "REQUESTED_MEASURE": measure,
            "REQUESTED_PERIOD": period,
        })
        accepted.append(row)
    if len({r["REQUESTED_PERIOD"] for r in accepted}) != len(accepted):
        raise ValueError(f"Duplicate months in BIS response for {requested_key}")
    return accepted


def fetch_one(session, iso3, measure, start, end, pause):
    bis_area = COUNTRIES[iso3][0]
    key = f"M.{bis_area}.{measure}"
    url = f"{API}/{key}"
    params = {"startPeriod": start, "endPeriod": end, "format": "csv"}
    # Prefer v2 API. Some installations accept csvfile instead of csv.
    failures = []
    for fmt in ("csv", "csvfile"):
        params["format"] = fmt
        full_url = f"{url}?{urlencode(params)}"
        try:
            resp = session.get(url, params=params, timeout=(15, 120))
            if resp.status_code in (204, 404):
                failures.append(f"{fmt}: HTTP {resp.status_code}")
                continue
            resp.raise_for_status()
            rows = extract_rows(resp.text, iso3, measure, key)
            time.sleep(pause)
            if not rows:
                failures.append(f"{fmt}: no nonmissing observations")
                continue
            return rows, full_url
        except (requests.RequestException, ValueError) as exc:
            failures.append(f"{fmt}: {exc}")
            continue
    raise RuntimeError(f"Could not download BIS {key}: " + " | ".join(failures))


def as_project_row(bis_row: dict) -> dict:
    iso3 = bis_row["REQUESTED_ISO3"]
    measure = bis_row["REQUESTED_MEASURE"]
    transformation, unit_measure, unit_label = MEASURES[measure]
    period = bis_row["REQUESTED_PERIOD"]
    # Source observation status may be missing; do not claim it's the same as OECD's.
    status = bis_row.get("OBS_STATUS", "") or ""
    row = {field: "" for field in PROJECT_36}
    row.update({
        "STRUCTURE": "DATAFLOW",
        "STRUCTURE_ID": "BIS:WS_LONG_CPI(1.0)",
        "STRUCTURE_NAME": "BIS long consumer prices; monthly national series",
        "ACTION": "I",
        "REF_AREA": iso3,
        "Country": COUNTRIES[iso3][1],
        "FREQ": "M",
        "Frequency of observation": "Monthly",
        "METHODOLOGY": "N",
        "Methodology": "BIS compiled national historical series",
        "MEASURE": "CPI",
        "Measure": "Consumer price index",
        "UNIT_MEASURE": unit_measure,
        "Unit of measure": unit_label,
        "EXPENDITURE": "_T",
        "Expenditure": "Total",
        "ADJUSTMENT": "N",
        "Adjustment": "Not seasonally adjusted",
        "TRANSFORMATION": transformation,
        "Transformation": "Growth rate, over 1 year" if measure == "771" else "Not applicable",
        "Period": period,
        "Value": bis_row.get("OBS_VALUE", ""),
        "OBS_STATUS": status,
        "Observation status": ("BIS monthly interpolated from quarterly official CPI" if iso3 == "NZL"
                               else "BIS published long-series observation"),
        "BASE_PER": "2010" if measure == "628" else "",
        "Year": period[:4],
        "Month": period[5:7],
    })
    return row


def perform_merge(base_file: Path, bis_rows: list[dict], output_file: Path,
                  allow_nz_interpolated: bool):
    if not base_file.is_file():
        raise FileNotFoundError(f"Original CPI CSV not found: {base_file}")
    if base_file.resolve() == output_file.resolve():
        raise ValueError("Output path must not equal input path; original data stays unchanged")
    if not output_file.parent.is_dir():
        raise FileNotFoundError(f"Output folder does not exist: {output_file.parent}")
    with base_file.open(encoding="utf-8-sig", newline="") as f:
        source = csv.DictReader(f)
        if source.fieldnames != PROJECT_36:
            raise ValueError("Input CSV does not match the expected 36-column project schema")
        existing = list(source)
    def key(row):
        return row["REF_AREA"], row["FREQ"], row["TRANSFORMATION"], row["Period"]
    seen = {key(row) for row in existing}
    added, skipped = [], Counter()
    for row in bis_rows:
        k = key(row)
        if row["REF_AREA"] == "NZL" and not allow_nz_interpolated:
            skipped["NZL: BIS monthly is interpolated from native quarterly"] += 1
        elif k in seen:
            skipped["already exists; kept original row"] += 1
        else:
            added.append(row)
            seen.add(k)
    # Preserve existing data exactly and append only genuinely new keys.
    merged = existing + sorted(added, key=lambda r: (
        r["REF_AREA"], r["Period"], r["TRANSFORMATION"]))
    with output_file.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=PROJECT_36, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print(f"Saved ONLY: {output_file.resolve()} ({len(merged):,} rows)")
    print(f"Existing rows: {len(existing):,}; added: {len(added):,}; skipped: {dict(skipped)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Download BIS CPI for 5 missing housing-project markets")
    p.add_argument("--countries", nargs="+", choices=tuple(COUNTRIES),
                   default=list(COUNTRIES), metavar="ISO3")
    p.add_argument("--start", default="1999-01", help="YYYY-MM (include 1999 for 2000 YoY)")
    p.add_argument("--end", default="2026-10", help="YYYY-MM")
    p.add_argument("--out", type=Path, default=Path("CPI_48countries.csv"),
                   help="Output filename (default: CPI_48countries.csv; only output file)")
    p.add_argument("--merge-with", type=Path, default=Path("CPI_month_data.csv"),
                   help="Input 36-col CPI CSV (default: CPI_month_data.csv; read-only)")
    p.add_argument("--include-nz-interpolated", action="store_true",
                   help="Allow merging BIS synthetic monthly NZL. Not recommended for MAD.")
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls")

    args = p.parse_args(argv)
    if not (re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", args.start) and
            re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", args.end)):
        p.error("--start and --end must be YYYY-MM")
    if args.start > args.end or args.sleep < 0:
        p.error("Invalid date range or negative --sleep")
    session = make_session()
    converted, errors = [], []
    for iso3 in args.countries:
        for measure in MEASURES:
            try:
                rows, url = fetch_one(session, iso3, measure, args.start, args.end, args.sleep)
            except RuntimeError as e:
                print(f"WARNING: {e}", file=sys.stderr)
                errors.append([iso3, measure, str(e)])
                continue
            converted.extend(map(as_project_row, rows))
            dates = sorted(r["REQUESTED_PERIOD"] for r in rows)
            print(f"OK {iso3} {measure}: {len(rows):,} obs ({dates[0]} to {dates[-1]})")
    if errors:
        print("ERROR: Some BIS downloads failed; no CSV was generated:", file=sys.stderr)
        for country, measure, reason in errors:
            print(f"  {country} / {measure}: {reason}", file=sys.stderr)
        raise SystemExit(1)
    if not converted:
        raise SystemExit("No BIS observations downloaded. No output generated.")
    converted.sort(key=lambda r: (r["REF_AREA"], r["Period"], r["TRANSFORMATION"]))
    keys = [(r["REF_AREA"], r["FREQ"], r["TRANSFORMATION"], r["Period"])
            for r in converted]
    if len(keys) != len(set(keys)):
        raise SystemExit("ERROR: repeated BIS observations; refused to write duplicates")
    perform_merge(args.merge_with, converted, args.out, args.include_nz_interpolated)
    print("Source: BIS long consumer prices database, www.bis.org/statistics/cp.htm")


if __name__ == "__main__":
    main()
