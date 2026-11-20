"""Build the quarterly panel from country inputs, Taiwan raw files and OECD AnaHPI."""

from __future__ import annotations

import argparse
from pathlib import Path

from housing_forecast.data.b_adapter import build_panel
from housing_forecast.data.contract import validate_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dir", default="data/processed", help="folder containing *_processed.csv")
    ap.add_argument("--anahpi", default="data/raw/global/anahpi_quarterly_raw.csv",
                    help="OECD AnaHPI file for the extra countries (HPI only)")
    ap.add_argument("--taiwan-dir", default="data/raw/taiwan", help="Taiwan HPI, CPI and macro raw files")
    ap.add_argument("--out-dir", default="data/interim")
    ap.add_argument("--no-anahpi", action="store_true", help="only the four-country inputs and Taiwan")
    args = ap.parse_args()

    panel, cpi = build_panel(args.b_dir, args.anahpi, add_anahpi_countries=not args.no_anahpi,
                             taiwan_dir=args.taiwan_dir)
    validate_panel(panel, value_cols=["hpi"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "panel_quarterly.csv", index=False)
    cpi.to_csv(out / "taiwan_cpi_monthly.csv", index=False)

    cov = panel.dropna(subset=["hpi"]).groupby("series_id")["quarter"].agg(["min", "max", "count"])
    print(f"{panel['series_id'].nunique()} series, {len(panel)} rows -> {out / 'panel_quarterly.csv'}")
    print(cov.sort_values("min").to_string())


if __name__ == "__main__":
    main()
