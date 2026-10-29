"""Build the backtest panel from B's processed files + Spring 2026 Raw (Taiwan, extra AnaHPI countries).

Example (run from the repo root):
    python scripts/build_panel.py --b-dir data/processed --spring-raw "../26spring data"

Writes data/processed/panel_quarterly.csv and data/processed/taiwan_cpi_monthly.csv
(both gitignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from housing_forecast.data.b_adapter import build_panel
from housing_forecast.data.contract import validate_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dir", default="data/processed", help="folder with B's *_processed.csv")
    ap.add_argument("--spring-raw", default="../26spring data", help="Spring 2026 data folder (has Global/ and Taiwan/)")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--no-anahpi", action="store_true", help="only B's countries + Taiwan")
    args = ap.parse_args()

    panel, cpi = build_panel(args.b_dir, args.spring_raw, add_anahpi_countries=not args.no_anahpi)
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
