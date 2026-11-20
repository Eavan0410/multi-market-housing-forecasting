"""Build quarterly housing policy actions from the IMF iMaPP workbook."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from housing_forecast.data.imapp import ARCHIVE, FEATURE, INSTRUMENTS, QUARTERLY, SOURCE_URL
from housing_forecast.data.imapp import attach_policy, load_monthly, quarterly_actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ARCHIVE)
    parser.add_argument("--panel", type=Path, default=Path("data/interim/panel_quarterly.csv"))
    parser.add_argument("--out", type=Path, default=QUARTERLY)
    args = parser.parse_args()
    monthly = load_monthly(args.input)
    quarterly = quarterly_actions(monthly)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    quarterly.to_csv(args.out, index=False)
    panel = pd.read_csv(args.panel)
    merged = attach_policy(panel, quarterly)
    coverage = merged[merged.geography_level.eq("national")].groupby("series_id").agg(
        hpi_quarters=("hpi", "count"), policy_quarters=(FEATURE, "count"))
    coverage.to_csv(args.out.with_name("imapp_market_coverage.csv"))
    metadata = {
        "source": SOURCE_URL, "input": str(args.input),
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "instruments": INSTRUMENTS, "feature": FEATURE,
        "aggregation": "sum of signed LTV, DSTI and LoanR_HH actions over three complete months",
        "timing": "implementation months; retrospective dataset, not historical publication vintages",
        "first_quarter": quarterly.quarter.min(), "last_quarter": quarterly.quarter.max(),
        "countries": int(quarterly.iso3.nunique()),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{metadata['countries']} countries, {metadata['first_quarter']}–{metadata['last_quarter']}")
    print(f"{int(coverage.policy_quarters.gt(0).sum())} national HPI markets overlap. Saved {args.out}")


if __name__ == "__main__":
    main()
