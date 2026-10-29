"""Score prediction CSVs: metrics table + paired block-bootstrap CIs against a reference model.

Example:
    python scripts/evaluate.py results/validation/preds_*.csv --reference no_change

Writes metrics.csv and bootstrap_vs_<reference>.csv next to the first input file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment, validate_predictions
from housing_forecast.evaluation.bootstrap import paired_bootstrap
from housing_forecast.evaluation.metrics import metrics_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--config", default="configs/experiment.toml")
    ap.add_argument("--reference", default="no_change")
    args = ap.parse_args()

    cfg = load_config(args.config)
    preds = pd.concat([pd.read_csv(p) for p in args.preds], ignore_index=True)
    validate_predictions(preds)
    deterministic = set(cfg["run"]["deterministic_models"])
    expected = {m: cfg["run"]["seeds"][:1] if m in deterministic else cfg["run"]["seeds"]
                for m in preds["model"].unique()}
    check_alignment(preds, expected_seeds=expected)
    out = Path(args.preds[0]).parent

    table = metrics_table(preds)
    table.to_csv(out / "metrics.csv", index=False)
    print(table.round(4).to_string(index=False))

    others = [m for m in preds["model"].unique() if m != args.reference]
    boots = [paired_bootstrap(preds, m, args.reference, cfg, metric=k)
             for m in others for k in ("rmse", "mae")]
    if boots:
        boot = pd.concat(boots, ignore_index=True)
        boot.to_csv(out / f"bootstrap_vs_{args.reference}.csv", index=False)
        print(f"\nPaired block bootstrap ({cfg['bootstrap']['ci']:.0%} CI), negative diff = better than {args.reference}:")
        print(boot.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
