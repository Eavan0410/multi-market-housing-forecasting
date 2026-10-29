"""Score prediction CSVs: metrics table + paired block-bootstrap CIs against a reference model.

Example:
    python scripts/evaluate.py results/<experiment>/validation/preds_*.csv --reference no_change

Files written by run_backtest.py carry a manifest (preds_<label>.meta.json); all
manifests must agree on the information set (features, window, periods, data
files) or the evaluation stops. Files without a manifest (e.g. a CSV from
another script) are scored with a warning, and their seeds are checked against
[run].deterministic_models.

Writes next to the first input file:
    metrics.csv                  point metrics per model and horizon (mean and sd over seeds)
    metric_ci.csv                block-bootstrap CI of each model's RMSE and MAE
    bootstrap_vs_<reference>.csv paired CI of each model's difference from the reference
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.backtesting.manifest import check_comparable, read_manifest
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment, validate_predictions
from housing_forecast.evaluation.bootstrap import metric_ci, paired_bootstrap
from housing_forecast.evaluation.metrics import metrics_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--config", default="configs/experiment.toml")
    ap.add_argument("--reference", default="no_change")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frames, manifests, expected = [], {}, {}
    deterministic = set(cfg["run"]["deterministic_models"])
    for p in args.preds:
        df = pd.read_csv(p)
        frames.append(df)
        m = read_manifest(p)
        if m is None:
            print(f"warning: {p} has no run manifest; its settings cannot be checked")
            for model in df["model"].unique():
                expected[model] = cfg["run"]["seeds"][:1] if model in deterministic else cfg["run"]["seeds"]
        else:
            manifests[p] = m
            expected[m["model"]] = m["seeds"]
    check_comparable(manifests)
    preds = pd.concat(frames, ignore_index=True)
    validate_predictions(preds)
    check_alignment(preds, expected_seeds=expected)
    out = Path(args.preds[0]).parent

    table = metrics_table(preds)
    table.to_csv(out / "metrics.csv", index=False)
    print(table.round(4).to_string(index=False))

    cis = pd.concat([metric_ci(preds, m, cfg, metric=k)
                     for m in preds["model"].unique() for k in ("rmse", "mae")], ignore_index=True)
    cis.to_csv(out / "metric_ci.csv", index=False)
    print(f"\nBlock bootstrap {cfg['bootstrap']['ci']:.0%} CI per model:")
    print(cis.round(4).to_string(index=False))

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
