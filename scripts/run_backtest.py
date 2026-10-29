"""Run models through the shared backtest and write prediction CSVs.

Example (validation period only; the test period is locked):
    python scripts/run_backtest.py --models no_change ar --pool target_only

Outputs results/<phase>/preds_<model>.csv in the shared prediction format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.backtesting.infoset import build_samples, features_from_config
from housing_forecast.backtesting.runner import (
    pool_all, pool_overseas, pool_overseas_plus_target, pool_target_only, run_backtest,
)
from housing_forecast.backtesting.splits import last_label_origin, make_splits
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment, validate_panel
from housing_forecast.data.transforms import prepare_hpi_panel
from housing_forecast.models.baselines import REGISTRY


def load_samples(cfg: dict, panel_path: str, cpi_path: str):
    panel = pd.read_csv(panel_path)
    validate_panel(panel, value_cols=["hpi"])
    cpi = pd.read_csv(cpi_path)
    prepared = prepare_hpi_panel(panel, cpi, cfg)
    return build_samples(prepared, cfg, features_from_config(cfg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.toml")
    ap.add_argument("--panel", default="data/processed/panel_quarterly.csv")
    ap.add_argument("--cpi", default="data/processed/taiwan_cpi_monthly.csv")
    ap.add_argument("--models", nargs="+", default=["no_change", "ar"], choices=sorted(REGISTRY))
    ap.add_argument("--pool", default="target_only", choices=["target_only", "overseas", "overseas_plus_target", "all"])
    ap.add_argument("--phase", default="validation", choices=["validation", "test"])
    ap.add_argument("--unlock-test", action="store_true", help="only for the final frozen run")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = cfg["target"]["series_id"]
    samples = load_samples(cfg, args.panel, args.cpi)
    splits = make_splits(cfg, args.phase, last_label_origin(samples, target), allow_test=args.unlock_test)
    pool = {
        "target_only": pool_target_only(target),
        "overseas": pool_overseas(cfg["target"]["country"]),
        "overseas_plus_target": pool_overseas_plus_target(target, cfg["target"]["country"]),
        "all": pool_all(),
    }[args.pool]

    out = Path(args.out_dir) / args.phase
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(samples)} samples, {samples.meta['series_id'].nunique()} series, "
          f"{len(splits)} folds ({splits[0].split_id} .. {splits[-1].split_id})")
    all_preds = []
    for name in args.models:
        preds = run_backtest(samples, splits, REGISTRY[name], cfg, pool, model_name=name)
        preds.to_csv(out / f"preds_{name}.csv", index=False)
        all_preds.append(preds)
        print(f"  {name}: {len(preds)} rows -> {out / f'preds_{name}.csv'}")
    expected = {name: cfg["run"]["seeds"] if REGISTRY[name]().stochastic
                else cfg["run"]["seeds"][:1] for name in args.models}
    check_alignment(pd.concat(all_preds), expected_seeds=expected)
    print("all models row-aligned")


if __name__ == "__main__":
    main()
