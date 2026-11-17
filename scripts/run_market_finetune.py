"""Compare local fine-tuning with the exact shared model it started from.

Default: every market is fine-tuned for a fixed number of epochs (--ft-epochs, first run).
With --select-epochs: each market's epoch count, including zero, is chosen on a time split
inside the fold's training data (SharedTransformerFineTuneSelect). This second recipe was
added after the fixed-epoch results had been seen.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform

import pandas as pd
import torch

from run_backtest import load_samples
from housing_forecast.backtesting.manifest import make_manifest, write_manifest
from housing_forecast.backtesting.runner import pool_national, run_backtest
from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment
from housing_forecast.models.transformer import SharedTransformerFineTune, SharedTransformerFineTuneSelect

SHARED = "transformer_shared__national"
FIXED = "transformer_shared_ft__national"
SELECTED = "transformer_shared_ftsel__national"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_long.toml")
    parser.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    parser.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: results/market_finetune_122/validation, or .../inner_select/validation with --select-epochs")
    parser.add_argument("--markets", nargs="+", help="markets to adapt and score; pretraining still uses all national markets")
    parser.add_argument("--folds", type=int, help="limit validation to its first N folds")
    parser.add_argument("--seeds", type=int, nargs="+", help="override the configured seed list")
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--ft-lr-scale", type=float, default=0.1)
    parser.add_argument("--min-market-rows", type=int, default=24)
    parser.add_argument("--select-epochs", action="store_true",
                        help="choose each market's epochs (0 = keep shared) on an inner time split")
    parser.add_argument("--ft-max-epochs", type=int, default=30, help="largest candidate with --select-epochs")
    parser.add_argument("--min-val-windows", type=int, default=4,
                        help="inner validation windows a market needs before its epochs are chosen")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = Path("results/market_finetune_122/inner_select/validation" if args.select_epochs
                            else "results/market_finetune_122/validation")

    cfg = load_config(args.config)
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("seeds must be unique")
        cfg["run"]["seeds"] = args.seeds
    samples = load_samples(cfg, args.panel, args.cpi)
    if args.markets:
        available = set(samples.meta.loc[samples.meta["geography_level"].eq("national"), "series_id"])
        missing = set(args.markets) - available
        if missing:
            parser.error(f"unknown national markets: {sorted(missing)}")
        cfg["target"]["eval_series"] = sorted(set(args.markets))
    splits = make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))
    if not splits:
        parser.error("no validation folds")
    if args.folds is not None:
        if args.folds < 1:
            parser.error("folds must be positive")
        splits = splits[:args.folds]
        cfg["periods"]["validation_end"] = str(splits[-1].test_origins[-1])
    if any(args.out_dir.glob("preds_*.csv")):
        parser.error("the output directory already contains predictions; use a new directory")
    if args.select_epochs:
        model_cls, ADAPTED = SharedTransformerFineTuneSelect, SELECTED
        recipe = {"ft_max_epochs": args.ft_max_epochs, "ft_lr_scale": args.ft_lr_scale,
                  "min_market_rows": args.min_market_rows, "min_val_windows": args.min_val_windows,
                  "adapt_markets": args.markets}
    else:
        model_cls, ADAPTED = SharedTransformerFineTune, FIXED
        recipe = {"ft_epochs": args.ft_epochs, "ft_lr_scale": args.ft_lr_scale,
                  "min_market_rows": args.min_market_rows, "adapt_markets": args.markets}
    model_cls(**recipe)
    print(f"Shared pretraining and local adaptation ({len(splits)} folds, {len(cfg['run']['seeds'])} seeds)",
          flush=True)
    preds = run_backtest(samples, splits, lambda: model_cls(**recipe), cfg, pool_national(),
                         model_name=ADAPTED, reference_model_name=SHARED)
    shared = preds.attrs.pop("reference_predictions")
    diagnostics = preds.attrs.pop("diagnostics")
    if preds.empty:
        parser.error("the chosen markets have no scored forecasts in these folds")
    check_alignment(pd.concat([preds, shared], ignore_index=True),
                    expected_seeds={name: cfg["run"]["seeds"] for name in [ADAPTED, SHARED]})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for frame, name in [(preds, ADAPTED), (shared, SHARED)]:
        path = args.out_dir / f"preds_{name}.csv"
        frame.to_csv(path, index=False)
        manifest = make_manifest(
            cfg, experiment="market_finetune", phase="validation", model=name,
            base_model=model_cls.name if name == ADAPTED else "transformer",
            pool="national", stochastic=True, seeds=cfg["run"]["seeds"],
            panel_path=args.panel, cpi_path=args.cpi,
        )
        manifest["shared_pretraining"] = "all national markets, including target-market histories"
        manifest["execution_environment"] = {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "python": platform.python_version(), "torch": torch.__version__,
        }
        if name == ADAPTED:
            manifest["adaptation"] = recipe
            if args.select_epochs:
                manifest["adaptation_rule"] = ("epochs 0..ft_max_epochs chosen per market, fold and seed on an inner "
                                               "time split; added after the fixed-epoch run had been seen")
        write_manifest(path, manifest)
    diagnostics.to_csv(args.out_dir / "fit_diagnostics.csv", index=False)
    print(f"Saved {len(preds)} forecasts per model to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
