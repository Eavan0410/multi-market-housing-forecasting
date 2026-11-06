"""Run the market-scale validation with one fixed set of evaluation markets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from generate_market_orders import EVALUATION_MARKETS, MARKET_LIST_PATH, market_orders
from run_backtest import load_samples
from housing_forecast.backtesting.manifest import make_manifest, read_manifest, write_manifest
from housing_forecast.backtesting.runner import run_backtest, training_samples
from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment
from housing_forecast.models.transformer import TransformerForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_long.toml")
    parser.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    parser.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    parser.add_argument("--market-list", type=Path, default=MARKET_LIST_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("results/market_scale/validation"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["target"]["eval_series"] = list(EVALUATION_MARKETS)
    ranked = [*EVALUATION_MARKETS, *market_orders(pd.read_csv(args.market_list))["1 - Region rotation"]]
    samples = load_samples(cfg, args.panel, args.cpi)
    available = set(samples.meta.loc[samples.meta["geography_level"].eq("national"), "series_id"])
    missing = set(ranked) - available
    if missing:
        parser.error(f"missing national markets in the prepared panel: {sorted(missing)}")
    splits = make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))
    if not splits:
        parser.error("no validation folds")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames, expected, coverage = [], {}, []
    seeds = cfg["run"]["seeds"]
    for size in (5, 10, 20, len(ranked)):
        markets = ranked[:size]

        def pool(meta):
            return meta["series_id"].isin(markets).to_numpy()

        for split in splits:
            train = training_samples(samples, split, pool)
            active = sorted(train.meta["series_id"].unique())
            coverage.append({
                "pool_size": size,
                "split_id": split.split_id,
                "n_train_markets": len(active),
                "n_train_rows": len(train),
                "train_markets": " | ".join(active),
            })

        label = f"transformer__scale_{size}"
        path = args.out_dir / f"preds_{label}.csv"
        manifest = make_manifest(
            cfg, experiment="market_scale", phase="validation", model=label, base_model="transformer",
            pool=f"scale_{size}", stochastic=True, seeds=seeds, panel_path=args.panel, cpi_path=args.cpi,
        )
        manifest["train_series"] = markets
        if path.exists():
            previous = read_manifest(path)
            if previous is None or any(previous.get(k) != v for k, v in manifest.items() if k != "created_utc"):
                parser.error(f"{path} has different run settings. Use a new output directory.")
            preds = pd.read_csv(path)
            print(f"Reusing {path}", flush=True)
        else:
            print(f"Training pool of {size} markets ({len(splits)} folds, {len(seeds)} seeds)", flush=True)
            preds = run_backtest(samples, splits, TransformerForecaster, cfg, pool, model_name=label)
            preds.attrs.clear()
            if set(preds["series_id"]) != set(EVALUATION_MARKETS):
                raise ValueError(f"pool {size} is missing predictions for an evaluation market")
            check_alignment(preds, expected_seeds={label: seeds})
            temp_path = path.with_suffix(".tmp")
            preds.to_csv(temp_path, index=False)
            write_manifest(path, manifest)
            temp_path.replace(path)
            print(f"Saved {len(preds)} predictions to {path}", flush=True)
        frames.append(preds)
        expected[label] = seeds

    check_alignment(pd.concat(frames, ignore_index=True), expected_seeds=expected)
    pd.DataFrame(coverage).to_csv(args.out_dir / "training_coverage.csv", index=False)
    print("All pools score the same forecast rows.", flush=True)


if __name__ == "__main__":
    main()
