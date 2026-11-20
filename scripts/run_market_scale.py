"""Compare 5, 10, 20 and 41 training markets on five fixed evaluation markets.

Region, history and random orders share their 5- and 41-market runs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from generate_market_orders import EVALUATION_MARKETS, MARKET_LIST_PATH, market_orders
from run_backtest import load_samples
from housing_forecast.backtesting.manifest import make_manifest, manifest_path, read_manifest, write_manifest
from housing_forecast.backtesting.runner import run_backtest, training_samples
from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment
from housing_forecast.models.transformer import TransformerForecaster

ORDERS = {
    "region": "1 - Region rotation",
    "history": "2 - Long-history first",
    "random": "3 - Seeded random",
}
BASE_DIR = Path("results/market_scale_122")       # 122-day forecast date (100-day runs: results/market_scale)
IGNORED_KEYS = {"created_utc", "market_order"}


def default_out_dir(order: str, phase: str = "validation") -> Path:
    return BASE_DIR / phase if order == "region" else BASE_DIR / order / phase


def same_settings(previous: dict | None, manifest: dict) -> bool:
    """Same run settings and the same set of training markets (their order does not matter)."""
    if previous is None:
        return False
    keys = (set(previous) | set(manifest)) - IGNORED_KEYS
    for key in keys:
        old, new = previous.get(key), manifest.get(key)
        if key == "train_series":
            old, new = sorted(old or []), sorted(new or [])
        if old != new:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--order", choices=sorted(ORDERS), default="region")
    parser.add_argument("--config", default="configs/experiment_long.toml")
    parser.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    parser.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    parser.add_argument("--market-list", type=Path, default=MARKET_LIST_PATH)
    parser.add_argument("--phase", default="validation", choices=["validation", "test"])
    parser.add_argument("--unlock-test", action="store_true", help="only for the final frozen run")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: results/market_scale_122/<phase> for region, else results/market_scale_122/<order>/<phase>")
    parser.add_argument("--reuse-from", type=Path, default=None,
                        help="folder whose runs may be copied when the training markets are the same "
                             "(default: the region-rotation folder of the same phase)")
    args = parser.parse_args()
    out_dir = args.out_dir or default_out_dir(args.order, args.phase)
    if args.reuse_from is None:
        args.reuse_from = default_out_dir("region", args.phase)

    cfg = load_config(args.config)
    cfg["target"]["eval_series"] = list(EVALUATION_MARKETS)
    order = [str(m) for m in market_orders(pd.read_csv(args.market_list))[ORDERS[args.order]]]
    ranked = [*EVALUATION_MARKETS, *order]
    samples = load_samples(cfg, args.panel, args.cpi)
    available = set(samples.meta.loc[samples.meta["geography_level"].eq("national"), "series_id"])
    missing = set(ranked) - available
    if missing:
        parser.error(f"missing national markets in the prepared panel: {sorted(missing)}")
    splits = make_splits(cfg, args.phase, last_label_origin(samples, end_series(cfg)), allow_test=args.unlock_test)
    if not splits:
        parser.error(f"no {args.phase} folds")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Order: {ORDERS[args.order]} -> {out_dir}", flush=True)

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
                "market_order": ORDERS[args.order],
                "pool_size": size,
                "split_id": split.split_id,
                "n_train_markets": len(active),
                "n_train_rows": len(train),
                "train_markets": " | ".join(active),
            })

        label = f"transformer__scale_{size}"
        path = out_dir / f"preds_{label}.csv"
        manifest = make_manifest(
            cfg, experiment="market_scale", phase=args.phase, model=label, base_model="transformer",
            pool=f"scale_{size}", stochastic=True, seeds=seeds, panel_path=args.panel, cpi_path=args.cpi,
        )
        manifest["train_series"] = markets
        manifest["market_order"] = ORDERS[args.order]

        source = args.reuse_from / path.name
        if not path.exists() and source.exists() and source.resolve() != path.resolve():
            if same_settings(read_manifest(source), manifest):
                shutil.copyfile(source, path)
                shutil.copyfile(manifest_path(source), manifest_path(path))
                print(f"Copied {source} (same settings and training markets)", flush=True)

        if path.exists():
            if not same_settings(read_manifest(path), manifest):
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
    pd.DataFrame(coverage).to_csv(out_dir / "training_coverage.csv", index=False)
    print("All pools score the same forecast rows.", flush=True)


if __name__ == "__main__":
    main()
