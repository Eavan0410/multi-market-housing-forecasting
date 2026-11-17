"""Forecast one market with a Transformer trained on the other national markets."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from run_backtest import load_samples
from housing_forecast.backtesting.manifest import check_comparable, make_manifest, read_manifest, write_manifest
from housing_forecast.backtesting.runner import pool_national, run_backtest, training_samples
from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment
from housing_forecast.models.transformer import TransformerForecaster

MARKETS = ("Australia", "Portugal", "Chile", "Taiwan_national")
REFERENCES = ("transformer__national", "ar__national", "no_change__national")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="Australia", choices=MARKETS)
    parser.add_argument("--config", default="configs/experiment_long.toml")
    parser.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    parser.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    parser.add_argument("--reference-dir", type=Path, default=Path("results/long_hpi_122/validation"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/market_holdout_122"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    market = args.market
    seeds = cfg["run"]["seeds"]
    label = "transformer__holdout"
    manifest = make_manifest(
        cfg, experiment="market_holdout", phase="validation", model=label, base_model="transformer",
        pool="national_except_market", stochastic=True, seeds=seeds, panel_path=args.panel, cpi_path=args.cpi,
    )
    frames, manifests = {}, {}
    for name in REFERENCES:
        path = args.reference_dir / f"preds_{name}.csv"
        ref = read_manifest(path)
        expected_seeds = seeds if name == REFERENCES[0] else seeds[:1]
        if (ref is None or ref.get("model") != name or ref.get("pool") != "national"
                or ref.get("countries") != "all" or ref.get("seeds") != expected_seeds):
            parser.error(f"{path} needs a matching national run manifest")
        check_comparable({"current config": manifest, str(path): ref})
        frames[name] = pd.read_csv(path).loc[lambda rows: rows["series_id"].eq(market)].copy()
        manifests[name] = dict(ref, source_file=str(path))
    check_alignment(pd.concat(frames.values(), ignore_index=True),
                    expected_seeds={name: ref["seeds"] for name, ref in manifests.items()})

    cfg = deepcopy(cfg)
    cfg["target"]["eval_series"] = [market]
    manifest["target"] = cfg["target"]
    manifest["held_out_series"] = market
    for ref in manifests.values():
        ref["target"] = cfg["target"]

    samples = load_samples(cfg, args.panel, args.cpi)
    national = pool_national()

    def training_pool(meta):
        return national(meta) & meta["series_id"].ne(market).to_numpy()

    splits = make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))
    coverage = []
    scored_splits = set(frames[REFERENCES[0]]["split_id"])
    for split in splits:
        if split.split_id not in scored_splits:
            continue
        full = training_samples(samples, split, national)
        train = training_samples(samples, split, training_pool)
        active = sorted(train.meta["series_id"].unique())
        if market in active:
            raise ValueError(f"{market} appears in training for {split.split_id}")
        coverage.append({
            "split_id": split.split_id,
            "n_train_markets": len(active),
            "n_train_rows": len(train),
            "held_out_rows_removed": int(full.meta["series_id"].eq(market).sum()),
            "train_markets": " | ".join(active),
        })

    print(f"Holding out {market} ({len(scored_splits)} scored folds, {len(seeds)} seeds)", flush=True)
    frames[label] = run_backtest(samples, splits, TransformerForecaster, cfg, training_pool, model_name=label)
    frames[label].attrs.clear()
    manifests[label] = manifest
    check_alignment(pd.concat(frames.values(), ignore_index=True),
                    expected_seeds={name: ref["seeds"] for name, ref in manifests.items()})

    out = args.out_dir / market.lower() / "validation"
    out.mkdir(parents=True, exist_ok=True)
    for name, preds in frames.items():
        path = out / f"preds_{name}.csv"
        preds.to_csv(path, index=False)
        write_manifest(path, manifests[name])
    pd.DataFrame(coverage).to_csv(out / "training_coverage.csv", index=False)
    print(f"Saved {len(frames[label])} held-out predictions aligned with all three references to {out}")


if __name__ == "__main__":
    main()
