"""Run forecasters on common folds and save predictions, diagnostics and run settings.

Results are stored under results/<experiment>/<phase>. Test runs require --unlock-test."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.backtesting.infoset import build_samples, features_from_config
from housing_forecast.backtesting.manifest import make_manifest, write_manifest
from housing_forecast.backtesting.runner import (
    pool_all, pool_national, pool_overseas, pool_overseas_plus_target, pool_target_only, run_backtest,
)
from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment, validate_panel
from housing_forecast.data.transforms import prepare_panel
from housing_forecast.models.registry import REGISTRY


def load_samples(cfg: dict, panel_path: str, cpi_path: str):
    panel = pd.read_csv(panel_path)
    validate_panel(panel, value_cols=["hpi"])
    cpi = pd.read_csv(cpi_path)
    prepared = prepare_panel(panel, cpi, cfg)
    return build_samples(prepared, cfg, features_from_config(cfg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.toml")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    ap.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    ap.add_argument("--features", nargs="+", default=None,
                    help="override [features].use, e.g. --features hpi_qoq interest_rate")
    ap.add_argument("--models", nargs="+", default=["no_change", "ar"], choices=sorted(REGISTRY))
    ap.add_argument("--pool", default="target_only", choices=["target_only", "overseas", "overseas_plus_target", "national", "all"])
    ap.add_argument("--phase", default="validation", choices=["validation", "test"])
    ap.add_argument("--unlock-test", action="store_true", help="only for the final frozen run")
    ap.add_argument("--experiment", default=None,
                    help="results sub-folder; default: the feature list joined by '+'")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--countries", nargs="+", default=None,
                    help="keep only these overseas countries (the target country is always kept), "
                         'e.g. --countries Australia Japan "South Korea" "United States"')
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.features:
        cfg["features"]["use"] = args.features
    target = cfg["target"]["series_id"]
    samples = load_samples(cfg, args.panel, args.cpi)
    countries = None
    if args.countries:
        home = cfg["target"]["country"]
        available = set(samples.meta["country"])
        unknown = sorted(set(args.countries) - available)
        if unknown:
            raise SystemExit(f"unknown countries {unknown}; available with these features: "
                             f"{sorted(available - {home})}")
        countries = sorted(set(args.countries) - {home})
        samples = samples.subset(samples.meta["country"].isin(countries + [home]).to_numpy())
        print(f"countries limited to: {countries} (+ {home})")
    splits = make_splits(cfg, args.phase, last_label_origin(samples, end_series(cfg)), allow_test=args.unlock_test)
    pool = {
        "target_only": pool_target_only(target),
        "overseas": pool_overseas(cfg["target"]["country"]),
        "overseas_plus_target": pool_overseas_plus_target(target, cfg["target"]["country"]),
        "national": pool_national(),
        "all": pool_all(),
    }[args.pool]

    experiment = args.experiment or "+".join(cfg["features"]["use"])
    out = Path(args.out_dir) / experiment / args.phase
    out.mkdir(parents=True, exist_ok=True)
    print(f"features: {samples.feature_names}")
    print(f"{len(samples)} samples, {samples.meta['series_id'].nunique()} series, "
          f"{len(splits)} folds ({splits[0].split_id} .. {splits[-1].split_id})")
    all_preds, expected = [], {}
    for name in args.models:
        label = name if args.pool == "target_only" else f"{name}__{args.pool}"
        if countries is not None:
            label += f"__{len(countries)}countries"
        stochastic = REGISTRY[name].stochastic
        seeds = cfg["run"]["seeds"] if stochastic else cfg["run"]["seeds"][:1]
        preds = run_backtest(samples, splits, REGISTRY[name], cfg, pool, model_name=label)
        path = out / f"preds_{label}.csv"
        preds.to_csv(path, index=False)
        diag = preds.attrs.pop("diagnostics", None)
        if diag is not None:   # Exclude diagnostics from the preds_* evaluation glob.
            diag.to_csv(out / f"diag_{label}.csv", index=False)
        write_manifest(path, make_manifest(
            cfg, experiment=experiment, phase=args.phase, model=label, base_model=name,
            pool=args.pool, stochastic=stochastic, seeds=seeds,
            panel_path=args.panel, cpi_path=args.cpi, countries=countries))
        all_preds.append(preds)
        expected[label] = seeds
        print(f"  {label}: {len(preds)} rows -> {path}")
    check_alignment(pd.concat(all_preds), expected_seeds=expected)
    print("all models row-aligned")


if __name__ == "__main__":
    main()
