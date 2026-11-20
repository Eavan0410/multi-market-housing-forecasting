"""Compare HPI-only forecasts with an added iMaPP policy feature on matched samples."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd

from run_backtest import load_samples
from run_cpi_experiment import coverage, hpi_only, same_settings
from housing_forecast.backtesting.infoset import build_samples, features_from_config
from housing_forecast.backtesting.manifest import make_manifest, read_manifest, write_manifest
from housing_forecast.backtesting.runner import pool_national, run_backtest
from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment, validate_panel
from housing_forecast.data.imapp import FEATURE, QUARTERLY, attach_policy
from housing_forecast.data.transforms import prepare_panel
from housing_forecast.evaluation.markets import metrics_by_market, paired_ratio_bootstrap
from housing_forecast.evaluation.markets import relative_to_reference, summary_by_horizon
from housing_forecast.models.registry import REGISTRY

PLAN = {
    "no_change": [("no_change__hpi", "hpi", "no_change")],
    "ar_pooled": [("ar_pooled__hpi", "hpi", "ar_pooled"),
                  ("arx_pooled__imapp", "imapp", "arx_pooled")],
    "transformer": [("transformer__hpi", "hpi", "transformer"),
                    ("transformer__imapp", "imapp", "transformer")],
}
COMPARISONS = [
    ("transformer__imapp", "transformer__hpi"),
    ("arx_pooled__imapp", "ar_pooled__hpi"),
    ("transformer__imapp", "arx_pooled__imapp"),
]


def policy_samples(cfg, panel_path, cpi_path, policy_path):
    panel = pd.read_csv(panel_path)
    validate_panel(panel, value_cols=["hpi"])
    quarterly = pd.read_csv(policy_path)
    merged = attach_policy(panel, quarterly)
    prepared = prepare_panel(merged, pd.read_csv(cpi_path), cfg)
    return build_samples(prepared, cfg, features_from_config(cfg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_long.toml")
    parser.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    parser.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    parser.add_argument("--policy", type=Path, default=QUARTERLY)
    parser.add_argument("--policy-lag-days", type=int, default=0,
                        help="assumed availability delay after implementation quarter")
    parser.add_argument("--models", nargs="+", choices=sorted(PLAN), default=list(PLAN))
    parser.add_argument("--out-dir", type=Path, default=Path("results/imapp/validation"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.policy_lag_days < 0:
        parser.error("policy-lag-days must be nonnegative")
    cfg = deepcopy(load_config(args.config))
    cfg["features"]["use"] = ["hpi_qoq", FEATURE]
    cfg["features"]["ffill_limit"] = 0
    cfg["release_lag_days"][FEATURE] = {"default": args.policy_lag_days}
    both = policy_samples(cfg, args.panel, args.cpi, args.policy)
    if not len(both):
        parser.error("no complete HPI and policy input windows")
    arms = {"imapp": both, "hpi": hpi_only(both)}
    main_samples = load_samples(load_config(args.config), args.panel, args.cpi)
    splits = make_splits(cfg, "validation", last_label_origin(both, end_series(cfg)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cov = coverage(both, main_samples, splits, cfg).rename(
        columns=lambda name: name.replace("_cpi", "_policy"))
    cov.to_csv(args.out_dir / "coverage.csv", index=False)
    metadata_path = args.policy.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    settings = {
        "source": {key: value for key, value in metadata.items() if key != "built_utc"},
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "policy_lag_days": args.policy_lag_days,
        "sample_rule": "complete HPI and iMaPP windows; identical samples in both arms; no forward fill",
        "status": "exploratory validation using retrospective implementation-date policy data",
    }
    (args.out_dir / "experiment.json").write_text(json.dumps(settings, indent=2) + "\n")
    eligible = cov[cov.forecast_origins_policy.ge(8)]
    print(f"{len(eligible)} markets with at least eight validation origins; {len(both)} complete windows", flush=True)
    if args.prepare_only:
        return
    frames, expected = [], {}
    for model in args.models:
        for label, arm, base in PLAN[model]:
            arm_cfg = deepcopy(cfg)
            arm_cfg["features"]["use"] = arms[arm].feature_names
            stochastic = REGISTRY[base].stochastic
            seeds = cfg["run"]["seeds"] if stochastic else cfg["run"]["seeds"][:1]
            manifest = make_manifest(arm_cfg, experiment="imapp", phase="validation", model=label,
                                     base_model=base, pool="national", stochastic=stochastic,
                                     seeds=seeds, panel_path=args.panel, cpi_path=args.cpi)
            manifest["imapp"] = settings
            path = args.out_dir / f"preds_{label}.csv"
            previous = read_manifest(path)
            if previous and "imapp" in previous:
                previous["imapp"].get("source", {}).pop("built_utc", None)
            if path.exists() and same_settings(previous, manifest):
                preds = pd.read_csv(path)
                check_alignment(preds, expected_seeds={label: seeds})
                print(f"Reusing {label}", flush=True)
            else:
                print(f"Running {label}", flush=True)
                preds = run_backtest(arms[arm], splits, REGISTRY[base], cfg, pool_national(), model_name=label)
                check_alignment(preds, expected_seeds={label: seeds})
                temporary = path.with_suffix(".tmp")
                preds.to_csv(temporary, index=False)
                write_manifest(path, manifest)
                temporary.replace(path)
            frames.append(preds)
            expected[label] = seeds
    predictions = pd.concat(frames, ignore_index=True)
    check_alignment(predictions, expected_seeds=expected)
    metrics = metrics_by_market(predictions)
    metrics.to_csv(args.out_dir / "metrics_by_market.csv", index=False)
    if "no_change__hpi" in expected:
        relative = relative_to_reference(metrics, "no_change__hpi")
        summary_by_horizon(relative).to_csv(args.out_dir / "summary_by_horizon.csv", index=False)
    for a, b in COMPARISONS:
        if {a, b} <= expected.keys():
            ci = paired_ratio_bootstrap(predictions, a, b, cfg)
            ci.to_csv(args.out_dir / f"ratio_ci_{a}_vs_{b}.csv", index=False)
            print(ci[ci.horizon.isin(["h3-6", "h16-20"])][["horizon", "median_ratio", "ci_low", "ci_high"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
