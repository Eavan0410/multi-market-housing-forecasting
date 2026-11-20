"""Compare HPI-only forecasts with an added CPI inflation input on validation folds.

Both versions use identical complete input windows. CPI availability is assumed
to be 30 days after quarter-end. Output is saved in results/cpi/validation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from housing_forecast.backtesting.infoset import build_samples, features_from_config  # noqa: E402
from housing_forecast.backtesting.manifest import make_manifest, read_manifest, write_manifest  # noqa: E402
from housing_forecast.backtesting.runner import pool_national, run_backtest, training_samples  # noqa: E402
from housing_forecast.backtesting.splits import end_series, eval_mask, last_label_origin, make_splits  # noqa: E402
from housing_forecast.config import load_config  # noqa: E402
from housing_forecast.data.contract import check_alignment, validate_panel  # noqa: E402
from housing_forecast.data.cpi import FALLBACK, GAPS, RAW, attach_cpi_yoy, load_cpi_raw, native_yoy, quarterly_yoy  # noqa: E402
from housing_forecast.data.transforms import prepare_panel  # noqa: E402
from housing_forecast.evaluation.markets import (  # noqa: E402
    metrics_by_market, paired_ratio_bootstrap, relative_to_reference, summary_by_horizon,
)
from housing_forecast.models.registry import REGISTRY  # noqa: E402

PLAN = {
    "no_change": [("no_change__hpi", "hpi")],
    "ar_pooled": [("ar_pooled__hpi", "hpi"), ("arx_pooled__hpi_cpi", "hpi_cpi")],
    "ar": [("ar__hpi", "hpi"), ("arx__hpi_cpi", "hpi_cpi")],
    "transformer": [("transformer__hpi", "hpi"), ("transformer__hpi_cpi", "hpi_cpi")],
    "transformer_per_market": [("transformer_per_market__hpi", "hpi"), ("transformer_per_market__hpi_cpi", "hpi_cpi")],
}
LINEAR_WITH_CPI = {"ar_pooled": "arx_pooled", "ar": "arx"}
COMPARISONS = [
    ("transformer__hpi_cpi", "transformer__hpi"),
    ("arx_pooled__hpi_cpi", "ar_pooled__hpi"),
    ("arx__hpi_cpi", "ar__hpi"),
    ("transformer_per_market__hpi_cpi", "transformer_per_market__hpi"),
    ("transformer__hpi_cpi", "arx_pooled__hpi_cpi"),
    ("transformer__hpi", "ar_pooled__hpi"),
]
IGNORED_KEYS = {"created_utc"}
DEFAULT_MODELS = ["no_change", "ar_pooled", "transformer"]


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def cpi_samples(cfg: dict, panel_path: str, taiwan_cpi_path: str):
    """Samples whose window has HPI growth and CPI; plus the main HPI-only samples for coverage."""
    panel = pd.read_csv(panel_path)
    validate_panel(panel, value_cols=["hpi"])
    taiwan_cpi = pd.read_csv(taiwan_cpi_path)
    quarterly = quarterly_yoy(native_yoy(load_cpi_raw()))
    prepared = prepare_panel(attach_cpi_yoy(panel, quarterly), taiwan_cpi, cfg)
    both = build_samples(prepared, cfg, features_from_config(cfg))
    hpi_cfg = {**cfg, "features": {**cfg["features"], "use": ["hpi_qoq"]}}
    main = build_samples(prepare_panel(panel, taiwan_cpi, hpi_cfg), hpi_cfg, features_from_config(hpi_cfg))
    return both, main


def hpi_only(samples):
    j = samples.feature_names.index("hpi_qoq")
    return samples.replace(X=samples.X[:, :, [j]], feature_names=["hpi_qoq"])


def coverage(both, main, splits, cfg) -> pd.DataFrame:
    """Per market: forecast rows and training rows (first / last fold), main experiment vs CPI experiment."""
    rows = {}
    for name, s in (("main", main), ("cpi", both)):
        ev = pd.concat([s.subset(eval_mask(s, sp, cfg["target"]["eval_series"])).meta for sp in splits])
        first = training_samples(s, splits[0], pool_national()).meta["series_id"].value_counts()
        last = training_samples(s, splits[-1], pool_national()).meta["series_id"].value_counts()
        rows[name] = pd.DataFrame({"forecast_origins": ev["series_id"].value_counts(),
                                   "train_rows_first_fold": first, "train_rows_last_fold": last})
    out = rows["main"].join(rows["cpi"], lsuffix="_main", rsuffix="_cpi", how="outer").fillna(0).astype(int)
    out.index.name = "series_id"
    return out.reset_index().sort_values("series_id")


def same_settings(old: dict | None, new: dict) -> bool:
    return old is not None and all(old.get(k) == new.get(k) for k in (set(old) | set(new)) - IGNORED_KEYS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiment_cpi.toml")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    ap.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv", help="Taiwan monthly CPI index")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(PLAN))
    ap.add_argument("--out-dir", type=Path, default=Path("results/cpi/validation"))
    ap.add_argument("--min-origins", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["features"]["use"] != ["hpi_qoq", "cpi_yoy"]:
        ap.error("the CPI config must use exactly ['hpi_qoq', 'cpi_yoy']")
    both, main_samples = cpi_samples(cfg, args.panel, args.cpi)
    arms = {"hpi_cpi": both, "hpi": hpi_only(both)}
    splits = make_splits(cfg, "validation", last_label_origin(both, end_series(cfg)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cov = coverage(both, main_samples, splits, cfg)
    cov.to_csv(args.out_dir / "coverage.csv", index=False)
    kept = cov[cov["forecast_origins_cpi"] >= args.min_origins]
    print(f"{len(both)} samples with HPI and CPI ({len(main_samples)} in the main experiment); "
          f"{len(kept)} markets with >= {args.min_origins} forecast dates "
          f"({(cov['forecast_origins_main'] >= args.min_origins).sum()} in the main experiment). -> coverage.csv")

    data_files = {"cpi_raw": RAW, "cpi_gaps": GAPS, "cpi_fallback": FALLBACK}
    frames, expected = [], {}
    for model in args.models:
        for label, arm in PLAN[model]:
            base = LINEAR_WITH_CPI.get(model, model) if arm == "hpi_cpi" else model
            stochastic = REGISTRY[base].stochastic
            seeds = cfg["run"]["seeds"] if stochastic else cfg["run"]["seeds"][:1]
            arm_cfg = {**cfg, "features": {**cfg["features"], "use": list(arms[arm].feature_names)}}
            manifest = make_manifest(arm_cfg, experiment="cpi", phase="validation", model=label, base_model=base,
                                     pool="national", stochastic=stochastic, seeds=seeds,
                                     panel_path=args.panel, cpi_path=args.cpi)
            manifest["sample_rule"] = "windows with both hpi_qoq and cpi_yoy; HPI-only arm drops the CPI column"
            manifest["cpi_yoy_files"] = {k: {"path": str(p), "sha256": sha256(p)} for k, p in data_files.items()}
            path = args.out_dir / f"preds_{label}.csv"
            if path.exists() and same_settings(read_manifest(path), manifest):
                print(f"Reusing {path}", flush=True)
                preds = pd.read_csv(path)
            else:
                print(f"Running {label} ({base}, features {arms[arm].feature_names})", flush=True)
                preds = run_backtest(arms[arm], splits, REGISTRY[base], cfg, pool_national(), model_name=label)
                preds.attrs.clear()
                preds.to_csv(path, index=False)
                write_manifest(path, manifest)
            frames.append(preds)
            expected[label] = seeds

    preds = pd.concat(frames, ignore_index=True)
    check_alignment(preds, expected_seeds=expected)
    by_market = metrics_by_market(preds)
    by_market.to_csv(args.out_dir / "metrics_by_market.csv", index=False)
    if "no_change__hpi" in expected:
        rel = relative_to_reference(by_market, "no_change__hpi", args.min_origins)
        summary_by_horizon(rel).to_csv(args.out_dir / "summary_by_horizon.csv", index=False)

    table = []
    for a, b in COMPARISONS:
        if a in expected and b in expected:
            ci = paired_ratio_bootstrap(preds, a, b, cfg, args.min_origins)
            ci.to_csv(args.out_dir / f"ratio_ci_{a}_vs_{b}.csv", index=False)
            band = ci[ci["horizon"].isin(["h3-6", "h16-20"])]
            for r in band.itertuples():
                table.append({"comparison": f"{a} vs {b}", "horizon": r.horizon, "n_markets": r.n_markets,
                              "ratio": f"{r.median_ratio:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"})
    if table:
        t = pd.DataFrame(table).pivot(index="comparison", columns="horizon", values="ratio")
        t = t[[c for c in ("h3-6", "h16-20") if c in t.columns]]
        print("\nMedian RMSE ratio across markets (below 1 = first model better), 95% CI:")
        print(t.to_string())


if __name__ == "__main__":
    main()
