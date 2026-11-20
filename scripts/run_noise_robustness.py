"""Input-noise robustness check on the validation folds (no new data, no test period).

Each model is trained exactly as in the main backtest. Its forecasts are then repeated with
Gaussian noise added to the input HPI growth. One perturbation is drawn per market x input
quarter x draw, so overlapping windows, folds and models all share it (each historical value
is measured wrong once). Noise SD = level x one fixed scale: the SD of HPI growth in the first
fold's training inputs (before the validation period). Training data are not changed.

    OMP_NUM_THREADS=1 python scripts/run_noise_robustness.py            # all models, then summary
    python scripts/run_noise_robustness.py --models ar ar_pooled no_change
    python scripts/run_noise_robustness.py --summary-only               # redo table and plot

Writes to results/robustness/noise/validation/ (use a new --out-dir for other settings):
  errors_<model>.csv / .json   error sums per market, horizon, level, draw and seed, with the
                               settings and data fingerprints they were produced with
  noise_summary.csv            medians across markets, per horizon and for 3-6 / 16-20 quarters
  noise_curve.png              the stability curves

Three numbers per model and noise level, each the median across markets (>= 8 forecast dates;
RMSE pooled over the range's horizons as in the main experiment, averaged over seeds):
  mean_abs_change   mean |forecast with noise - forecast without noise|, in cumulative log growth:
                    how much the forecast itself moves
  vs_clean          RMSE(noisy) / RMSE(same model without noise): how much accuracy is lost
  vs_no_change      RMSE(noisy) / RMSE(no-change): above 1 = worse than no forecast
Ranges show the lowest and highest of the noise draws. No-change ignores its inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_backtest import load_samples  # noqa: E402

from housing_forecast.backtesting.robustness import band_metrics, noise_backtest  # noqa: E402
from housing_forecast.backtesting.runner import pool_national  # noqa: E402
from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits  # noqa: E402
from housing_forecast.config import load_config  # noqa: E402
from housing_forecast.models.registry import REGISTRY  # noqa: E402

DEFAULT_MODELS = ["no_change", "ar", "ar_pooled", "transformer", "transformer_per_market"]
NAMES = {
    "transformer__national": "Shared Transformer",
    "transformer_per_market__national": "Per-market Transformer",
    "ar__national": "Per-market AR",
    "ar_pooled__national": "Pooled AR",
    "no_change__national": "No-change",
}
NO_CHANGE = "no_change__national"
BANDS = {"h3-6": range(3, 7), "h16-20": range(16, 21)}
MIN_ORIGINS = 8


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def current_settings(args) -> dict:
    return {
        "phase": "validation",
        "feature": "hpi_qoq",
        "levels": [0.0] + sorted(l for l in args.levels if l > 0),
        "draws": args.draws,
        "noise_seed": args.noise_seed,
        "noise": "level x SD of hpi_qoq in the first fold's training inputs x z(market, quarter, draw)",
        "config": args.config, "config_sha256": sha256(args.config),
        "panel_sha256": sha256(args.panel), "cpi_sha256": sha256(args.cpi),
    }


def run_model(model: str, samples, splits, cfg, args, settings: dict) -> None:
    label = f"{model}__national"
    path = args.out_dir / f"errors_{label}.csv"
    side = path.with_suffix(".json")
    if path.exists() and not args.overwrite:
        if not side.exists() or json.loads(side.read_text()) != settings:
            raise SystemExit(f"{path} was made with other settings or data. Use a new --out-dir "
                             "(or --overwrite to replace every model in this folder).")
        print(f"Reusing {path}", flush=True)
        return
    print(f"Running {label} ({len(splits)} folds, levels {settings['levels']}, {args.draws} draws)", flush=True)
    errors = noise_backtest(samples, splits, REGISTRY[model], cfg, pool_national(), levels=args.levels,
                            draws=args.draws, noise_seed=args.noise_seed, model_name=label)
    errors.to_csv(path, index=False)
    side.write_text(json.dumps(settings, indent=2))
    print(f"Saved {path} (noise scale: SD {errors.attrs['noise_sd']:.4f})", flush=True)

    reference = Path(args.reference_dir) / "metrics_by_market.csv"
    if reference.exists():
        clean = errors[errors["noise_level"].eq(0)].copy()
        clean["rmse"] = (clean["sse"] / clean["n"]) ** 0.5
        clean = clean.groupby(["series_id", "horizon"])["rmse"].mean().reset_index()
        ref = pd.read_csv(reference)
        ref = ref[ref["model"].eq(label)][["series_id", "horizon", "rmse"]]
        both = clean.merge(ref, on=["series_id", "horizon"], suffixes=("", "_main"))
        if len(both):
            diff = (both["rmse"] / both["rmse_main"] - 1).abs()
            print(f"  no-noise check vs {reference}: median |RMSE difference| {diff.median():.2%}, "
                  f"max {diff.max():.2%}", flush=True)


def load_errors(out_dir: Path) -> pd.DataFrame:
    files = sorted(out_dir.glob("errors_*.csv"))
    if not files:
        raise SystemExit(f"no errors_*.csv in {out_dir}")
    settings = {f.name: json.loads(f.with_suffix(".json").read_text()) if f.with_suffix(".json").exists() else None
                for f in files}
    if None in settings.values() or len({json.dumps(s, sort_keys=True) for s in settings.values()}) > 1:
        raise SystemExit(f"The files in {out_dir} were made with different settings or data: {list(settings)}. "
                         "Keep one set of settings per folder.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarize(out_dir: Path) -> pd.DataFrame:
    errors = load_errors(out_dir)
    groups = {f"h{h}": [h] for h in range(1, 21)} | {k: list(v) for k, v in BANDS.items()}
    rows = []
    for band, hs in groups.items():
        m = band_metrics(errors, hs, MIN_ORIGINS)
        clean = m[m["noise_level"].eq(0)][["model", "series_id", "rmse"]].rename(columns={"rmse": "rmse_clean"})
        m = m.merge(clean, on=["model", "series_id"])
        m["vs_clean"] = m["rmse"] / m["rmse_clean"]
        has_nc = (m["model"] == NO_CHANGE).any()
        if has_nc:
            nc = clean[clean["model"].eq(NO_CHANGE)][["series_id", "rmse_clean"]].rename(columns={"rmse_clean": "rmse_nc"})
            m = m.merge(nc, on="series_id")
            m["vs_no_change"] = m["rmse"] / m["rmse_nc"]
        stats = ["mean_abs_change", "vs_clean"] + (["vs_no_change"] if has_nc else [])
        med = m.groupby(["model", "noise_level", "draw"]).agg(
            n_markets=("series_id", "nunique"), **{s: (s, "median") for s in stats}).reset_index()
        agg = {"n_markets": ("n_markets", "max")}
        for s in stats:
            agg |= {s: (s, "mean"), f"{s}_min": (s, "min"), f"{s}_max": (s, "max")}
        rows.append(med.groupby(["model", "noise_level"]).agg(**agg).reset_index().assign(horizon=band))
    summary = pd.concat(rows, ignore_index=True)
    summary.insert(1, "model_name", summary["model"].map(NAMES).fillna(summary["model"]))
    summary.to_csv(out_dir / "noise_summary.csv", index=False)
    return summary


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("mean_abs_change", "Mean |forecast change|\n(cumulative log growth)"),
              ("vs_clean", "RMSE relative to the same\nmodel without noise")]
    if "vs_no_change" in summary:
        panels.append(("vs_no_change", "RMSE relative to no-change\n(above 1 = worse)"))
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 3.3 * len(panels)), squeeze=False)
    styles = {"Shared Transformer": ("#1f4e79", "o", "-"), "Per-market Transformer": ("#6fa8dc", "s", "--"),
              "Per-market AR": ("#b45f06", "^", "--"), "Pooled AR": ("#e69138", "v", "-"),
              "No-change": ("#777777", "", ":")}
    for r, (col, ylabel) in enumerate(panels):
        for c, band in enumerate(BANDS):
            ax = axes[r, c]
            for name, g in summary[summary["horizon"].eq(band)].groupby("model_name"):
                g = g.sort_values("noise_level")
                color, marker, ls = styles.get(name, ("black", "o", "-"))
                ax.plot(g["noise_level"], g[col], color=color, marker=marker, ls=ls, label=name)
                ax.fill_between(g["noise_level"], g[f"{col}_min"], g[f"{col}_max"], color=color, alpha=0.12)
            if col != "mean_abs_change":
                ax.axhline(1, color="black", lw=0.8)
            if r == 0:
                ax.set_title(f"{band[1:].replace('-', '–')} quarters ahead")
            if r == len(panels) - 1:
                ax.set_xlabel("Noise SD / SD of HPI growth (first training fold)")
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Stability under input noise (validation 2000–2009, median across markets)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "noise_curve.png", dpi=150)
    print(f"-> {out_dir / 'noise_curve.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    ap.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(REGISTRY))
    ap.add_argument("--levels", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0],
                    help="noise SD as a multiple of the first fold's training SD of HPI growth")
    ap.add_argument("--draws", type=int, default=3, help="noise draws per level")
    ap.add_argument("--noise-seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, default=Path("results/robustness/noise/validation"))
    ap.add_argument("--reference-dir", default="results/long_hpi_122/validation",
                    help="main validation results, used to check the no-noise run reproduces them")
    ap.add_argument("--overwrite", action="store_true", help="replace every model's file in --out-dir")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        settings = current_settings(args)
        if args.overwrite:
            for f in list(args.out_dir.glob("errors_*.csv")) + list(args.out_dir.glob("errors_*.json")):
                f.unlink()
        cfg = load_config(args.config)
        samples = load_samples(cfg, args.panel, args.cpi)
        splits = make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))
        for model in args.models:
            run_model(model, samples, splits, cfg, args, settings)

    summary = summarize(args.out_dir)
    bands = summary[summary["horizon"].isin(BANDS)]
    cols = ["model_name", "horizon", "noise_level", "mean_abs_change", "vs_clean"]
    cols += ["vs_no_change"] if "vs_no_change" in bands else []
    pd.set_option("display.width", 200)
    print(bands[cols].round(3).to_string(index=False))
    plot(summary, args.out_dir)


if __name__ == "__main__":
    main()
