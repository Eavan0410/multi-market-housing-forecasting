"""Score forecasts and bootstrap paired ratios. Use --plot-failure-cases on saved metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.backtesting.manifest import check_comparable, read_manifest
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment, validate_predictions
from housing_forecast.evaluation.markets import (
    history_groups, metrics_by_market, paired_ratio_bootstrap, relative_to_reference, summary_by_history,
    summary_by_horizon,
)


CURVE_COLORS = {
    "no_change__national": "tab:grey",
    "ar__national": "tab:blue",
    "ar_pooled8__national": "tab:orange",
    "ar_pooled__national": "tab:green",
    "arima__national": "tab:red",
    "transformer__national": "tab:purple",
    "transformer_per_market__national": "tab:brown",
}


def plot_curve(summary: pd.DataFrame, reference: str, path: Path, *, export_pdf=False) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    labels = {
        "transformer__national": "Shared Transformer",
        "transformer_per_market__national": "Per-market Transformer (with fallback)",
        "ar__national": "Per-market AR (with fallback)",
        "arima__national": "Per-market ARIMA (with fallback)",
        "ar_pooled__national": "Pooled AR(4)",
        "ar_pooled8__national": "Pooled AR(8)",
        "no_change__national": "No-change",
    }
    with plt.rc_context(rc=matplotlib.rcParamsDefault):
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, g in summary[summary["model"] != reference].groupby("model"):
            g = g.sort_values("horizon")
            ax.plot(g["horizon"], g["median_rel_rmse"], marker="o", ms=3,
                    color=CURVE_COLORS.get(model), label=labels.get(model, model))
        ax.axhline(1.0, color="grey", lw=1, ls="--")
        ax.axvspan(3, 6, color="tab:blue", alpha=0.06)
        ax.axvspan(16, 20, color="tab:orange", alpha=0.06)
        ax.set_xlabel("Forecast horizon (quarters)")
        ax.set_ylabel(f"Median RMSE ratio to {labels.get(reference, reference).lower()}")
        ax.set_title("Forecast error across markets (below 1 = better than the reference)")
        ax.set_xticks([1, 3, 6, 10, 16, 20])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=300 if export_pdf else 150)
        if export_pdf:
            fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    return True


def plot_failure_cases(metrics: pd.DataFrame, markets: list[str], path: Path,
                       min_origins: int = 8) -> None:
    """Use saved seed-averaged RMSE, dividing within each market and horizon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(markets), figsize=(5 * len(markets), 4), squeeze=False)
    for ax, market in zip(axes[0], markets):
        rows = metrics[(metrics["series_id"] == market) & (metrics["n_origins"] >= min_origins)]
        errors = rows.pivot(index="horizon", columns="model", values="rmse").sort_index()
        for baseline, label in [("ar__national", "AR per market (with fallback)"),
                                ("ar_pooled__national", "Pooled AR")]:
            ratio = errors["transformer__national"] / errors[baseline].where(errors[baseline] > 0)
            ax.plot(ratio.index, ratio, label=label)
        ax.axhline(1.0, color="grey", lw=1, ls="--")
        ax.axvspan(3, 6, color="tab:blue", alpha=0.06)
        ax.axvspan(16, 20, color="tab:orange", alpha=0.06)
        ax.set_title(market.replace("_national", ""))
        ax.set_xlabel("Forecast horizon (quarters)")
        ax.set_xticks([1, 3, 6, 10, 16, 20])
        ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel("Transformer / baseline RMSE (>1 = worse)")
    axes[0, 0].legend(fontsize=8)
    phase = "test" if "test" in path.parent.parts else "validation"
    fig.suptitle(f"Selected failure cases — saved {phase} errors, RMSE averaged over seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="prediction CSVs, or one metrics CSV with --plot-failure-cases")
    ap.add_argument("--plot-failure-cases", action="store_true",
                    help="only plot saved metrics; do not recompute or overwrite result tables")
    ap.add_argument("--markets", nargs="+", default=["Taiwan_national", "Germany", "Mexico"],
                    help="markets for --plot-failure-cases")
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--reference", default="no_change")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv",
                    help="used only to group markets by HPI history length")
    ap.add_argument("--min-origins", type=int, default=8)
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="bootstrap CI for median RMSE(A)/RMSE(B) across markets; repeatable")
    ap.add_argument("--block-length", type=int, default=None,
                    help="bootstrap block length in origins (default: [bootstrap] block_length)")
    args = ap.parse_args()

    if args.plot_failure_cases:
        if len(args.preds) != 1:
            ap.error("--plot-failure-cases requires one saved metrics_by_market.csv")
        source = Path(args.preds[0])
        path = source.parent / "failure_cases.png"
        plot_failure_cases(pd.read_csv(source), args.markets, path, args.min_origins)
        print(f"plot -> {path}")
        return

    cfg = load_config(args.config)
    frames, manifests, expected = [], {}, {}
    for p in args.preds:
        df = pd.read_csv(p)
        frames.append(df)
        m = read_manifest(p)
        if m is None:
            print(f"warning: {p} has no run manifest; its settings cannot be checked")
            det = set(cfg["run"]["deterministic_models"])
            for model in df["model"].unique():
                expected[model] = cfg["run"]["seeds"][:1] if model.split("__")[0] in det else cfg["run"]["seeds"]
        else:
            manifests[p] = m
            expected[m["model"]] = m["seeds"]
    check_comparable(manifests)
    preds = pd.concat(frames, ignore_index=True)
    validate_predictions(preds)
    check_alignment(preds, expected_seeds=expected)
    out = Path(args.preds[0]).parent
    comparisons = [paired_ratio_bootstrap(preds, a, b, cfg, args.min_origins, args.block_length)
                   for a, b in args.compare]

    by_market = metrics_by_market(preds)
    by_market.to_csv(out / "metrics_by_market.csv", index=False)
    rel = relative_to_reference(by_market, args.reference, args.min_origins)
    rel.to_csv(out / "relative_by_market.csv", index=False)
    summary = summary_by_horizon(rel)
    summary.to_csv(out / "summary_by_horizon.csv", index=False)
    print(f"Relative RMSE vs {args.reference} across markets (below 1 = better):")
    print(summary.pivot(index="horizon", columns="model", values="median_rel_rmse").round(3).to_string())

    if Path(args.panel).exists():
        groups = history_groups(pd.read_csv(args.panel))
        hist = summary_by_history(rel, groups)
        hist.to_csv(out / "summary_by_history.csv", index=False)
        print("\nBy HPI history length (median relative RMSE):")
        print(hist.pivot_table(index=["history", "horizon_band"], columns="model",
                               values="median_rel_rmse").round(3).to_string())
    if plot_curve(summary, args.reference, out / "rel_rmse_by_horizon.png"):
        print(f"\nplot -> {out / 'rel_rmse_by_horizon.png'}")

    for (a, b), ci in zip(args.compare, comparisons):
        path = out / f"ratio_ci_{a}_vs_{b}.csv"
        ci.to_csv(path, index=False)
        print(f"\nMedian RMSE({a}) / RMSE({b}) across markets, {cfg['bootstrap']['ci']:.0%} CI -> {path.name}")
        print(ci[["horizon", "n_markets", "median_ratio", "ci_low", "ci_high", "share_draws_below_1"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
