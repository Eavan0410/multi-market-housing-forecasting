"""Score multi-market runs: per-market errors, error-by-horizon curve, history-length split.

Example (after run_backtest.py --config configs/experiment_long.toml ...):
    python scripts/evaluate_markets.py results/long_hpi/validation/preds_*.csv \\
        --config configs/experiment_long.toml --reference no_change

Writes next to the first input file:
    metrics_by_market.csv     RMSE / MAE / direction accuracy per model, market and horizon
    relative_by_market.csv    the same divided by the reference model's errors in that market
    summary_by_horizon.csv    per model and horizon: median and geometric-mean relative RMSE
                              across markets, share of markets beating the reference
    summary_by_history.csv    the same split by how long each market's HPI history is
    rel_rmse_by_horizon.png   median relative RMSE against horizon (if matplotlib is installed)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.backtesting.manifest import check_comparable, read_manifest
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment, validate_predictions
from housing_forecast.evaluation.markets import (
    history_groups, metrics_by_market, relative_to_reference, summary_by_history, summary_by_horizon,
)


def plot_curve(summary: pd.DataFrame, reference: str, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, g in summary[summary["model"] != reference].groupby("model"):
        ax.plot(g["horizon"], g["median_rel_rmse"], marker="o", ms=3, label=model)
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    ax.axvspan(3, 6, color="tab:blue", alpha=0.06)
    ax.axvspan(16, 20, color="tab:orange", alpha=0.06)
    ax.set_xlabel("horizon (quarters)")
    ax.set_ylabel(f"median RMSE / {reference} RMSE")
    ax.set_title("Error relative to the reference, across markets (below 1 = better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--reference", default="no_change")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv",
                    help="used only to group markets by HPI history length")
    ap.add_argument("--min-origins", type=int, default=8)
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
