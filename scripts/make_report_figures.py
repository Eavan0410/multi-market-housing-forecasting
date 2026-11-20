"""Draw report figures from saved forecasts and summaries; export PNG and PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_markets import CURVE_COLORS, plot_curve

BLUE, DARK, GRAY, LIGHT = "#1f77b4", "#222222", "#808080", "#e5e5e5"
GREEN = "#2ca02c"
MODELS = {
    "transformer__national": ("Shared Transformer", GREEN, "-"),
    "transformer_per_market__national": ("Per-market Transformer", "#9467bd", "-"),
    "ar__national": ("Per-market AR", BLUE, "-"),
    "arima__national": ("Per-market ARIMA", "#d62728", "-"),
    "ar_pooled__national": ("Pooled AR(4)", "#ff7f0e", "-"),
    "ar_pooled8__national": ("Pooled AR(8)", "#8c564b", "--"),
    "no_change__national": ("No-change", GRAY, "-"),
}
BASELINES = list(MODELS)[1:]
BANDS = {"h3-6": "3–6 quarters", "h16-20": "16–20 quarters"}
ORDERS = {
    "training_pool_region": ("Region order", BLUE, "o", "-"),
    "training_pool_history": ("History order", "#ff7f0e", "s", "-"),
    "training_pool_random": ("Random order", GREEN, "^", "-"),
}
# Match the defaults used by evaluate_markets.plot_curve.
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({"legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})


def reference(ax, axis="x", value=1):
    draw = ax.axvline if axis == "x" else ax.axhline
    draw(value, color=GRAY, linewidth=1.1, linestyle="--", zorder=0)


def grid(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(False)


def save(fig, out, name):
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(name)


def panel_title(ax, letter, title):
    ax.set_title(f"({letter}) {title}", pad=10)


def summary_frame(summary, section, phase, markets="all"):
    return summary[(summary.section == section) & (summary.phase == phase)
                   & (summary.markets == markets)].copy()


def fig1_horizon(res, out):
    for phase in ("test",):
        data = pd.read_csv(res / f"long_hpi_122/{phase}/summary_by_horizon.csv")
        name = "fig1_horizon" if phase == "test" else "fig1_horizon_validation"
        plot_curve(data, "no_change__national", out / f"{name}.png", export_pdf=True)
        print(name)


def get_cell(frame, **filters):
    for key, value in filters.items():
        frame = frame[frame[key] == value]
    if frame.empty:
        return None
    if len(frame) != 1:
        raise ValueError(f"Ambiguous table entry: {filters}")
    return frame.iloc[0]


def market_band_ratios(res, min_origins):
    pieces = []
    for model in ("transformer__national", "transformer_per_market__national"):
        d = pd.read_csv(res / f"long_hpi_122/test/preds_{model}.csv",
                        usecols=["series_id", "origin_quarter", "horizon", "seed", "y_pred", "y_true"])
        d["se"] = (d.y_pred - d.y_true) ** 2
        for band, (lo, hi) in zip(BANDS, ((3, 6), (16, 20))):
            q = d[d.horizon.between(lo, hi)]
            eligible = q.groupby("series_id").origin_quarter.nunique()
            q = q[q.series_id.isin(eligible[eligible >= min_origins].index)]
            rmse = q.groupby(["series_id", "seed"]).se.mean().pow(0.5)
            rmse = rmse.groupby("series_id").mean().rename("rmse").reset_index()
            pieces.append(rmse.assign(model=model, horizon=band))
    table = pd.concat(pieces).pivot(index=["series_id", "horizon"], columns="model", values="rmse")
    return (table["transformer__national"] / table["transformer_per_market__national"]).rename("ratio").reset_index()


def fig3_markets(res, docs, out):
    data = market_band_ratios(res, 8)
    groups = pd.read_csv(docs / "market_list.csv")[["series_id", "history"]]
    groups["group"] = groups.history.str.split().str[0]
    data = data.merge(groups[["series_id", "group"]], on="series_id", validate="many_to_one")
    names = {"long": "HPI starts by 1979", "medium": "Starts 1980–1999", "short": "Starts 2000 or later"}
    if data.group.isna().any():
        raise ValueError("Missing history group")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True, sharey=True)
    for ax, band, letter in zip(axes, BANDS, "ab"):
        labels = []
        for i, (group, name) in enumerate(names.items()):
            q = data[(data.horizon == band) & (data.group == group)].sort_values(["ratio", "series_id"])
            y = 2 - i
            # Deterministic vertical offsets separate nearby market points.
            offsets = (np.arange(len(q)) % 5 - 2) * 0.055
            ax.scatter(q.ratio, y + offsets, s=24, color=GREEN, alpha=0.8, linewidths=0, zorder=2)
            median = q.ratio.median()
            ax.plot(median, y, "|", ms=17, color=DARK, mew=1.4, zorder=3)
            labels.append(f"{name} (n={len(q)})")
            ax.text(0.98, y + 0.26, f"Median {median:.2f}", transform=ax.get_yaxis_transform(),
                    ha="right", fontsize=8, color=DARK)
        ax.set_yticks([2, 1, 0], labels)
        ax.set_ylim(-0.45, 2.55)
        ax.spines.left.set_visible(True)
        reference(ax)
        grid(ax)
        panel_title(ax, letter, BANDS[band])
        ax.set_xlabel("RMSE ratio: shared / per-market Transformer")
    fig.subplots_adjust(left=0.23, right=0.99, bottom=0.18, top=0.87, wspace=0.12)
    save(fig, out, "fig3_markets")


def fig4_pool(summary, out):
    sizes = np.array([5, 10, 20, 41])
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.7), sharex=True, sharey="row")
    for j, phase in enumerate(("validation", "test")):
        for i, band in enumerate(BANDS):
            ax = axes[i, j]
            for k, (section, (label, color, marker, style)) in enumerate(ORDERS.items()):
                d = summary[(summary.section == section) & (summary.phase == phase) & (summary.horizon == band)]
                rows = [d[d.model_a == f"transformer__scale_{n}"].iloc[0] for n in sizes[1:]]
                values = [1] + [r.median_ratio for r in rows]
                ax.plot(sizes, values, color=color, ls=style, lw=1.1, label=label)
                for n, r in zip(sizes[1:], rows):
                    if n == 41 and k > 0:
                        continue
                    x = n + (k - 1) * 0.45 if n != 41 else n
                    ax.errorbar(x, r.median_ratio,
                                yerr=[[r.median_ratio-r.ci_low], [r.ci_high-r.median_ratio]],
                                fmt=marker if n != 41 else "o", color=color if n != 41 else DARK,
                                ms=3.5, lw=0.8, capsize=2)
            ax.plot(5, 1, "o", ms=3.5, color=DARK)
            reference(ax, "y")
            grid(ax, "y")
            ax.set_xticks(sizes)
            ax.set_xlim(3, 43)
            panel_title(ax, "abcd"[i * 2 + j], f"{phase.capitalize()} · {BANDS[band]}")
            if j == 0:
                ax.set_ylabel("RMSE / 5-market pool RMSE")
            if i == 1:
                ax.set_xlabel("Training markets")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.14, wspace=0.13, hspace=0.43)
    save(fig, out, "fig4_training_pool")


def fig5_level(res, out):
    pieces = []
    for n in (5, 41):
        d = pd.read_csv(res / f"market_scale_122/test/preds_transformer__scale_{n}.csv")
        d = d[d.horizon.between(16, 20)]
        q = d.groupby("series_id").agg(pred=("y_pred", "mean"), actual=("y_true", "mean")).reset_index()
        pieces.append(q.assign(pool=n))
    data = pd.concat(pieces)
    actual = data.groupby("series_id").actual.first().sort_values()
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for pool, color, marker, label in [(5, "#ff7f0e", "o", "5-market pool"), (41, GREEN, "s", "41-market pool")]:
        q = data[data.pool == pool].set_index("series_id").loc[actual.index]
        ax.plot(q.pred * 100, np.arange(len(q)), ls="none", marker=marker, ms=5,
                color=color, mfc="white" if pool == 5 else color, label=label, zorder=3)
    ax.plot(actual * 100, np.arange(len(actual)), "D", ls="none", color=DARK, ms=4.5, label="Observed")
    reference(ax, value=0)
    ax.set_yticks(range(len(actual)), actual.index)
    ax.invert_yaxis()
    ax.set_ylim(len(actual)-0.5, -0.5)
    ax.spines.left.set_visible(True)
    grid(ax)
    ax.set_xlabel("Mean cumulative real HPI growth (100 × log change)")
    ax.set_title("Test · 16–20 quarters", pad=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.86, bottom=0.26)
    save(fig, out, "fig5_pool_level")


def comparison_bars(series, band, out, name, title):
    x = np.arange(len(BASELINES))
    width = 0.76 / len(series)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (data, label, color) in enumerate(series):
        rows = [get_cell(data, model_b=model, horizon=band) for model in BASELINES]
        if any(r is None for r in rows):
            raise ValueError(f"Incomplete bar comparison: {label}, {band}")
        values = np.array([r.median_ratio for r in rows])
        errors = np.array([[r.median_ratio - r.ci_low for r in rows],
                           [r.ci_high - r.median_ratio for r in rows]])
        ax.bar(x + (i - (len(series)-1)/2) * width, values, width=width,
               color=color, label=label, yerr=errors, capsize=3,
               error_kw={"elinewidth": 1, "capthick": 1}, zorder=2)
    reference(ax, "y")
    labels = ["Per-market\nTransformer", "Per-market\nAR", "Per-market\nARIMA",
              "Pooled AR(4)", "Pooled AR(8)", "No-change"]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median RMSE ratio: shared / baseline")
    ax.set_title(title)
    ax.set_ylim(0, 1.45)
    ax.legend(loc="upper right", ncol=len(series), fontsize=8)
    fig.text(0.12, 0.015, "Error bars: 95% paired bootstrap intervals, 20-quarter blocks. Below 1 favors the shared model.", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.9)
    save(fig, out, name)


def fig2_main(summary, res, out):
    main = summary[(summary.section == "main") & (summary.markets == "all")]
    band = "h16-20"
    series = [(main[main.phase == "validation"], "Validation (38 markets)", "#aeb4bb"),
              (main[main.phase == "test"], "Test (48 markets)", BLUE)]
    comparison_bars(series, band, out, "fig2_main_comparison",
                    f"Shared Transformer versus baselines: {BANDS[band]}")


def figA_holdouts(summary, out):
    from matplotlib.colors import TwoSlopeNorm
    markets = ["australia", "chile", "taiwan_national", "portugal"]
    bases = ["transformer__national", "ar__national", "no_change__national"]
    selected = summary[summary.section.isin([f"holdout_{m}" for m in markets]) & summary.model_b.isin(bases)]
    lower = min(0.35, selected.median_ratio.min())
    upper = max(2.5, selected.median_ratio.max())
    norm = TwoSlopeNorm(vmin=lower, vcenter=1, vmax=upper)
    for phase in ("test",):
        for band in ("h16-20",):
            data = np.array([[get_cell(selected, section=f"holdout_{market}", phase=phase,
                                        markets="all", horizon=band, model_b=base).median_ratio
                              for base in bases] for market in markets])
            fig, ax = plt.subplots(figsize=(7, 4.6))
            image = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    rgba = image.cmap(norm(data[i, j]))
                    lightness = np.dot(rgba[:3], [0.299, 0.587, 0.114])
                    ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                            fontsize=10, color="white" if lightness < 0.5 else DARK)
            ax.set_xticks(range(3), ["Full shared model", "Per-market AR", "No-change"])
            ax.set_yticks(range(4), [m.replace("_national", "").title() for m in markets])
            ax.set_title(f"Market holdouts: {phase}, {BANDS[band]}")
            ax.tick_params(axis="both", length=0)
            ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", length=0)
            bar = fig.colorbar(image, ax=ax, pad=0.03)
            bar.set_label("RMSE ratio: held-out / baseline")
            bar.set_ticks([0.5, 1, 1.5, 2, 2.5])
            fig.text(0.15, 0.025, "Blue: lower error; red: higher error. Color does not indicate statistical significance.", fontsize=8)
            fig.subplots_adjust(left=0.17, right=0.91, bottom=0.14, top=0.89)
            suffix = ("_validation" if phase == "validation" else "") + ("_short" if band == "h3-6" else "")
            save(fig, out, "figA_holdouts" + suffix)


def figB_blocks(res, out):
    data = pd.read_csv(res / "block_length_sensitivity_122_test.csv")
    data = data[(data.experiment == "main") & data.horizon.isin(BANDS)].copy()
    data["interval_width"] = data.ci_high - data.ci_low
    with plt.rc_context(rc=matplotlib.rcParamsDefault):
        for band in ("h16-20",):
            fig, ax = plt.subplots(figsize=(9, 5))
            for model, group in data[data.horizon == band].groupby("model_b"):
                d = group.sort_values("block_length")
                ax.plot(d.block_length, d.interval_width, marker="o", ms=3,
                        label=MODELS[model][0], color=CURVE_COLORS[model])
            ax.set_xticks([4, 8, 12, 20])
            ax.set_ylim(0, data.interval_width.max() * 1.28)
            ax.set_xlabel("Bootstrap block length (quarters)")
            ax.set_ylabel("95% confidence interval width")
            ax.set_title(f"Uncertainty by block length: test, {BANDS[band]}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            save(fig, out, "figB_block_length" + ("_short" if band == "h3-6" else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(res / "test_summary.csv")
    fig1_horizon(res, out)
    fig2_main(summary, res, out)
    fig3_markets(res, Path(args.docs), out)
    fig4_pool(summary, out)
    fig5_level(res, out)
    figA_holdouts(summary, out)
    figB_blocks(res, out)
    (out / "captions.txt").write_text(
        "1. Test-period median per-market RMSE relative to no-change. Shading marks 3–6 and 16–20 quarters. "
        "Market coverage varies with horizon.\n\n"
        "2. Median market RMSE(shared)/RMSE(baseline), with 95% paired circular-block bootstrap "
        "intervals (20-quarter blocks), 16–20 quarters ahead. Validation: 38 markets; test: 48 markets. "
        "Ratios below 1 favor the shared model.\n\n"
        "3. One dot per test market; vertical marks show history-group medians. "
        "Within each horizon band, pool squared errors, take RMSE per seed and average across seeds. "
        "Group differences are descriptive; history length is not isolated from market composition.\n\n"
        "4. RMSE relative to the five-market training pool on the same five evaluation markets. "
        "Intervals use 20-quarter blocks. Endpoints are shared across addition orders. "
        "The two periods use the same vertical scale within each horizon range.\n\n"
        "5. Mean observed and predicted cumulative real HPI log growth across 16–20-quarter "
        "test forecasts, averaged over dates, horizons and seeds. Values are log change × 100, "
        "not percentage changes. Mean bias is descriptive and does not establish the cause of RMSE differences.\n\n"
        "A. Test, 16–20 quarters. Target markets excluded from training; their recent HPI windows remain available at forecast time. "
        "Median RMSE ratios and 95% intervals use 20-quarter blocks.\n\n"
        "B. Main test comparison at 16–20 quarters under four bootstrap block lengths. The point estimate is unchanged; "
        "only the uncertainty calculation differs. The figure plots interval width; full bounds are in results/block_length_sensitivity_122_test.csv.\n"
    )


if __name__ == "__main__":
    main()
