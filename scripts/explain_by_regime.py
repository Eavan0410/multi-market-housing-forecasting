"""When does the shared Transformer weight the inputs differently from pooled AR? (step 3)

Uses the per-forecast weights, input windows and AR coefficients saved by
explain_lag_weights.py (no retraining, no data rebuild). Before grouping, it checks that the
saved file and the forecast files come from the same run: the shared-Transformer forecasts must
have the fingerprint recorded in row_attributions.npz, the AR forecast files must come from the
same panel, and every saved forecast must match exactly one row of each forecast file.

Main comparison: pooled AR(8), which sees the same eight quarters as the Transformer, so a
difference cannot come from a longer window. Pooled AR(4) is kept as a second reference.

Groups (known at the forecast date):
  past-year growth   sum of the last four quarters of real HPI growth: falling (< 0),
                     slow (0-4%), fast (4-8%), boom (> 8%)
  turning point      last two quarters vs the four before: steady rise, steady fall,
                     turning up, turning down
  period             forecast dates 2000-2006 vs 2007-2009
  history            long / medium / short HPI history (docs/market_list.csv)
  market             markets with at least 8 forecast dates (weights_by_market.csv)

Columns, for the 3-6 and 16-20 quarter ranges:
  tr_recent, tr_recent_p10, tr_recent_p90   Transformer weight on the latest two quarters (mean,
                                            10th and 90th percentile across the group's forecasts)
  ar8_recent, ar4_recent                    the same for pooled AR (fixed within a fold)
  tr_older, ar8_older                       weight on quarters 5-8 (AR(4) has none by design)
  rmse_vs_ar8, rmse_vs_ar4                  RMSE(Transformer) / RMSE(AR) on the group's forecasts:
                                            squared errors pooled over markets, dates and the
                                            range's horizons, square root per seed, mean over seeds.
                                            Descriptive (pools markets, no interval).
weights_within_fold.csv shows, for each fold, how much the Transformer's weight on the latest
two quarters varies across that fold's forecasts; AR's weight does not vary within a fold.

    python scripts/explain_by_regime.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = {"h3-6": range(3, 7), "h16-20": range(16, 21)}
GROWTH_BINS = [-np.inf, 0.0, 0.04, 0.08, np.inf]
GROWTH_LABELS = ["falling (<0%)", "slow (0-4%)", "fast (4-8%)", "boom (>8%)"]
MODELS = {"tr": "transformer__national", "ar8": "ar_pooled8__national", "ar4": "ar_pooled__national"}
MIN_ORIGINS = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checked(z, preds_dir: Path) -> dict[str, pd.DataFrame]:
    """Forecast files, after checking they belong to the run saved in row_attributions.npz."""
    if "prov_reference_sha256" not in z.files or "x_window" not in z.files:
        raise SystemExit("row_attributions.npz is from an older version; rerun explain_lag_weights.py")
    preds = {}
    for key, name in MODELS.items():
        path = preds_dir / f"preds_{name}.csv"
        if key == "tr":
            if sha256(path) != str(z["prov_reference_sha256"]):
                raise SystemExit(f"{path} is not the forecast file the attributions were checked against")
        else:
            meta = json.loads(path.with_suffix(".meta.json").read_text()) if path.with_suffix(".meta.json").exists() else {}
            if meta.get("panel_sha256") != str(z["prov_panel_sha256"]):
                raise SystemExit(f"{path} was made from a different panel than the attributions")
        preds[key] = pd.read_csv(path)
    return preds


def situations(meta: pd.DataFrame, x_window: np.ndarray, market_list: pd.DataFrame) -> pd.DataFrame:
    g = x_window[:, ::-1]                                   # (n, L), newest first
    past_year = g[:, :4].sum(axis=1)
    recent, before = g[:, :2].sum(axis=1), g[:, 2:6].sum(axis=1)
    turn = np.select([(recent >= 0) & (before >= 0), (recent < 0) & (before < 0), recent >= 0],
                     ["steady rise", "steady fall", "turning up"], "turning down")
    year = pd.PeriodIndex(meta["origin_quarter"], freq="Q").year
    history = meta["series_id"].map(market_list.set_index("series_id")["history"].str.split(" ").str[0])
    return meta.assign(past_year_growth=pd.cut(past_year, GROWTH_BINS, labels=GROWTH_LABELS).astype(str),
                       turning_point=turn, period=np.where(year >= 2007, "2007-2009", "2000-2006"),
                       history=history.fillna("unknown"), past_year=past_year)


def matched(preds: pd.DataFrame, keys: pd.DataFrame, hs, name: str) -> pd.DataFrame:
    """Forecast rows of the range's horizons joined to the saved forecasts; every row must match."""
    p = preds[preds["horizon"].isin(list(hs))]
    out = p.merge(keys, on=["series_id", "origin_quarter"], how="outer", indicator=True, validate="many_to_one")
    if not out["_merge"].eq("both").all():
        bad = out.loc[~out["_merge"].eq("both"), "_merge"].value_counts().to_dict()
        raise SystemExit(f"{name}: forecasts and saved attributions do not match one to one ({bad})")
    return out.drop(columns="_merge")


def rmse(p: pd.DataFrame, by: str) -> pd.Series:
    p = p.assign(se=(p["y_pred"] - p["y_true"]) ** 2)
    return (p.groupby([by, "seed"])["se"].mean() ** 0.5).groupby(level=0).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/explain/validation"))
    ap.add_argument("--preds-dir", type=Path, default=Path("results/long_hpi_122/validation"))
    ap.add_argument("--market-list", default="docs/market_list.csv")
    args = ap.parse_args()

    z = np.load(args.dir / "row_attributions.npz", allow_pickle=True)   # string columns may be object arrays
    preds = load_checked(z, args.preds_dir)
    meta = pd.DataFrame({"series_id": z["series_id"], "origin_quarter": z["origin_quarter"], "split_id": z["split_id"]})
    if meta.duplicated(["series_id", "origin_quarter"]).any():
        raise SystemExit("duplicate forecasts in row_attributions.npz")
    horizons = list(z["horizons"])
    fold = {s: i for i, s in enumerate(z["ar_split_id"])}
    ar_rows = {k: z[f"{k}_weights"][[fold[s] for s in meta["split_id"]]] for k in ("ar4", "ar8")}
    sit = situations(meta, z["x_window"], pd.read_csv(args.market_list))

    rows, by_market, within = [], [], []
    for band, hs in BANDS.items():
        ix = [horizons.index(h) for h in hs]
        w = z["tr_grad"][:, ix, :].mean(axis=1)
        frame = sit.assign(tr_recent=w[:, :2].sum(axis=1), tr_older=w[:, 4:].sum(axis=1),
                           **{f"{k}_recent": ar_rows[k][:, ix, :2].mean(axis=1).sum(axis=1) for k in ("ar8", "ar4")},
                           ar8_older=ar_rows["ar8"][:, ix, 4:].mean(axis=1).sum(axis=1), market=sit["series_id"])
        groupings = ["past_year_growth", "turning_point", "period", "history", "market"]
        joined = {k: matched(preds[k], frame[["series_id", "origin_quarter", *groupings]], hs, MODELS[k]) for k in MODELS}
        for by in groupings:
            g = frame.groupby(by).agg(
                n_forecasts=("series_id", "size"), n_markets=("series_id", "nunique"),
                tr_recent=("tr_recent", "mean"), tr_recent_p10=("tr_recent", lambda s: s.quantile(0.1)),
                tr_recent_p90=("tr_recent", lambda s: s.quantile(0.9)), ar8_recent=("ar8_recent", "mean"),
                ar4_recent=("ar4_recent", "mean"), tr_older=("tr_older", "mean"), ar8_older=("ar8_older", "mean"))
            tr = rmse(joined["tr"], by)
            g["rmse_vs_ar8"] = tr / rmse(joined["ar8"], by)
            g["rmse_vs_ar4"] = tr / rmse(joined["ar4"], by)
            g = g.reset_index().rename(columns={by: "group"}).assign(grouping=by, horizon=band)
            (by_market if by == "market" else rows).append(g)
        f = frame.groupby("split_id")
        within.append(pd.DataFrame({
            "tr_recent_mean": f["tr_recent"].mean(), "tr_recent_sd": f["tr_recent"].std(),
            "tr_recent_p10": f["tr_recent"].quantile(0.1), "tr_recent_p90": f["tr_recent"].quantile(0.9),
            "ar8_recent": f["ar8_recent"].first(), "ar4_recent": f["ar4_recent"].first(),
        }).reset_index().assign(horizon=band))

    cols = ["grouping", "group", "horizon", "n_forecasts", "n_markets", "tr_recent", "tr_recent_p10", "tr_recent_p90",
            "ar8_recent", "ar4_recent", "tr_older", "ar8_older", "rmse_vs_ar8", "rmse_vs_ar4"]
    table = pd.concat(rows, ignore_index=True)[cols]
    table.to_csv(args.dir / "weights_by_regime.csv", index=False)
    markets = pd.concat(by_market, ignore_index=True)
    markets[markets["n_forecasts"] >= MIN_ORIGINS].rename(columns={"group": "series_id"}).drop(
        columns=["grouping", "n_markets"]).to_csv(args.dir / "weights_by_market.csv", index=False)
    within = pd.concat(within, ignore_index=True)
    within.to_csv(args.dir / "weights_within_fold.csv", index=False)

    plot(sit, z, ar_rows, horizons, table, args.dir / "weights_by_regime.png")
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print("\nWithin-fold spread of the Transformer's weight on the latest two quarters (AR: none):")
    print(within.groupby("horizon")[["tr_recent_sd"]].describe().round(3).to_string())


def plot(sit, z, ar_rows, horizons, table, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    bins = np.quantile(sit["past_year"], np.linspace(0, 1, 11))
    which = np.clip(np.digitize(sit["past_year"], bins[1:-1]), 0, 9)
    centers = np.array([np.median(sit["past_year"][which == b]) for b in range(10)]) * 100
    for c, (band, hs) in enumerate(BANDS.items()):
        ix = [horizons.index(h) for h in hs]
        w = z["tr_grad"][:, ix, :2].mean(axis=1).sum(axis=1)
        ax = axes[0, c]
        ax.plot(centers, [w[which == b].mean() for b in range(10)], "o-", color="#1f4e79",
                label="Shared Transformer (mean)")
        ax.fill_between(centers, [np.percentile(w[which == b], 10) for b in range(10)],
                        [np.percentile(w[which == b], 90) for b in range(10)], color="#1f4e79", alpha=0.15,
                        label="Transformer, 10th–90th percentile")
        for k, style, label in (("ar8", ("^-", "#b45f06"), "Pooled AR(8)"), ("ar4", ("v:", "#e69138"), "Pooled AR(4)")):
            a = ar_rows[k][:, ix, :2].mean(axis=1).sum(axis=1)
            ax.plot(centers, [a[which == b].mean() for b in range(10)], style[0], color=style[1], label=label)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{band[1:].replace('-', '–')} quarters ahead")
        ax.set_xlabel("Past-year real HPI growth at forecast date (%, decile medians)")
        ax.grid(alpha=0.3)
        if c == 0:
            ax.set_ylabel("Weight on the latest two quarters", fontsize=9)
            ax.legend(fontsize=8, frameon=False)

        parts = []
        for grouping, order in (("past_year_growth", GROWTH_LABELS),
                                ("turning_point", ["steady rise", "steady fall", "turning up", "turning down"])):
            sub = table[(table["horizon"] == band) & (table["grouping"] == grouping)].set_index("group")
            parts.append(sub.reindex([g for g in order if g in sub.index]))
        sub = pd.concat(parts)
        y = np.arange(len(sub))
        ax = axes[1, c]
        ax.barh(y - 0.2, sub["rmse_vs_ar8"], height=0.4, color="#b45f06", label="vs pooled AR(8)")
        ax.barh(y + 0.2, sub["rmse_vs_ar4"], height=0.4, color="#e69138", alpha=0.6, label="vs pooled AR(4)")
        ax.set_yticks(y, sub.index, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(1, color="black", lw=0.8)
        ax.set_xlabel("RMSE ratio, Transformer / AR  (<1: Transformer better)")
        ax.grid(alpha=0.3, axis="x")
        if c == 0:
            ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("When does the shared Transformer weight recent growth differently from pooled AR? "
                 "(validation 2000–2009)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
