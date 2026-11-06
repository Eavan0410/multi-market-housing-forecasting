"""Compare shared and per-market Transformers with and without no-change fallbacks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from housing_forecast.backtesting.manifest import check_comparable, read_manifest
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment
from housing_forecast.evaluation.markets import history_groups, paired_ratio_bootstrap

KEY = ["series_id", "split_id"]


def fitted_blocks(diag: pd.DataFrame) -> pd.DataFrame:
    """(market, fold) blocks where every seed fitted and nothing fell back."""
    ok = (diag["fit_status"] == "fitted") & (diag["n_fallback_forecasts"] == 0)
    if diag["fallback_horizons"].notna().any():
        raise ValueError("partial (per-horizon) fallback found; this script only drops whole blocks")
    per_block = diag.assign(ok=ok).groupby(KEY)["ok"].all()
    return per_block[per_block].index.to_frame(index=False)


def compare(preds: pd.DataFrame, a: str, b: str, cfg: dict, min_origins: int, label: str) -> pd.DataFrame:
    return paired_ratio_bootstrap(preds, a, b, cfg, min_origins=min_origins).assign(rows=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/long_hpi/validation")
    ap.add_argument("--pool", default="national")
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    ap.add_argument("--min-origins", type=int, default=8)
    args = ap.parse_args()

    d = Path(args.dir)
    a, b = f"transformer__{args.pool}", f"transformer_per_market__{args.pool}"
    cfg = load_config(args.config)
    paths = [d / f"preds_{m}.csv" for m in (a, b)]
    manifests = {str(path): read_manifest(path) for path in paths}
    if any(m is None for m in manifests.values()):
        raise ValueError("both prediction files need run manifests")
    check_comparable(manifests)
    preds = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    check_alignment(preds, expected_seeds={m["model"]: m["seeds"] for m in manifests.values()})
    diag = pd.read_csv(d / f"diag_{b}.csv")
    groups = history_groups(pd.read_csv(args.panel))

    diag_key = [*KEY, "seed"]
    scored = preds.loc[preds["model"].eq(b), diag_key].drop_duplicates()
    matched = scored.merge(diag, on=diag_key, how="left", validate="one_to_one")
    if matched["fit_status"].isna().any():
        raise ValueError("a scored market, fold and seed has no diagnostics")
    keep_blocks = fitted_blocks(matched)
    tagged = preds.merge(keep_blocks.assign(fitted=True), on=KEY, how="left", validate="many_to_one")
    tagged["fitted"] = tagged["fitted"].eq(True)
    fallback = tagged.loc[tagged["model"].eq(b) & ~tagged["fitted"], "y_pred"]
    if not np.allclose(fallback, 0.0):
        raise ValueError("nonzero prediction in a block marked as no-change")

    fitted = tagged[tagged["fitted"]].drop(columns="fitted")
    coverage = fitted.groupby(["series_id", "horizon"])["origin_quarter"].nunique().unstack(fill_value=0)
    coverage = coverage.reindex(columns=sorted(preds["horizon"].unique()), fill_value=0)
    eligible = coverage.index[coverage.ge(args.min_origins).all(axis=1)]
    sets = {
        "full": preds,
        "same_markets": preds[preds["series_id"].isin(eligible)],
        "fitted_only": fitted[fitted["series_id"].isin(eligible)],
    }
    ci = pd.concat([compare(p, a, b, cfg, args.min_origins, k) for k, p in sets.items()], ignore_index=True)
    ci.to_csv(d / "fitted_only_ratio_ci.csv", index=False)

    hist_rows = []
    for name, p in sets.items():
        p = p.merge(groups[["series_id", "history"]], on="series_id", how="left")
        for g, pg in p.groupby("history"):
            try:
                r = paired_ratio_bootstrap(pg.drop(columns="history"), a, b, cfg, min_origins=args.min_origins)
            except ValueError as e:      # e.g. too few forecast dates left for one bootstrap block
                print(f"skip {name} / {g}: {e}")
                continue
            r = r[r["horizon"].isin(["1", "4", "h3-6", "h16-20"])]
            hist_rows.append(r.assign(rows=name, history=g))
    by_hist = pd.concat(hist_rows, ignore_index=True)
    by_hist.to_csv(d / "fitted_only_ratio_by_history.csv", index=False)

    # Count forecasts once rather than once per seed.
    s0 = tagged[(tagged["model"] == b) & (tagged["seed"] == tagged["seed"].min())]
    s0 = s0.merge(groups, on="series_id", how="left")
    per_market = s0.groupby(["history", "hpi_start", "series_id"]).agg(
        forecasts=("fitted", "size"), kept=("fitted", "sum"),
        folds=("split_id", "nunique"),
        folds_fitted=("split_id", lambda s: s[s0.loc[s.index, "fitted"]].nunique()),
        first_fitted_origin=("origin_quarter", lambda s: s[s0.loc[s.index, "fitted"]].min()),
    ).reset_index()
    per_market["eligible_for_comparison"] = per_market["series_id"].isin(eligible)
    per_market["kept_share"] = per_market["kept"] / per_market["forecasts"]
    per_market.sort_values(["history", "kept_share"]).to_csv(d / "fitted_only_markets.csv", index=False)

    def count(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "markets": len(g), "markets_with_fallback": int((g["kept"] < g["forecasts"]).sum()),
            "markets_kept": int((g["kept"] > 0).sum()), "markets_dropped": int((g["kept"] == 0).sum()),
            "markets_in_comparison": int(g["eligible_for_comparison"].sum()),
            "forecasts": int(g["forecasts"].sum()), "forecasts_kept": int(g["kept"].sum()),
            "share_kept": g["kept"].sum() / g["forecasts"].sum(),
        })
    counts = per_market.groupby("history").apply(count).reset_index()
    total = count(per_market).to_frame().T.assign(history="all")
    counts = pd.concat([counts, total], ignore_index=True)
    counts.to_csv(d / "fitted_only_counts.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"{a} / {b}: median market RMSE ratio [95% CI], share of markets where shared is better\n")
    wide = ci.assign(cell=ci.apply(lambda r: f"{r.median_ratio:.3f} [{r.ci_low:.3f},{r.ci_high:.3f}] "
                                             f"{r.share_markets_a_better:.0%} n={r.n_markets}", axis=1))
    order = list(dict.fromkeys(ci["horizon"]))
    print(wide.pivot(index="horizon", columns="rows", values="cell").loc[order, list(sets)].to_string())
    print("\nCounts (forecasts = scored rows per seed of the per-market model):")
    print(counts.to_string(index=False))
    print("\nBy history group:")
    bh = by_hist.assign(cell=by_hist.apply(lambda r: f"{r.median_ratio:.3f} [{r.ci_low:.3f},{r.ci_high:.3f}] "
                                                     f"n={r.n_markets}", axis=1))
    print(bh.pivot_table(index=["history", "horizon"], columns="rows", values="cell", aggfunc="first")
          [list(sets)].to_string())


if __name__ == "__main__":
    main()
