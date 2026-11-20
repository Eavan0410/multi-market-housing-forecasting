"""Summarize national HPI coverage, forecast dates and training samples by market."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from housing_forecast.backtesting.infoset import build_samples, features_from_config
from housing_forecast.backtesting.runner import pool_national
from housing_forecast.backtesting.splits import (
    end_series, last_label_origin, make_splits, mask_unpublished_labels, train_mask,
)
from housing_forecast.config import load_config
from housing_forecast.data.transforms import prepare_panel

REGION = {
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Canada": "North America", "United States": "North America", "Mexico": "Latin America",
    "Brazil": "Latin America", "Chile": "Latin America", "Colombia": "Latin America",
    "China": "Asia", "India": "Asia", "Indonesia": "Asia", "Japan": "Asia", "South Korea": "Asia",
    "Taiwan_national": "Asia", "Israel": "Middle East", "Saudi Arabia": "Middle East", "Türkiye": "Middle East",
    "South Africa": "Africa", "Russia": "Eastern Europe",
    "Bulgaria": "Eastern Europe", "Croatia": "Eastern Europe", "Czechia": "Eastern Europe", "Estonia": "Eastern Europe",
    "Hungary": "Eastern Europe", "Latvia": "Eastern Europe", "Lithuania": "Eastern Europe", "Poland": "Eastern Europe",
    "Romania": "Eastern Europe", "Slovak Republic": "Eastern Europe", "Slovenia": "Eastern Europe",
    "Denmark": "Northern Europe", "Finland": "Northern Europe", "Iceland": "Northern Europe",
    "Norway": "Northern Europe", "Sweden": "Northern Europe",
    "Austria": "Western Europe", "Belgium": "Western Europe", "France": "Western Europe", "Germany": "Western Europe",
    "Ireland": "Western Europe", "Luxembourg": "Western Europe", "Netherlands": "Western Europe",
    "Switzerland": "Western Europe", "United Kingdom": "Western Europe",
    "Greece": "Southern Europe", "Italy": "Southern Europe", "Portugal": "Southern Europe", "Spain": "Southern Europe",
}
B_FILES = {"Australia", "Japan", "South Korea", "United States"}       # HPI from data/raw/four_countries


def history_group(start: pd.Period) -> str:
    if start <= pd.Period("1979Q4", "Q"):
        return "long (by 1979)"
    if start <= pd.Period("1999Q4", "Q"):
        return "medium (1980-1999)"
    return "short (2000 or later)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    ap.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    ap.add_argument("--folds-out", default="docs/market_fold_samples.csv")
    ap.add_argument("--validation", default="results/long_hpi_122/validation/metrics_by_market.csv")
    ap.add_argument("--out", default="docs/market_list.csv")
    ap.add_argument("--jump", type=float, default=0.10, help="flag |quarterly real log change| above this")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prepared = prepare_panel(pd.read_csv(args.panel), pd.read_csv(args.cpi), cfg)
    nat = prepared[prepared["geography_level"] == "national"].dropna(subset=["hpi_real"]).copy()
    nat["q"] = pd.PeriodIndex(nat["quarter"], freq="Q")

    samples = build_samples(prepared, cfg, features_from_config(cfg))
    splits = make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))
    fold_rows = []
    for sp in splits:
        tr = mask_unpublished_labels(samples.subset(train_mask(samples, sp) & pool_national()(samples.meta)), sp)
        counts = tr.meta["series_id"].value_counts()
        for sid in sorted(nat["series_id"].unique()):
            fold_rows.append({"split_id": sp.split_id, "train_origin_end": str(sp.train_origin_end),
                              "series_id": sid, "n_train_samples": int(counts.get(sid, 0))})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(args.folds_out, index=False)
    first, last = splits[0].split_id, splits[-1].split_id
    per = folds.pivot(index="series_id", columns="split_id", values="n_train_samples")

    origins = {}
    if Path(args.validation).exists():
        v = pd.read_csv(args.validation)
        v = v[v["horizon"] == 1].groupby("series_id")["n_origins"].max()
        origins = v.to_dict()

    rows = []
    for sid, g in nat.groupby("series_id"):
        g = g.sort_values("q")
        start, end = g["q"].min(), g["q"].max()
        span = (end - start).n + 1
        observed = g["q"].nunique()
        full = g.set_index("q")["hpi_real"].reindex(pd.period_range(start, end, freq="Q"))
        qoq = np.log(full).diff()
        big = qoq[qoq.abs() > args.jump]
        rows.append({
            "series_id": sid,
            "region": REGION.get(sid, "?"),
            "hpi_source": "Taiwan raw (nominal HPI, deflated with Taiwan CPI in the pipeline)" if sid.startswith("Taiwan")
                          else ("four_countries" if sid in B_FILES else "OECD AnaHPI (real)"),
            "hpi_start": str(start), "hpi_end": str(end),
            "n_quarters": observed, "missing_quarters_inside": span - observed,
            "history": history_group(start),
            "n_big_jumps": len(big),
            "big_jump_quarters": " ".join(f"{q}({v:+.2f})" for q, v in big.items())[:200],
            "validation_origins_2000_2009": int(origins.get(sid, 0)),
            "in_long_hpi_summary": int(origins.get(sid, 0)) >= 8,
            "folds_with_training_data": int((per.loc[sid] > 0).sum()) if sid in per.index else 0,
            "n_folds": len(splits),
            "train_samples_first_fold": int(per.loc[sid, first]) if sid in per.index else 0,
            "train_samples_last_fold": int(per.loc[sid, last]) if sid in per.index else 0,
        })
    out = pd.DataFrame(rows).sort_values(["history", "hpi_start", "series_id"])
    out.to_csv(args.out, index=False)
    print(out.groupby("history").size().to_string())
    print(f"{len(out)} national markets -> {args.out}; per-fold training samples -> {args.folds_out}")
    eff = folds[folds["n_train_samples"] > 0].groupby("split_id")["series_id"].nunique()
    print("markets that actually contribute training samples, by fold:")
    print(eff.to_string())


if __name__ == "__main__":
    main()
