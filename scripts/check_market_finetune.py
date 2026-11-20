"""Compare fine-tuning with its shared starting model on all and eligible rows.

With --select, eligibility requires an epoch choice in every seed, including zero."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.backtesting.manifest import check_comparable, read_manifest
from housing_forecast.config import load_config
from housing_forecast.data.contract import check_alignment
from housing_forecast.evaluation.markets import paired_ratio_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=None,
                        help="default: results/market_finetune_122/<phase> (inner_select/<phase> with --select)")
    parser.add_argument("--phase", default="validation", choices=["validation", "test"])
    parser.add_argument("--config", default="configs/experiment_long.toml")
    parser.add_argument("--select", action="store_true", help="check the inner-selection run")
    args = parser.parse_args()
    if args.dir is None:
        base = Path("results/market_finetune_122")
        args.dir = (base / "inner_select" / args.phase) if args.select else (base / args.phase)
    cfg = load_config(args.config)
    a = "transformer_shared_ftsel__national" if args.select else "transformer_shared_ft__national"
    b = "transformer_shared__national"
    paths = [args.dir / f"preds_{name}.csv" for name in (a, b)]
    manifests = {str(path): read_manifest(path) for path in paths}
    if any(m is None for m in manifests.values()):
        raise ValueError("both prediction files need run manifests")
    check_comparable(manifests)
    expected = {m["model"]: m["seeds"] for m in manifests.values()}
    preds = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    check_alignment(preds, expected_seeds=expected)
    diag = pd.read_csv(args.dir / "fit_diagnostics.csv")
    key = ["series_id", "split_id", "seed"]
    scored = preds.loc[preds.model.eq(a), key].drop_duplicates()
    scored = scored.merge(diag[diag.stage.eq("adaptation")], on=key, how="left", validate="one_to_one")
    if scored.fit_status.isna().any():
        raise ValueError("a scored market, fold and seed has no diagnostics")
    if args.select:
        scored["adapted"] = scored.select_reason.eq("selected") & scored.fallback_horizons.fillna("").eq("")
    else:
        scored["adapted"] = scored.fit_status.eq("fine_tuned") & scored.n_fallback_forecasts.eq(0)
    blocks = scored.groupby(key[:2]).agg(adapted=("adapted", "all"), seeds=("seed", "nunique"))
    if not blocks.seeds.eq(len(expected[a])).all():
        raise ValueError("incomplete seed diagnostics")
    keep = blocks[blocks.adapted].reset_index()[key[:2]]
    adapted = preds.merge(keep, on=key[:2], how="inner", validate="many_to_one")
    check_alignment(adapted, expected_seeds=expected)
    coverage = adapted.groupby(["series_id", "horizon"]).origin_quarter.nunique().unstack(fill_value=0)
    eligible = coverage.index[coverage.ge(8).all(axis=1)]
    sets = {"all": preds, "adapted_only": adapted[adapted.series_id.isin(eligible)]}
    results = []
    for label, frame in sets.items():
        for block in (4, 8, 12, 20):
            ci = paired_ratio_bootstrap(frame, a, b, cfg, min_origins=8, block_length=block)
            results.append(ci.assign(rows=label))
    results = pd.concat(results, ignore_index=True)
    results.to_csv(args.dir / "adaptation_check.csv", index=False)
    counts = scored.groupby("fit_status").agg(
        market_fold_seeds=("seed", "size"), forecast_windows=("n_forecasts", "sum"),
        fallback_windows=("n_fallback_forecasts", "sum"),
    )
    counts.to_csv(args.dir / "adaptation_counts.csv")
    if args.select:
        chosen = scored[scored.select_reason.eq("selected")].groupby("series_id").agg(
            market_fold_seeds=("seed", "size"), share_zero=("selected_epochs", lambda x: float((x == 0).mean())),
            median_epochs=("selected_epochs", "median"), mean_epochs=("selected_epochs", "mean"),
        ).sort_values("share_zero")
        chosen.to_csv(args.dir / "selected_epochs.csv")
        print(chosen.round(2).to_string())
    print(counts.to_string())
    print(results[results.horizon.isin(["h3-6", "h16-20"])][
        ["rows", "block_length", "horizon", "n_markets", "median_ratio", "ci_low", "ci_high"]
    ].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
