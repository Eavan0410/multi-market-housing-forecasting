"""How much do the confidence intervals depend on the bootstrap block length?

Re-runs the paired ratio bootstrap from the saved prediction files with several block
lengths (in forecast origins). Point estimates do not change; only the intervals do.
No retraining. Covers the main shared-model comparisons, the training-pool runs (#9, all three market orders)
and the leave-one-market-out runs (#10), each held-out model against the shared model,
AR and no-change.

Example:
    python scripts/block_length_check.py
Writes results/block_length_sensitivity.csv (one row per comparison, block length and horizon).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.config import load_config
from housing_forecast.evaluation.markets import paired_ratio_bootstrap

MAIN = "results/long_hpi/validation"
SCALE = "results/market_scale/validation"
SCALE_HISTORY = "results/market_scale/history/validation"
SCALE_RANDOM = "results/market_scale/random/validation"
HOLDOUT = "results/market_holdout"
HOLDOUT_MARKETS = ("australia", "chile", "taiwan_national", "portugal")


def comparisons(main_dir=None):
    """(experiment, folder, model A, model B)."""
    shared = "transformer__national"
    for other in ("transformer_per_market__national", "ar__national", "arima__national", "ar_pooled__national",
                  "no_change__national"):
        yield "main", main_dir or MAIN, shared, other
    if main_dir is not None:
        return
    for order, folder in (("region", SCALE), ("history", SCALE_HISTORY), ("random", SCALE_RANDOM)):
        sizes = (10, 20, 41) if order == "region" else (10, 20)   # 5 and 41 are shared by all orders
        for size in sizes:
            yield f"market_scale_{order}", folder, f"transformer__scale_{size}", "transformer__scale_5"
    for market in HOLDOUT_MARKETS:
        for other in (shared, "ar__national", "no_change__national"):
            yield f"holdout_{market}", f"{HOLDOUT}/{market}/validation", "transformer__holdout", other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 12, 20])
    ap.add_argument("--main-dir", type=Path, help="check only the main comparisons in this prediction directory")
    ap.add_argument("--out", default="results/block_length_sensitivity.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = []
    for experiment, folder, a, b in comparisons(args.main_dir):
        preds = pd.concat([pd.read_csv(Path(folder) / f"preds_{m}.csv") for m in (a, b)], ignore_index=True)
        for block in args.blocks:
            try:
                r = paired_ratio_bootstrap(preds, a, b, cfg, block_length=block)
            except ValueError as e:          # block longer than the forecast calendar
                if args.main_dir is not None:
                    raise
                print(f"skip {experiment} {a} vs {b}, block {block}: {e}")
                continue
            rows.append(r.assign(experiment=experiment))
            print(f"{experiment:24s} {a} vs {b}  block {block:2d}  done", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out["excludes_1"] = (out["ci_high"] < 1) | (out["ci_low"] > 1)
    out.to_csv(args.out, index=False)

    bands = out[out["horizon"].isin(["h3-6", "h16-20"])].copy()
    bands["cell"] = bands.apply(lambda r: f"{r.median_ratio:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
    pd.set_option("display.width", 220)
    print(bands.pivot_table(index=["experiment", "model_a", "model_b", "horizon"], columns="block_length",
                            values="cell", aggfunc="first").to_string())
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
