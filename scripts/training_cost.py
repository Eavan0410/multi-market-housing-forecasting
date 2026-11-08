"""How many models each approach trains, and how long one fold takes.

Counts come from the saved validation runs (no retraining). A "final model" is one set
of parameters used to forecast a fold: one per seed for the Transformers, one per fold for
the deterministic AR models (the 20 horizon regressions of one AR count as one model).
"Training runs" also include the extra fit that picks the epoch count on a time split
inside the fold, and the inner-stage fits of the second fine-tuning setting.

    python scripts/training_cost.py                      # counts only (no torch needed)
    OMP_NUM_THREADS=1 python scripts/training_cost.py --time   # also time one fold per approach

Writes results/training_cost.csv and, with --time, results/training_time.csv.
Timings are wall-clock seconds on the machine that runs the script, for the last
validation fold (the largest training set) and one seed.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import pandas as pd

MAIN = Path("results/long_hpi/validation")
FIXED = Path("results/market_finetune/validation")
SELECT = Path("results/market_finetune/inner_select/validation")
FOLD_SAMPLES = Path("docs/market_fold_samples.csv")
AR_MIN_ROWS = 12          # ARDirect.min_rows: fewer labelled rows than this use the pooled fit


def count_models(seeds: int = 5, folds: int = 10) -> pd.DataFrame:
    rows = []
    shared = folds * seeds
    rows.append({"approach": "Shared Transformer", "final_models": shared, "training_runs": 2 * shared,
                 "note": "one network per fold and seed; each fit also runs one epoch-selection fit"})

    per = pd.read_csv(MAIN / "diag_transformer_per_market__national.csv")
    fitted = per[per["fit_status"].eq("fitted")]
    runs = len(fitted) + int(fitted["epochs_from"].eq("early_stopping").sum())
    rows.append({"approach": "Per-market Transformer", "final_models": len(fitted), "training_runs": runs,
                 "note": f"{fitted['series_id'].nunique()} markets fitted at least once; "
                         f"{len(per) - len(fitted)} market-fold-seeds fell back to no-change"})

    samples = pd.read_csv(FOLD_SAMPLES)
    samples = samples[samples["split_id"].str.startswith("val_")]
    ar_markets = int((samples["n_train_samples"] >= AR_MIN_ROWS).sum())
    rows.append({"approach": "Per-market AR", "final_models": ar_markets + folds, "training_runs": ar_markets + folds,
                 "note": "one least-squares fit per market and fold with enough rows, plus one pooled fallback per fold"})
    rows.append({"approach": "Pooled AR", "final_models": folds, "training_runs": folds,
                 "note": "one least-squares fit per fold"})

    if (FIXED / "fit_diagnostics.csv").exists():
        d = pd.read_csv(FIXED / "fit_diagnostics.csv")
        tuned = int(d["fit_status"].eq("fine_tuned").sum())
        rows.append({"approach": "Shared + fine-tuning (fixed 10 epochs)", "final_models": shared + tuned,
                     "training_runs": 2 * shared + tuned,
                     "note": "shared networks plus one fine-tuned copy per adapted market, fold and seed"})
    if (SELECT / "fit_diagnostics.csv").exists():
        d = pd.read_csv(SELECT / "fit_diagnostics.csv")
        a = d[d["stage"].eq("adaptation")]
        tuned = int(a["fit_status"].eq("fine_tuned").sum())
        curves = int(a["select_reason"].eq("selected").sum())
        rows.append({"approach": "Shared + fine-tuning (epochs chosen inside training)",
                     "final_models": shared + tuned, "training_runs": 4 * shared + curves + tuned,
                     "note": "inner and full shared fits (each with epoch selection), one up-to-30-epoch "
                             "selection run per eligible market, then the chosen fine-tuning"})
    return pd.DataFrame(rows)


def time_one_fold(config: str, panel: str, cpi: str) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from run_backtest import load_samples
    from housing_forecast.backtesting.runner import pool_national, run_backtest
    from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits
    from housing_forecast.config import load_config
    from housing_forecast.models.registry import REGISTRY
    from housing_forecast.models.transformer import SharedTransformerFineTune, SharedTransformerFineTuneSelect

    cfg = load_config(config)
    cfg["run"]["seeds"] = [0]
    samples = load_samples(cfg, panel, cpi)
    split = make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))[-1:]
    approaches = {
        "Shared Transformer": REGISTRY["transformer"],
        "Per-market Transformer": REGISTRY["transformer_per_market"],
        "Per-market AR": REGISTRY["ar"],
        "Pooled AR": REGISTRY["ar_pooled"],
        "Shared + fine-tuning (fixed 10 epochs)": lambda: SharedTransformerFineTune(ft_epochs=10),
        "Shared + fine-tuning (epochs chosen inside training)": lambda: SharedTransformerFineTuneSelect(),
    }
    rows = []
    for name, factory in approaches.items():
        started = time.perf_counter()
        preds = run_backtest(samples, split, factory, cfg, pool_national(), model_name=name)
        seconds = time.perf_counter() - started
        print(f"{name:55s} {seconds:8.1f} s", flush=True)
        rows.append({"approach": name, "split_id": split[0].split_id, "seed": 0,
                     "markets_forecast": preds["series_id"].nunique(), "wall_seconds": round(seconds, 1)})
    env = {"machine": platform.machine(), "processor": platform.processor() or platform.machine(),
           "system": platform.system(), "python": platform.python_version(), "torch": torch.__version__,
           "torch_threads": torch.get_num_threads(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    return pd.DataFrame(rows).assign(**env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time", action="store_true", help="also time the last validation fold for each approach")
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    ap.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    args = ap.parse_args()

    counts = count_models()
    counts.to_csv("results/training_cost.csv", index=False)
    pd.set_option("display.width", 200)
    print(counts[["approach", "final_models", "training_runs"]].to_string(index=False))
    if args.time:
        timing = time_one_fold(args.config, args.panel, args.cpi)
        timing.to_csv("results/training_time.csv", index=False)
        print(timing[["approach", "wall_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
