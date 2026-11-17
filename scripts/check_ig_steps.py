"""Are the Integrated Gradients results sensitive to the number of integration steps?

Retrains the shared Transformer for the folds and seeds with the largest completeness error in
results/explain/validation/checks.csv, recomputes Integrated Gradients with several step counts
and reports, for each:
  completeness_error   max |sum of contributions - (forecast - baseline forecast)|
  max_share_change     largest change in a lag's |contribution| share (3-6 and 16-20 ranges)
                       compared with the most steps
  rank_corr            Spearman correlation of the eight lags' shares with the most steps
  same_top_lag         whether the most important lag is unchanged
Local-gradient weights do not depend on the step count.

    OMP_NUM_THREADS=1 python scripts/check_ig_steps.py            # worst 3 fold-seed pairs
Writes results/explain/validation/ig_steps_check.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explain_lag_weights import abs_share  # noqa: E402
from run_backtest import load_samples  # noqa: E402

from housing_forecast.backtesting.runner import pool_national, training_samples  # noqa: E402
from housing_forecast.backtesting.scaling import WindowStandardizer  # noqa: E402
from housing_forecast.backtesting.splits import end_series, eval_mask, last_label_origin, make_splits  # noqa: E402
from housing_forecast.config import load_config  # noqa: E402
from housing_forecast.evaluation.explain import transformer_attributions  # noqa: E402
from housing_forecast.models.base import set_global_seed  # noqa: E402

BANDS = {"h3-6": range(3, 7), "h16-20": range(16, 21)}


def main() -> None:
    from housing_forecast.models.transformer import TransformerForecaster

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    ap.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    ap.add_argument("--dir", type=Path, default=Path("results/explain/validation"))
    ap.add_argument("--worst", type=int, default=3, help="number of fold-seed pairs to check")
    ap.add_argument("--steps", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    args = ap.parse_args()

    checks = pd.read_csv(args.dir / "checks.csv").nlargest(args.worst, "ig_completeness_error")
    cfg = load_config(args.config)
    samples = load_samples(cfg, args.panel, args.cpi)
    splits = {s.split_id: s for s in make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))}
    j = samples.feature_names.index("hpi_qoq")
    hidx = {h: i for i, h in enumerate(samples.horizons)}
    rows = []
    for r in checks.itertuples():
        split = splits[r.split_id]
        train = training_samples(samples, split, pool_national())
        test = samples.subset(eval_mask(samples, split, cfg["target"]["eval_series"]))
        scaler = WindowStandardizer().fit(train.X)
        set_global_seed(int(r.seed))
        model = TransformerForecaster()
        model.fit(train.replace(X=scaler.transform(train.X)), int(r.seed))
        shares, errors = {}, {}
        for steps in sorted(args.steps):
            att = transformer_attributions(model, scaler, test.X, feature=j, steps=steps)
            errors[steps] = att["completeness_error"]
            shares[steps] = {b: np.nanmean(abs_share(att["ig"])[:, [hidx[h] for h in hs], :], axis=(0, 1))
                             for b, hs in BANDS.items()}
            print(f"{r.split_id} seed {r.seed} steps {steps}: completeness error {errors[steps]:.2e}", flush=True)
        ref = max(args.steps)
        for steps in sorted(args.steps):
            for band in BANDS:
                a, b = shares[steps][band], shares[ref][band]
                rows.append({"split_id": r.split_id, "seed": r.seed, "steps": steps, "horizon": band,
                             "completeness_error": errors[steps], "max_share_change": float(np.abs(a - b).max()),
                             "rank_corr": float(pd.Series(a).rank().corr(pd.Series(b).rank())),
                             "same_top_lag": bool(np.argmax(a) == np.argmax(b))})
    out = pd.DataFrame(rows)
    out.to_csv(args.dir / "ig_steps_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(f"-> {args.dir / 'ig_steps_check.csv'}")


if __name__ == "__main__":
    main()
