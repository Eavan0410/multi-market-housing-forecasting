#!/usr/bin/env bash
# Rerun the secondary validation experiments at the 122-day forecast date (Josh, 11/16:
# keep Chile, use 122 days). The main comparison is already in results/long_hpi_122.
# Run from the repository root in the hpf environment:   bash scripts/rerun_secondary_122.sh
# Takes about an hour on a laptop. Training steps whose prediction files already exist are
# skipped, so the script can be restarted after an interruption (delete a folder to redo it).
set -euo pipefail
export OMP_NUM_THREADS=1          # Transformer runs are only repeatable single-threaded
C=configs/experiment_long.toml

echo "== 1/5 Single-thread reference for the holdouts (shared Transformer, 122 days)"
# Holdout gaps are a few percent, the size of multi-thread run-to-run noise, so the held-out
# model is compared with a reference trained the same way.
REF=results/long_hpi_122_st/validation
if [ ! -f "$REF/preds_transformer__national.csv" ]; then
  python scripts/run_backtest.py --config $C --models transformer --pool national --experiment long_hpi_122_st
fi
for m in ar__national no_change__national; do
  cp -n results/long_hpi_122/validation/preds_$m.csv results/long_hpi_122/validation/preds_$m.meta.json "$REF/" || true
done
python - <<'PY'
import pandas as pd
k = ["series_id", "origin_quarter", "horizon", "seed"]
a = pd.read_csv("results/long_hpi_122/validation/preds_transformer__national.csv").set_index(k)["y_pred"]
b = pd.read_csv("results/long_hpi_122_st/validation/preds_transformer__national.csv").set_index(k)["y_pred"]
d = (a - b.reindex(a.index)).abs()
print(f"main 122-day Transformer vs single-thread rerun: max |prediction difference| {d.max():.2e}")
PY

echo "== 2/5 Training-pool size, three market orders"
python scripts/run_market_scale.py --config $C
python scripts/run_market_scale.py --config $C --order history
python scripts/run_market_scale.py --config $C --order random
for d in results/market_scale_122/validation results/market_scale_122/history/validation results/market_scale_122/random/validation; do
  python scripts/evaluate_markets.py $d/preds_*.csv --config $C --reference transformer__scale_5 \
      --compare transformer__scale_10 transformer__scale_5 \
      --compare transformer__scale_20 transformer__scale_5 \
      --compare transformer__scale_41 transformer__scale_5
done

echo "== 3/5 Leave-one-market-out"
for m in Australia Chile Taiwan_national Portugal; do
  d=results/market_holdout_122/$(echo $m | tr '[:upper:]' '[:lower:]')/validation
  [ -f "$d/preds_transformer__holdout.csv" ] || python scripts/run_market_holdout.py --config $C --market $m --reference-dir $REF
  python scripts/evaluate_markets.py $d/preds_*.csv --config $C --reference no_change__national \
      --compare transformer__holdout no_change__national \
      --compare transformer__holdout ar__national \
      --compare transformer__holdout transformer__national
done

echo "== 4/5 Shared pretraining + per-market fine-tuning (fixed 10 epochs, then epochs chosen inside training)"
F=results/market_finetune_122
[ -f "$F/validation/preds_transformer_shared_ft__national.csv" ] || python scripts/run_market_finetune.py --config $C
python scripts/check_market_finetune.py --config $C
python scripts/evaluate_markets.py $F/validation/preds_*.csv results/long_hpi_122/validation/preds_transformer_per_market__national.csv \
    --config $C --reference transformer_shared__national \
    --compare transformer_shared_ft__national transformer_shared__national \
    --compare transformer_shared_ft__national transformer_per_market__national
[ -f "$F/inner_select/validation/preds_transformer_shared_ftsel__national.csv" ] || python scripts/run_market_finetune.py --config $C --select-epochs
python scripts/check_market_finetune.py --config $C --select
python scripts/evaluate_markets.py $F/inner_select/validation/preds_*.csv --config $C --reference transformer_shared__national \
    --compare transformer_shared_ftsel__national transformer_shared__national

# Recreate the Germany and Portugal case table from the saved forecasts.
python - "$C" "$F" <<'PYTHON'
import sys
from pathlib import Path

import pandas as pd

from housing_forecast.config import load_config
from housing_forecast.evaluation.markets import paired_ratio_bootstrap

cfg = load_config(sys.argv[1])
root = Path(sys.argv[2])
shared = "transformer_shared__national"
rows = []
for setting, folder, model in (
    ("Fixed", root / "validation", "transformer_shared_ft__national"),
    ("Chosen", root / "inner_select/validation", "transformer_shared_ftsel__national"),
):
    preds = pd.concat(
        [pd.read_csv(folder / f"preds_{name}.csv") for name in (model, shared)],
        ignore_index=True,
    )
    for market in ("Germany", "Portugal"):
        market_preds = preds[preds.series_id.eq(market)]
        for block in (4, 8, 12, 20):
            ci = paired_ratio_bootstrap(market_preds, model, shared, cfg, block_length=block)
            for r in ci[ci.horizon.isin(("h3-6", "h16-20"))].itertuples():
                rows.append({
                    "market": market, "setting": setting, "block": block, "horizon": r.horizon,
                    "ratio": round(r.median_ratio, 3), "lo": round(r.ci_low, 3), "hi": round(r.ci_high, 3),
                })
pd.DataFrame(rows).to_csv(root / "case_markets_ci.csv", index=False)
PYTHON

echo "== 5/5 Block-length check and training cost"
python scripts/block_length_check.py --config $C
python scripts/training_cost.py

echo "Done. New results: results/market_scale_122, results/market_holdout_122, results/market_finetune_122,"
echo "results/block_length_sensitivity_122.csv, results/training_cost.csv"
