#!/usr/bin/env bash
# Run from the repository root with the hpf environment active.
set -euo pipefail
export OMP_NUM_THREADS=1
C=configs/experiment_long.toml
T="--phase test --unlock-test"
INFO=results/test_run_info.txt

echo "== 0/7 Check the frozen commit"
if [ -n "$(git status --porcelain -- . ':(exclude)results')" ]; then
  echo "Uncommitted changes outside results/:"; git status --short -- . ':(exclude)results'
  echo "Commit the frozen state first (docs/freeze.md), then rerun."; exit 1
fi
RUN_COMMIT=$(git rev-parse HEAD)
if [ -f "$INFO" ]; then
  RECORDED=$(grep '^commit' "$INFO" | awk '{print $2}')
  if [ "$RECORDED" != "$RUN_COMMIT" ]; then
    echo "results/test_run_info.txt records commit $RECORDED but HEAD is $RUN_COMMIT. The test must be run on one commit."; exit 1
  fi
else
  mkdir -p results
  PYTHON_VERSION=$(python -c 'import platform; print(platform.python_version())')
  TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)')
  INFO_TEMP=$(mktemp "results/.test-run-info.XXXXXX")
  {
    echo "commit $RUN_COMMIT"
    echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "python $PYTHON_VERSION"
    echo "torch $TORCH_VERSION"
  } > "$INFO_TEMP"
  mv "$INFO_TEMP" "$INFO"
fi
echo "Frozen commit $RUN_COMMIT"

echo "== 1/7 Main models (validation settings, test period)"
M=results/long_hpi_122/test
run_step() {
  local directory=$1 name=$2 models=$3 diagnostics=$4
  shift 4
  local files=() model diagnostic
  for model in $models; do
    files+=("preds_${model}.csv" "preds_${model}.meta.json")
  done
  for diagnostic in $diagnostics; do
    files+=("$diagnostic")
  done
  python scripts/run_test_step.py --directory "$directory" --name "$name" --files "${files[@]}" -- "$@"
}
run_step "$M" baselines "no_change__national ar__national ar_pooled__national ar_pooled8__national arima__national" \
  "diag_arima__national.csv" \
  python scripts/run_backtest.py --config $C $T --models no_change ar ar_pooled ar_pooled8 arima --pool national --experiment long_hpi_122
run_step "$M" transformers "transformer__national transformer_per_market__national" \
  "diag_transformer_per_market__national.csv" \
  python scripts/run_backtest.py --config $C $T --models transformer transformer_per_market --pool national --experiment long_hpi_122

echo "== 2/7 Main evaluation, failure cases and fitted-only check"
python scripts/evaluate_markets.py $M/preds_*.csv --config $C --reference no_change__national \
    --compare transformer__national transformer_per_market__national \
    --compare transformer__national ar__national \
    --compare transformer__national arima__national \
    --compare transformer__national ar_pooled__national \
    --compare transformer__national ar_pooled8__national \
    --compare transformer__national no_change__national
python scripts/evaluate_markets.py $M/metrics_by_market.csv --plot-failure-cases
python scripts/fitted_only_check.py --dir $M --config $C

echo "== 3/7 Shared pretraining + per-market fine-tuning (both setups)"
F=results/market_finetune_122
run_step "$F/test" finetune "transformer_shared_ft__national transformer_shared__national" "fit_diagnostics.csv" \
  python scripts/run_market_finetune.py --config $C $T
python scripts/check_market_finetune.py --config $C --phase test
python scripts/evaluate_markets.py $F/test/preds_*.csv $M/preds_transformer_per_market__national.csv \
    --config $C --reference transformer_shared__national \
    --compare transformer_shared_ft__national transformer_shared__national \
    --compare transformer_shared_ft__national transformer_per_market__national
run_step "$F/inner_select/test" finetune_selected "transformer_shared_ftsel__national transformer_shared__national" "fit_diagnostics.csv" \
  python scripts/run_market_finetune.py --config $C $T --select-epochs
python scripts/check_market_finetune.py --config $C --phase test --select
python scripts/evaluate_markets.py $F/inner_select/test/preds_*.csv --config $C --reference transformer_shared__national \
    --compare transformer_shared_ftsel__national transformer_shared__national

echo "== 4/7 Training-pool size, three market orders"
SCALE_MODELS="transformer__scale_5 transformer__scale_10 transformer__scale_20 transformer__scale_41"
run_step results/market_scale_122/test scale_region "$SCALE_MODELS" training_coverage.csv \
  python scripts/run_market_scale.py --config $C $T
for order in history random; do
  run_step "results/market_scale_122/$order/test" "scale_$order" "$SCALE_MODELS" training_coverage.csv \
    python scripts/run_market_scale.py --config $C $T --order "$order"
done
for d in results/market_scale_122/test results/market_scale_122/history/test results/market_scale_122/random/test; do
  python scripts/evaluate_markets.py $d/preds_*.csv --config $C --reference transformer__scale_5 \
      --compare transformer__scale_10 transformer__scale_5 \
      --compare transformer__scale_20 transformer__scale_5 \
      --compare transformer__scale_41 transformer__scale_5
done

echo "== 5/7 Leave-one-market-out (reference: the single-threaded main test run)"
for m in Australia Chile Taiwan_national Portugal; do
  d=results/market_holdout_122/$(echo $m | tr '[:upper:]' '[:lower:]')/test
  run_step "$d" holdout "transformer__holdout transformer__national ar__national no_change__national" training_coverage.csv \
    python scripts/run_market_holdout.py --config $C --market "$m" $T
  python scripts/evaluate_markets.py $d/preds_*.csv --config $C --reference no_change__national \
      --compare transformer__holdout no_change__national \
      --compare transformer__holdout ar__national \
      --compare transformer__holdout transformer__national
done

echo "== 6/7 Block-length check"
python scripts/block_length_check.py --config $C --phase test

echo "== 7/7 Germany and Portugal: fine-tuning vs shared model, by block length"
python - <<'PY'
import pandas as pd
from housing_forecast.config import load_config
from housing_forecast.evaluation.markets import paired_ratio_bootstrap

cfg = load_config("configs/experiment_long.toml")
rows = []
for setting, folder, model in (("Fixed", "results/market_finetune_122/test", "transformer_shared_ft__national"),
                               ("Chosen", "results/market_finetune_122/inner_select/test",
                                "transformer_shared_ftsel__national")):
    preds = pd.concat([pd.read_csv(f"{folder}/preds_transformer_shared__national.csv"),
                       pd.read_csv(f"{folder}/preds_{model}.csv")], ignore_index=True)
    for market in ("Germany", "Portugal"):
        q = preds[preds["series_id"].eq(market)]
        for block in (4, 8, 12, 20):
            try:
                ci = paired_ratio_bootstrap(q, model, "transformer_shared__national", cfg, block_length=block)
            except ValueError as e:
                print(f"skip {market} {setting} block {block}: {e}")
                continue
            for r in ci[ci["horizon"].isin(["h3-6", "h16-20"])].itertuples():
                rows.append({"market": market, "setting": setting, "block": block, "horizon": r.horizon,
                             "ratio": r.median_ratio, "lo": r.ci_low, "hi": r.ci_high})
out = pd.DataFrame(rows)
out.to_csv("results/market_finetune_122/case_markets_ci_test.csv", index=False)
print(out.round(3).to_string(index=False))
PY

echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$INFO"
echo "Done. Test results: results/long_hpi_122/test, results/market_finetune_122/test,"
echo "results/market_finetune_122/inner_select/test, results/market_scale_122/*/test, results/market_holdout_122/*/test,"
echo "results/block_length_sensitivity_122_test.csv, results/market_finetune_122/case_markets_ci_test.csv"
