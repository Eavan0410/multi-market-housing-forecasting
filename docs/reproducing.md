# Reproducing the experiments

Commands run from the repository root. See the [project overview](../README.md) and [frozen test plan](freeze.md).

## Run locally

Use Python 3.11 from the repository root.

```bash
conda create -n hpf python=3.11 -y
conda activate hpf
pip install -e ".[dev,dl,stats]" matplotlib
pytest -q
python scripts/build_panel.py
```

Main comparison at 122 days:

```bash
C=configs/experiment_long.toml
D=results/long_hpi_122/validation
OMP_NUM_THREADS=1 python scripts/run_backtest.py --config "$C" --models no_change ar ar_pooled arima transformer transformer_per_market --pool national --experiment long_hpi_122
python scripts/evaluate_markets.py "$D"/preds_*.csv --config $C --reference no_change__national \
    --compare transformer__national transformer_per_market__national \
    --compare transformer__national ar__national \
    --compare transformer__national arima__national \
    --compare transformer__national ar_pooled__national \
    --compare transformer__national no_change__national
python scripts/fitted_only_check.py --dir "$D"
python scripts/block_length_check.py --main-dir "$D" --out "$D/block_length_sensitivity.csv"
```

Training-pool size, market holdouts, fine-tuning, the block-length check and training cost, all at 122 days (about an hour; finished steps are skipped on a rerun). The individual commands are listed in the script.

```bash
bash scripts/rerun_secondary_122.sh
OMP_NUM_THREADS=1 python scripts/training_cost.py --time    # optional timing
```

Pooled AR(8) and the explainability analysis (the second command retrains the shared model and checks it against the saved forecasts):

```bash
python scripts/run_backtest.py --config configs/experiment_long.toml --models ar_pooled8 --pool national --experiment long_hpi_122
OMP_NUM_THREADS=1 python scripts/explain_lag_weights.py
python scripts/explain_by_regime.py
OMP_NUM_THREADS=1 python scripts/check_ig_steps.py
```

Noise and CPI checks:

```bash
OMP_NUM_THREADS=1 python scripts/run_noise_robustness.py
pip install requests && python scripts/download_cpi_gaps.py   # once, needs internet: BIS CPI for Japan and Russia, FRED for New Zealand
python scripts/build_cpi_assessment.py
OMP_NUM_THREADS=1 python scripts/run_cpi_experiment.py
```

The earlier 100-day runs can be reproduced from commit `d48a065`.

### Final test

The 11/18 meeting confirmed the main comparisons, both fine-tuning setups, all three training-pool orders and four market holdouts. Test-phase support and the resumable entry point are implemented. The final run completed on November 19, 2026 (New York time), using commit `b474b77`. Results are summarized in [Final test results](results.md#final-test-results). [Frozen settings](../docs/freeze.md)

```bash
bash scripts/run_test_122.sh
```

Run from the repository root in the `hpf` environment after committing the code, configuration and source data. Completed steps are checked and reused; interrupted outputs are archived before retrying. Results go to each experiment’s `test` folder. Noise, CPI, attribution and iMaPP results are exploratory validation analyses; the final test compares the frozen HPI-only models.

More detail: [backtest notes](../docs/backtest_framework.md) · [experiment settings](../docs/experiment_contract.md) · [data formats](../docs/design_notes.md)
