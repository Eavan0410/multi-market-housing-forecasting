# Final test plan

Scope frozen on 11/19 following the 11/18 meeting. The run records its commit in `results/test_run_info.txt`. Keep that commit and the evaluation settings fixed for the full run. Report subsequent model changes as work developed after this test.

## Settings

- Target: cumulative real HPI log growth, 1–20 quarters ahead. Focus on 3–6 and 16–20 quarters.
- Inputs: eight quarters of real HPI growth, using `configs/experiment_long.toml`.
- Markets: all eligible national markets. Validation produced forecasts for 41 markets, with 38 meeting the scoring threshold.
- Timing: forecasts dated 122 days after quarter-end. This is an assumption based on recent Chilean nominal HPI releases. Historical availability of real HPI remains uncertain.
- Training: expanding window, refit every four origins. Training origins end at least four quarters before each fold. Mask labels until their estimated publication date; fit scaling on training data only.
- Test: starts in 2015Q1, with the end determined by available labels. Current data give 11 folds through 2025Q2, with 42 forecast origins at horizon 1 and 23 through 2020Q3 at horizon 20. Coverage varies by market.
- Transformers: seeds 0–4, model dimension 32, two layers, four heads, dropout 0.1 and learning rate 0.001. Select epochs within training. Local models and fine-tuning require at least 24 training rows.
- Baselines: no-change, per-market AR(4), pooled AR(4), pooled AR(8) and per-market ARIMA. Deterministic models use seed 0.
- Taiwan: deflate the nominal Sinyi index with CPI. It is not seasonally adjusted; OECD series are seasonally adjusted.

## Experiments

Run all agreed comparisons on the test period:

1. Shared Transformer versus per-market Transformer and the statistical baselines.
2. Local fine-tuning with ten fixed epochs and with epochs selected inside training.
3. Training pools of 5, 10, 20 and 41 markets, using all three fixed market orders.
4. Hold out Australia, Chile, Taiwan and Portugal separately. Compare with the single-threaded main run on matching forecast rows.

Noise, CPI, attribution and iMaPP analyses have completed exploratory validation runs. The iMaPP feature showed no clear improvement on matched samples and is reported alongside those results. The final test uses the HPI-only inputs listed above. The LLM-derived Taiwan policy index has been extracted and checked, but its forecasting value remains untested.

## Reporting

Report per-market RMSE and the median market RMSE ratio, averaging Transformer metrics over seeds. Require at least eight forecast dates per market and horizon. Include fitted-only comparisons, fine-tuning results on all and adapted samples, training-history groups, costs and failure cases.

Use 2,000 paired circular block bootstrap draws for 95% intervals. Report block lengths of 4, 8, 12 and 20 quarters. Long-horizon test samples are short, so interpret intervals alongside their block-length sensitivity.

Show validation and test results side by side, including unfavorable results. Disclose that the Taiwan pilot already examined 2015–2017 at horizons 1–4. Germany and Portugal were selected as fine-tuning examples during validation.

## Running

From the repository root, with the `hpf` environment active:

```bash
bash scripts/run_test_122.sh
```

The script requires committed code, configuration and data. Results may change. Completed training steps are reused only when their output checksums and command match the recorded commit. Interrupted outputs are archived before retrying; completion is recorded after all expected files pass validation.

Outputs are saved in the experiments' `test` folders, `results/block_length_sensitivity_122_test.csv`, and `results/market_finetune_122/case_markets_ci_test.csv`.
