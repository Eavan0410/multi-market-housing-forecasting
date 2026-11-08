# Freeze plan for the final test (draft for 11/19)

Status: **draft**. Items marked **Decide** need agreement with Josh and the team before the test period is unlocked. Once the test is run, nothing listed under "Frozen" may change, and the test results may not be used to choose models, settings or comparisons.

## Frozen settings

- **Config:** `configs/experiment_long.toml` as committed at the freeze. Record its commit hash here when frozen.
- **Target:** cumulative real log growth, 1–20 quarters ahead. Focus ranges: 3–6 and 16–20 quarters.
- **Input:** eight quarters of real HPI growth (`hpi_qoq`) only.
- **Markets:** all national markets with HPI data (41 had validation forecasts). Taiwan's Sinyi HPI is deflated with CPI and not seasonally adjusted.
- **Timing:** common forecast date 100 days after quarter-end (`forecast_lag_days = "max_hpi_lag"`). Training labels are masked until their estimated publication date. *If B's release-lag check finds a market above 100 days before the freeze, update both configs, rerun validation, then freeze.*
- **Backtest:** expanding window, one refit every four test origins, training origins at least four quarters before each fold's first test origin.
- **Test period:** forecast origins from 2015Q1 to the last origin with a label (`test_end = "auto"`). The models train on everything before each fold, including the validation years. Horizon-20 labels end around 2020Q2, so long horizons have fewer test dates than short ones.
- **Seeds:** 0–4 for the Transformers; seed 0 for deterministic models.
- **Model settings:** as used in validation. Transformer: d_model 32, 2 layers, 4 heads, dropout 0.1, learning rate 1e-3, epoch count chosen on the inner time split. Per-market Transformer and both fine-tuning settings use at least 24 local training rows. AR order 4.

## Models run on the test period

| Model | Label | Role |
|---|---|---|
| Shared Transformer | `transformer__national` | main model |
| Per-market Transformer | `transformer_per_market__national` | one model per market (Q2) |
| Per-market AR | `ar__national` | one model per market (Q2) |
| Per-market ARIMA | `arima__national` | one model per market (Q2); validation run complete |
| Pooled AR | `ar_pooled__national` | shared linear model |
| No-change | `no_change__national` | benchmark |
| Shared + fine-tuning, fixed 10 epochs | `transformer_shared_ft__national` | third Transformer comparison |
| Shared + fine-tuning, epochs chosen inside training | `transformer_shared_ftsel__national` | third Transformer comparison (added after the fixed results) |

**Decide:** whether the training-pool runs (5/10/20/41 markets) and the four holdouts are repeated on the test period. They are secondary analyses. `run_market_scale.py`, `run_market_holdout.py` and `run_market_finetune.py` currently write validation results only and need a `--phase test --unlock-test` option before they can run on the test period.

## Comparisons and how they are reported

Primary (the answers to the three questions):

1. Shared Transformer vs per-market Transformer, per-market AR, per-market ARIMA, pooled AR and no-change.
2. Metric: median across markets of RMSE(shared) / RMSE(baseline), RMSE averaged over seeds, reported for every horizon and for the 3–6 and 16–20 ranges.
3. Uncertainty: paired circular block bootstrap over forecast dates, 2,000 draws, 95% intervals, 20-quarter blocks, with 4-, 8- and 12-quarter blocks reported as a sensitivity check (`scripts/block_length_check.py`).
4. Markets with fewer than eight test dates at a horizon are left out of that horizon's summary.

Secondary: fine-tuning vs its own shared starting model (all forecasts and adapted-only), the fitted-only check for the per-market Transformer, results by history length, and Germany and Portugal as the examples already named in validation.

All results are reported, including comparisons that do not favor the shared model. Validation and test numbers are shown side by side.

## Not allowed after unlocking

- Changing any frozen setting, model or comparison because of test results.
- Adding markets, features, seeds or fine-tuning settings, or choosing a block length, based on test results.
- Rerunning the test with different settings and reporting the better run.

New ideas after the freeze (for example the policy index or a lookback of 16 quarters) are reported as exploratory, or are developed on validation and frozen separately before they touch the test period.

## Open questions for Josh

1. **Test period overlap.** The test starts in 2015Q1 and overlaps the 2012–2017 years examined in the early Taiwan pilot. Is that acceptable if we state it, or should the split change?
2. **Course test vs paper test.** If the project continues toward a paper, is this test run also the paper's test, with no further model changes afterwards? Or should a later period be kept unseen for the paper?
3. **Fine-tuning.** Should the fine-tuning comparison count as a core result or a secondary one?
4. **Secondary experiments on the test period.** Should the training-pool and holdout analyses be repeated on the test period?

## Run order on the frozen commit

```bash
C=configs/experiment_long.toml
python scripts/run_backtest.py --config $C --phase test --unlock-test --models no_change ar ar_pooled arima --pool national --experiment long_hpi
OMP_NUM_THREADS=1 python scripts/run_backtest.py --config $C --phase test --unlock-test --models transformer transformer_per_market --pool national --experiment long_hpi
python scripts/evaluate_markets.py results/long_hpi/test/preds_*.csv --config $C --reference no_change__national \
    --compare transformer__national transformer_per_market__national \
    --compare transformer__national ar__national \
    --compare transformer__national arima__national \
    --compare transformer__national ar_pooled__national \
    --compare transformer__national no_change__national
```

The fine-tuning runs are added once their script supports the test phase.
