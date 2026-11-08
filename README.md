# Multi-Market Housing Price Forecasting

This project studies whether a single Transformer can forecast house prices across countries. We compare it with separate models for each market, test larger training pools, and forecast countries left out of training. The aim is to explore how far a shared model can carry across housing markets, including markets with short data histories.

The study asks three questions:

1. Can one shared model serve many housing markets?
2. How does it compare with fitting a separate model for each market?
3. How does forecast accuracy change from 1 to 20 quarters ahead?

The work started with Taiwan and now uses national house price series from OECD AnaHPI alongside Taiwan's Sinyi index. The main experiments use house price history alone and forecast 1 to 20 quarters ahead. We focus on 3–6 quarters for buying or selling and 16–20 quarters for construction. Earlier Taiwan experiments also included economic indicators and compared joint training with pretraining followed by fine-tuning. [Taiwan pilot](docs/taiwan_pilot.md)

The repository contains the data preparation code, a common backtest for all models, saved forecasts, error tables and plotting scripts. The results here are validation results. Final testing is still pending.

## Findings so far

1. **Market coverage.** One shared Transformer produced forecasts for 41 national markets, with 38 included in the main comparison. The full training pool improved 3–6-quarter forecasts, while the intermediate results varied across three orders of adding markets. The four market holdouts gave examples of forecasting without local training data.
2. **Accuracy against separate models.** The shared model had lower error than separate Transformers and, from 5 quarters ahead, separate AR models across all four block lengths checked. The shared model also had lower error than per-market ARIMA at 16–20 quarters across all four block lengths. Pooled AR was much closer to the shared Transformer. Fine-tuning a copy for each market did not improve on the shared model overall, although Germany and Portugal benefited.
3. **Forecast horizon.** The relative advantage over separate models grew at longer horizons, but absolute errors also grew. Against no-change, only the 1–3-quarter improvement held across all block lengths. Accuracy sufficient for buying, selling or construction decisions still needs to be established.

## Data and backtest

- **Model:** a small Transformer takes eight quarters of inflation-adjusted house price growth and predicts the total log change over each of the next 1–20 quarters.
- **Data:** national HPI series from OECD AnaHPI and Taiwan's Sinyi index. We build the panel from raw observations and adjust Taiwan's nominal HPI with CPI. Taiwan's index is not seasonally adjusted, while the OECD series are. We generated forecasts for 41 markets and report the 38 with at least eight forecast dates. [Market coverage](docs/market_list.csv) · [Data evidence](TODO_B.md) · [HPI release timing](docs/hpi_release_lags.md)
- **Backtest:** validation origins run from 2000Q1 to 2009Q4. We expand the training window and refit every four quarters. Training origins are at least four quarters earlier than the first validation origin in each fold. Scaling uses training data only, and training labels are masked until their estimated publication date. All markets share a forecast date 100 days after quarter-end. Publication lags remain estimates.
- **Baselines:** a Transformer trained separately for each market (it falls back to no-change when a market has too little history), per-market AR (falls back to pooled AR), per-market ARIMA (falls back to its own average growth when history is too short or a fit fails), one pooled AR across markets, and no-change.
- **Scoring:** RMSE per market and horizon, averaged over five seeds for each Transformer. We report the median across markets of RMSE(shared) / RMSE(baseline), so a ratio below 1 favors the shared model. Brackets are 95% intervals from a paired block bootstrap over forecast dates with 20-quarter blocks. We also check 4-, 8- and 12-quarter blocks. [Block-length check](results/block_length_sensitivity.csv)

## Multi-market results

### Shared and per-market models

| Shared Transformer vs | 3–6 quarters | 16–20 quarters |
|---|---:|---:|
| Per-market Transformer | 0.895 [0.796, 0.930] | 0.804 [0.791, 0.858] |
| Per-market AR | 0.943 [0.883, 0.977] | 0.869 [0.795, 0.949] |
| Per-market ARIMA | 0.948 [0.859, 0.999] | 0.910 [0.841, 0.975] |
| Pooled AR | 0.978 [0.918, 1.058] | 0.953 [0.917, 1.005] |
| No-change | 0.849 [0.649, 0.979] | 0.889 [0.831, 1.027] |

- **Per-market Transformer:** the interval is below 1 at every horizon and block length. About 23% of its forecasts were no-change fallbacks. Keeping only forecasts where both Transformers were trained (24 markets with medium or long histories) gives 0.816 [0.796, 0.899] and 0.788 [0.771, 0.836]. [Fitted-only check](results/long_hpi/validation/fitted_only_ratio_ci.csv)
- **Per-market AR:** the interval is below 1 from 5 quarters on at every block length.
- **Per-market ARIMA:** the order (p ≤ 4, q ≤ 2) is chosen by AIC on each market's training history. Markets with fewer than 24 training quarters use their own average growth; this covers about 20% of forecasts, and no fit failed. The interval is below 1 at 16–20 quarters for every block length, but at 3–6 quarters only with 20-quarter blocks. Like per-market AR, it is about even with or behind no-change from 10 quarters on (1.10 at 20 quarters). [ARIMA fits and fallbacks](results/long_hpi/validation/diag_arima__national.csv)
- **Pooled AR:** its errors are close to the shared Transformer's, and the intervals include 1 in both ranges.
- **No-change:** at 3–6 quarters the interval includes 1 with 8- or 12-quarter blocks.

**Training cost.** Over the 10 validation folds and five seeds, the shared Transformer trains 50 networks, while one Transformer per market trains 1,165. Shared-model fits include an extra training run to choose the epoch count. Per-market models do this when enough history is available. The per-market count excludes markets that fell back to no-change. On one Apple-silicon CPU thread, the largest fold took 5.5 seconds for the shared model and 7.7 seconds for the per-market models, with one seed. The timing difference was small in this single-fold measurement. The shared approach uses one model across markets, while the per-market approach needs a separate model for each market with enough history. The shared model also covers markets too short to train on their own. Fine-tuning took 6.4 seconds with fixed epochs and 18.3 seconds with epochs chosen inside training. Both AR baselines took under 0.2 seconds. [Model counts](results/training_cost.csv) · [Timing](results/training_time.csv)

### Shared pretraining and local fine-tuning

We adapted a copy of the shared model to each market's own training history. Each fold starts from one shared model trained on the available markets, including each target market's past data. The holdout experiment below instead excludes the target market from training. Each market's copy keeps the shared scaling and uses a learning rate of 0.0001. Markets with fewer than 24 training rows keep the shared forecast. We tried two settings:

- **Fixed:** every eligible market is fine-tuned for 10 epochs.
- **Chosen inside training:** added after we had seen the fixed results. Within each fold, the training data are split by date: the shared model is pretrained and rescaled on the earlier part, and each market's epoch count, from 0 (keep the shared model) to 30, is chosen on the later part. The model is then refitted on the full fold with the chosen counts. Markets without enough rows in the earlier part keep the shared model.

Ratios compare fine-tuning with the exact shared model it started from; both settings start from identical shared forecasts. Below 1 favors fine-tuning. The two rows use different market samples, so they should not be compared with each other.

| Setting | Markets compared | 3–6 quarters | 16–20 quarters |
|---|---:|---:|---:|
| Fixed 10 epochs (folds where all seeds were fine-tuned) | 24 | 1.022 [0.985, 1.038] | 1.038 [0.994, 1.082] |
| Chosen inside training (folds where a choice was made) | 22 | 1.020 [1.000, 1.037] | 1.032 [0.995, 1.046] |

**Neither fine-tuning setting showed an overall improvement, but some markets benefited.** Among the markets and folds included in the table, median error was about 2–4% higher than with the shared model. Most intervals include 1; the exception is the chosen setting at 3–6 quarters with 4-quarter blocks, where the interval lies just above 1. When a choice was possible, the training data favored keeping the shared model 42% of the time. Most short-history markets, including Taiwan, did not have enough data for fine-tuning under these rules, so we could not assess its benefit there. Both settings still beat the separate Transformers trained from scratch; for the fixed setting the ratios were 0.896 [0.779, 0.938] and 0.843 [0.831, 0.897].

Germany and Portugal, two of the shared model's weakest markets, improved in both settings. The chosen setting fine-tuned them in every fold and seed where a choice was possible. Over all of each market's forecasts:

| Market | Setting | 3–6 quarters | 16–20 quarters |
|---|---|---:|---:|
| Germany | Fixed | 0.735 [0.654, 0.788] | 0.649 [0.622, 0.672] |
| Germany | Chosen | 0.690 [0.621, 0.733] | 0.540 [0.519, 0.570] |
| Portugal | Fixed | 0.884 [0.794, 0.921] | 0.852 [0.819, 0.889] |
| Portugal | Chosen | 0.894 [0.835, 0.958] | 0.826 [0.757, 0.925] |

All eight intervals stay below 1 with 4-, 8-, 12- and 20-quarter blocks. Fine-tuning narrows these markets' gap rather than closing it: after the chosen fine-tuning, Germany's 16–20-quarter error is still about 11% above per-market AR (1.108 [1.041, 1.170]) and both markets remain above no-change in that range. We picked these two markets as examples after seeing the fixed results, so they show where local adaptation can help rather than a tested selection rule. Other markets lost accuracy, for example Switzerland (1.41 fixed, 1.15 chosen at 16–20 quarters). [Fixed setting](results/market_finetune/validation) · [Chosen setting](results/market_finetune/inner_select/validation) · [Germany and Portugal intervals](results/market_finetune/case_markets_ci.csv)

### Accuracy by horizon

![Median RMSE relative to no-change by forecast horizon](results/long_hpi/validation/rel_rmse_by_horizon.png)

The per-market Transformer falls back to no-change when local history is too short. These fallbacks are included in the curve. The fitted-only comparison above checks whether the shared model still performs better when both Transformers were actually trained.

The curves compare the shared Transformer and four baselines with no-change. The shaded ranges mark 3–6 and 16–20 quarters. The shared Transformer's median ratio to no-change stays between 0.77 and 0.89 at every horizon, while per-market AR falls behind no-change from 10 quarters on (1.09 at 20 quarters). The RMSE ratio against the per-market Transformer falls from about 0.88–0.90 at 1–8 quarters to 0.80–0.83 at 12–20 quarters. Absolute error grows fast: the shared model's median RMSE is 0.017 at one quarter and 0.235 at 20 quarters, in cumulative log growth. Whether errors of that size are useful for construction decisions is still open. [Errors by market and horizon](results/long_hpi/validation/metrics_by_market.csv)

### Training on more markets

We trained on nested pools of 5, 10, 20 and 41 markets in three orders, always scoring the same five markets on the same dates. Ratios below 1 mean lower error than the five-market run. The 5- and 41-market pools are identical across orders, so those forecasts are reused.

| Order of adding markets | Markets on the list | Markets used per fold | 3–6 quarters | 16–20 quarters |
|---|---:|---:|---:|---:|
| Common starting pool | 5 | 3–5 | Reference | Reference |
| Region rotation | 10 | 7–10 | 0.922 [0.797, 0.978] | 0.925 [0.798, 1.080] |
| Region rotation | 20 | 15–20 | 1.002 [0.862, 1.039] | 0.905 [0.679, 1.136] |
| Long history first | 10 | 8–10 | 0.999 [0.761, 1.063] | 0.895 [0.670, 1.073] |
| Long history first | 20 | 18–20 | 0.913 [0.858, 0.978] | 0.835 [0.619, 1.079] |
| Fixed random order | 10 | 6–10 | 0.907 [0.801, 0.949] | 0.994 [0.877, 1.045] |
| Fixed random order | 20 | 11–19 | 0.883 [0.834, 0.951] | 0.836 [0.696, 1.119] |
| Common full pool | 41 | 24–38 | 0.912 [0.802, 0.972] | 0.818 [0.651, 0.982] |

The intermediate results depend on which markets are added. At 20 markets, the 3–6-quarter error was about even with the five-market run under region rotation, 9% lower with long histories first and 12% lower with the random order. More markets did not improve every step.

The full pool lowered error by about 9% at 3–6 quarters and 18% at 16–20 quarters. The shorter-range interval stays below 1 at all block lengths checked. The longer-range interval includes 1 with 8- or 12-quarter blocks, and only three of the five markets improved in that range. [Market selection](docs/market_selection.md) · [Region rotation](results/market_scale/validation) · [Long history first](results/market_scale/history/validation) · [Random order](results/market_scale/random/validation)

### Forecasting a market left out of training

We left out Australia, Chile, Taiwan and Portugal one at a time. At forecast time the model still sees that market's own recent HPI history. Ratios below 1 favor the model that never trained on the market.

| Held-out market | Dates | vs included, 3–6 q | vs included, 16–20 q | Against AR and no-change |
|---|---:|---:|---:|---|
| [Australia](results/market_holdout/australia/validation) | 40 | 1.008 [0.988, 1.040] | 1.013 [0.980, 1.051] | Intervals against both include 1 in these ranges |
| [Chile](results/market_holdout/chile/validation) | 24 | 1.021 [0.986, 1.051] | 0.901 [0.870, 0.951] | Better than both at 16–20 q; worse than AR at 3–6 q |
| [Taiwan](results/market_holdout/taiwan_national/validation) | 28 | 1.011 [1.005, 1.024] | 0.990 [0.979, 1.007] | About even with no-change; error about 24% above AR at 16–20 q, but this gap depends on block length |
| [Portugal](results/market_holdout/portugal/validation) | 40 | 0.956 [0.919, 0.978] | 1.009 [0.991, 1.030] | Worse than both at 16–20 q |

Leaving a market out usually changed RMSE by a few percent. Each held-out market supplied at most about 5% of the original training rows. The held-out model beat simple baselines in Chile at long horizons, while it over-predicted Portugal's five-year growth whether or not Portugal was in training. Chile's 10% gain at 16–20 quarters and Taiwan's gap to AR are not robust: their intervals include 1 with 8- or 12-quarter blocks. Except for Taiwan's noted gap to AR, the last column highlights comparisons that hold across the four block lengths checked. Portugal uses a matched single-thread reference run, while the other markets use the original shared run. Run-to-run differences of a few percent mean small gaps between the Transformers need care.

## Limits and remaining work

- The results cover one validation period, which includes the 2008 crisis. The proposed test starts in 2015 and overlaps years examined in the earlier Taiwan pilot. [Taiwan pilot](docs/taiwan_pilot.md)
- Most comparisons have 40 forecast dates, so a draw with 20-quarter blocks uses only two blocks. Claims that depend on the block length are marked above.
- Taiwan remains a difficult case. Its shared-model errors are close to no-change in the ranges shown here.
- The broader study still needs the final test. We also need to finish checking release lags.

## Run locally

Use Python 3.11, from the repository root:

```bash
conda create -n hpf python=3.11 -y
conda activate hpf
pip install -e ".[dev,dl,stats]" matplotlib
pytest -q
python scripts/build_panel.py
```

Shared model and baselines:

```bash
C=configs/experiment_long.toml
python scripts/run_backtest.py --config $C --models no_change ar ar_pooled arima --pool national --experiment long_hpi
python scripts/run_backtest.py --config $C --models transformer transformer_per_market --pool national --experiment long_hpi
python scripts/evaluate_markets.py results/long_hpi/validation/preds_*.csv --config $C --reference no_change__national \
    --compare transformer__national transformer_per_market__national \
    --compare transformer__national ar__national \
    --compare transformer__national arima__national \
    --compare transformer__national ar_pooled__national \
    --compare transformer__national no_change__national
python scripts/fitted_only_check.py
python scripts/block_length_check.py
OMP_NUM_THREADS=1 python scripts/training_cost.py --time   # model counts and one-fold timing
```

Shared pretraining and local fine-tuning (the output directory must be new):

```bash
OMP_NUM_THREADS=1 python scripts/run_market_finetune.py
python scripts/evaluate_markets.py results/market_finetune/validation/preds_*.csv \
    results/long_hpi/validation/preds_transformer_per_market__national.csv \
    --reference transformer_shared__national \
    --compare transformer_shared_ft__national transformer_shared__national \
    --compare transformer_shared_ft__national transformer_per_market__national
python scripts/check_market_finetune.py
# Epochs chosen inside training (second setting)
OMP_NUM_THREADS=1 python scripts/run_market_finetune.py --select-epochs
python scripts/check_market_finetune.py --select
```

Training on more markets:

```bash
OMP_NUM_THREADS=1 python scripts/run_market_scale.py --order region
OMP_NUM_THREADS=1 python scripts/run_market_scale.py --order history
OMP_NUM_THREADS=1 python scripts/run_market_scale.py --order random
python scripts/evaluate_markets.py results/market_scale/validation/preds_*.csv \
    --reference transformer__scale_5 \
    --compare transformer__scale_10 transformer__scale_5 \
    --compare transformer__scale_20 transformer__scale_5 \
    --compare transformer__scale_41 transformer__scale_5
```

Use the same evaluation command with `results/market_scale/history/validation/` or `results/market_scale/random/validation/` to score the other orders.

Leaving a market out (`--market Australia`, `Chile`, `Taiwan_national` or `Portugal`). The Portugal comparison uses a single-thread rerun of the reference models, passed with `--reference-dir`.

```bash
OMP_NUM_THREADS=1 python scripts/run_market_holdout.py --market Australia
python scripts/evaluate_markets.py results/market_holdout/australia/validation/preds_*.csv \
    --reference no_change__national \
    --compare transformer__holdout no_change__national \
    --compare transformer__holdout ar__national \
    --compare transformer__holdout transformer__national
```

More detail: [backtest notes](docs/backtest_framework.md) · [experiment settings](docs/experiment_contract.md) · [data formats](docs/design_notes.md)

## Taiwan policy extension

As an extension, we use a local language model to extract policy signals from Taiwan announcements. Qwen2.5-14B (GPTQ 4-bit) has processed 30 announcements from 2020–2025. The outputs and a quarterly index are saved. Manual checking is pending. These sample-based policy signals have not yet been tested as forecasting inputs. [Outputs](results/policy) · [Setup](data/raw/policy/README.md)

## Repository layout

- `src/housing_forecast/` contains the data, model, backtest and evaluation code.
- `scripts/` contains the commands used to run the experiments.
- `configs/` holds the experiment and policy settings.
- `results/` contains saved forecasts, error tables, confidence intervals and figures.
- `tests/` checks the data rules, model interfaces and evaluation.
