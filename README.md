# Multi-Market Housing Price Forecasting

This project studies whether a single Transformer can forecast house prices across countries. We compare it with separate models for each market, test larger training pools, and forecast countries left out of training. The aim is to explore how far a shared model can carry across housing markets, including markets with short data histories.

The study asks three questions:

1. Can one shared model serve many housing markets?
2. How does it compare with fitting a separate model for each market?
3. How does forecast accuracy change from 1 to 20 quarters ahead?

The work started with Taiwan and now uses national house price series from OECD AnaHPI alongside Taiwan's Sinyi index. The main experiments use house price history alone and forecast 1 to 20 quarters ahead. We focus on 3–6 quarters for buying or selling and 16–20 quarters for construction. Earlier Taiwan experiments also included economic indicators and compared joint training with pretraining followed by fine-tuning. [Taiwan pilot](docs/taiwan_pilot.md)

The repository contains the data preparation code, a common backtest for all models, saved forecasts, error tables and plotting scripts. The results here are validation results. Final testing is still pending.

## Findings so far

1. **Market coverage.** One shared Transformer produced forecasts for 41 national markets, with 38 included in the main comparison. Expanding the training pool helped at 3–6 quarters in the market order tested. The four market holdouts gave examples of forecasting without local training data.
2. **Accuracy against separate models.** The shared model had lower error than separate Transformers and, from 5 quarters ahead, separate AR models across all four block lengths checked. Pooled AR was much closer to the shared Transformer.
3. **Forecast horizon.** The relative advantage over separate models grew at longer horizons, but absolute errors also grew. Against no-change, only the 1–3-quarter improvement held across all block lengths. Accuracy sufficient for buying, selling or construction decisions still needs to be established.

## Data and backtest

- **Model:** a small Transformer takes eight quarters of inflation-adjusted house price growth and predicts the total log change over each of the next 1–20 quarters.
- **Data:** national HPI series from OECD AnaHPI and Taiwan's Sinyi index. We build the panel from raw observations and adjust Taiwan's nominal HPI with CPI. Taiwan's index is not seasonally adjusted, while the OECD series are. We generated forecasts for 41 markets and report the 38 with at least eight forecast dates. [Market coverage](docs/market_list.csv) · [Data evidence](TODO_B.md) · [HPI release timing](docs/hpi_release_lags.md)
- **Backtest:** validation origins run from 2000Q1 to 2009Q4. We expand the training window and refit every four quarters. Training origins are at least four quarters earlier than the first validation origin in each fold. Scaling uses training data only, and training labels are masked until their estimated publication date. All markets share a forecast date 100 days after quarter-end. Publication lags remain estimates.
- **Baselines:** a Transformer trained separately for each market (it falls back to no-change when a market has too little history), per-market AR (falls back to pooled AR), one pooled AR across markets, and no-change.
- **Scoring:** RMSE per market and horizon, averaged over five seeds for each Transformer. We report the median across markets of RMSE(shared) / RMSE(baseline), so a ratio below 1 favors the shared model. Brackets are 95% intervals from a paired block bootstrap over forecast dates with 20-quarter blocks. We also check 4-, 8- and 12-quarter blocks. [Block-length check](results/block_length_sensitivity.csv)

## Multi-market results

### Shared and per-market models

| Shared Transformer vs | 3–6 quarters | 16–20 quarters |
|---|---:|---:|
| Per-market Transformer | 0.895 [0.796, 0.930] | 0.804 [0.791, 0.858] |
| Per-market AR | 0.943 [0.883, 0.977] | 0.869 [0.795, 0.949] |
| Pooled AR | 0.978 [0.918, 1.058] | 0.953 [0.917, 1.005] |
| No-change | 0.849 [0.649, 0.979] | 0.889 [0.831, 1.027] |

- **Per-market Transformer:** the interval is below 1 at every horizon and block length. About 23% of its forecasts were no-change fallbacks. Keeping only forecasts where both Transformers were trained (24 markets with medium or long histories) gives 0.816 [0.796, 0.899] and 0.788 [0.771, 0.836]. [Fitted-only check](results/long_hpi/validation/fitted_only_ratio_ci.csv)
- **Per-market AR:** the interval is below 1 from 5 quarters on at every block length.
- **Pooled AR:** its errors are close to the shared Transformer's, and the intervals include 1 in both ranges.
- **No-change:** at 3–6 quarters the interval includes 1 with 8- or 12-quarter blocks.

### Accuracy by horizon

![Median RMSE relative to no-change by forecast horizon](results/long_hpi/validation/rel_rmse_by_horizon.png)

The shaded ranges mark 3–6 and 16–20 quarters. The shared Transformer's median ratio to no-change stays between 0.77 and 0.89 at every horizon, while per-market AR falls behind no-change from 10 quarters on (1.09 at 20 quarters). The RMSE ratio against the per-market Transformer falls from about 0.88–0.90 at 1–8 quarters to 0.80–0.83 at 12–20 quarters. Absolute error grows fast: the shared model's median RMSE is 0.017 at one quarter and 0.235 at 20 quarters, in cumulative log growth. Whether errors of that size are useful for construction decisions is still open. [Errors by market and horizon](results/long_hpi/validation/metrics_by_market.csv)

### Training on more markets

We trained on 5, 10, 20 and 41 markets, adding markets in a region-rotation order, and scored every run on the same five markets and dates.

| Markets on the training list | 3–6 quarters | 16–20 quarters |
|---|---:|---:|
| 5 | Reference | Reference |
| 10 | 0.922 [0.797, 0.978] | 0.925 [0.798, 1.080] |
| 20 | 1.002 [0.862, 1.039] | 0.905 [0.679, 1.136] |
| 41 | 0.912 [0.802, 0.972] | 0.818 [0.651, 0.982] |

Depending on the fold, the four lists supplied training data from 3–5, 7–10, 15–20 and 24–38 markets. The 41-market run lowered error by about 9% at 3–6 quarters, and that interval stays below 1 at every block length. The 18% gain at 16–20 quarters is less certain: the interval includes 1 with 8- or 12-quarter blocks, and only three of the five markets improved. The middle runs were mixed, and we have tried only one order of adding markets. [Market selection](docs/market_selection.md) · [Results](results/market_scale/validation)

### Forecasting a market left out of training

We left out Australia, Chile, Taiwan and Portugal one at a time. At forecast time the model still sees that market's own recent HPI history. Ratios below 1 favor the model that never trained on the market.

| Held-out market | Dates | vs included, 3–6 q | vs included, 16–20 q | Against AR and no-change |
|---|---:|---:|---:|---|
| [Australia](results/market_holdout/australia/validation) | 40 | 1.008 [0.988, 1.040] | 1.013 [0.980, 1.051] | Intervals against both include 1 in these ranges |
| [Chile](results/market_holdout/chile/validation) | 24 | 1.021 [0.986, 1.051] | 0.901 [0.870, 0.951] | Better than both at 16–20 q; worse than AR at 3–6 q |
| [Taiwan](results/market_holdout/taiwan_national/validation) | 28 | 1.011 [1.005, 1.024] | 0.990 [0.979, 1.007] | About even with no-change; error about 24% above AR at 16–20 q |
| [Portugal](results/market_holdout/portugal/validation) | 40 | 0.956 [0.919, 0.978] | 1.009 [0.991, 1.030] | Worse than both at 16–20 q |

Leaving a market out usually changed RMSE by a few percent. Each held-out market supplied at most about 5% of the original training rows. The held-out model beat simple baselines in Chile at long horizons, while it over-predicted Portugal's five-year growth whether or not Portugal was in training. Chile's 10% gain at 16–20 quarters and Taiwan's gap to AR are not robust: their intervals include 1 with 8- or 12-quarter blocks. The last column highlights comparisons that hold across the four block lengths checked. Portugal uses a matched single-thread reference run, while the other markets use the original shared run. Run-to-run differences of a few percent mean small gaps between the Transformers need care.

## Limits and remaining work

- The results cover one validation period, which includes the 2008 crisis. The proposed test starts in 2015 and overlaps years examined in the earlier Taiwan pilot. [Taiwan pilot](docs/taiwan_pilot.md)
- Most comparisons have 40 forecast dates, so a draw with 20-quarter blocks uses only two blocks. Claims that depend on the block length are marked above.
- Taiwan remains a difficult case. Its shared-model errors are close to no-change in the ranges shown here.
- The broader study still needs per-market ARIMA, shared pretraining followed by per-market fine-tuning, the other two market orders, and the final test. We also need to finish checking release lags and compare model counts and training time consistently.

## Run locally

Use Python 3.11, from the repository root:

```bash
conda create -n hpf python=3.11 -y
conda activate hpf
pip install -e ".[dev,dl]"
pytest -q
python scripts/build_panel.py
```

Shared model and baselines:

```bash
C=configs/experiment_long.toml
python scripts/run_backtest.py --config $C --models no_change ar ar_pooled --pool national --experiment long_hpi
python scripts/run_backtest.py --config $C --models transformer transformer_per_market --pool national --experiment long_hpi
python scripts/evaluate_markets.py results/long_hpi/validation/preds_*.csv --config $C --reference no_change__national \
    --compare transformer__national transformer_per_market__national \
    --compare transformer__national ar__national \
    --compare transformer__national ar_pooled__national \
    --compare transformer__national no_change__national
python scripts/fitted_only_check.py
python scripts/block_length_check.py
```

Training on more markets:

```bash
OMP_NUM_THREADS=1 python scripts/run_market_scale.py
python scripts/evaluate_markets.py results/market_scale/validation/preds_*.csv \
    --reference transformer__scale_5 \
    --compare transformer__scale_10 transformer__scale_5 \
    --compare transformer__scale_20 transformer__scale_5 \
    --compare transformer__scale_41 transformer__scale_5
```

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
