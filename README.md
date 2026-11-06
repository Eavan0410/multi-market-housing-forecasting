# Multi-Market Housing Price Forecasting

Can one model forecast house prices across many countries? We train a shared Transformer on quarterly house price data and compare it with models fitted separately for each market. We also test what happens when we add training markets or ask the model to forecast a market it never trained on.

The project grew out of a forecasting study on Taiwan. The main experiments cover 1 to 20 quarters ahead, with a closer look at 3–6 quarters for buying or selling and 16–20 quarters for construction. [Early Taiwan experiments](docs/taiwan_pilot.md)

## Data and evaluation

The model takes eight quarters of inflation-adjusted house price growth and predicts the total log change in prices over the forecast horizon. The validation forecasts start between 2000Q1 and 2009Q4. We refit every four quarters using the data available at the time. All markets share a forecast date 100 days after quarter-end in this run, while each market's training labels follow its own release lag. Each fold uses training forecast dates at least four quarters before its first validation date. We fit input scaling on training data only.

We produced forecasts for 41 national markets and report the 38 with at least eight forecast dates. To compare models, we average RMSE over five seeds for each Transformer, then take the ratio of errors for each market and report the median across markets. A ratio below 1 favors the first model. Brackets show 95% confidence intervals based on resampling the same 20-quarter blocks for both models and all markets.

The data come from the Spring 2026 dataset, including AnaHPI and Taiwan's HPI and CPI. Publication lags are estimates, and the source data include later revisions. We treat Taiwan's HPI as nominal and adjust it using CPI. Its price basis still needs confirmation. [Data formats](docs/design_notes.md) · [Release lag evidence](results/availability_matrix.csv)

## Multi-market results

### One shared model versus separate models

The shared Transformer had lower error than the per-market Transformer in both forecast ranges. It was close to per-market AR one quarter ahead and had about 13% lower error over 16–20 quarters. Pooled AR, which fits one linear model across markets, was closer to the shared Transformer. The 95% intervals for that comparison include 1 in both ranges.

| Baseline | 3–6 quarters | 16–20 quarters |
|---|---:|---:|
| Per-market Transformer | 0.895 [0.796, 0.930] | 0.804 [0.791, 0.858] |
| Per-market AR | 0.943 [0.883, 0.977] | 0.869 [0.795, 0.949] |
| Pooled AR | 0.978 [0.918, 1.058] | 0.953 [0.917, 1.005] |
| No-change | 0.849 [0.649, 0.979] | 0.889 [0.831, 1.027] |

The per-market Transformer used a no-change forecast when there was too little training history, about 23% of the scored forecasts. We repeated the comparison using only dates when both Transformers were fitted. The shared model still had lower error, with ratios of 0.816 [0.796, 0.899] at 3–6 quarters and 0.788 [0.771, 0.836] at 16–20 quarters. That check covered 24 markets with medium or long histories. Per-market AR uses a pooled fit when it has too few local labels. [Comparison on fitted forecasts](results/long_hpi/validation/fitted_only_ratio_ci.csv) · [Error by market and horizon](results/long_hpi/validation/metrics_by_market.csv)

![Median RMSE relative to no-change by forecast horizon](results/long_hpi/validation/rel_rmse_by_horizon.png)

The chart shows the shared Transformer and AR baselines relative to no-change. The shaded ranges mark 3–6 and 16–20 quarters. Absolute error grows with the horizon. For the shared Transformer, median RMSE rose from 0.017 at one quarter to 0.235 at 20 quarters, measured in cumulative log growth.

### Training on more markets

We started with five markets and added more by rotating through regions. All four runs were scored on the same five markets and dates. The table compares RMSE with the five-market run. A ratio below 1 means lower error, and brackets show 95% confidence intervals.

| Markets on the list | Markets used per fold | 3–6 quarters | 16–20 quarters |
|---|---:|---:|---:|
| 5 | 3–5 | Reference | Reference |
| 10 | 7–10 | 0.922 [0.797, 0.978] | 0.925 [0.798, 1.080] |
| 20 | 15–20 | 1.002 [0.862, 1.039] | 0.905 [0.679, 1.136] |
| 41 | 24–38 | 0.912 [0.802, 0.972] | 0.818 [0.651, 0.982] |

The 41-market run cut error by about **9% at 3–6 quarters** and **18% at 16–20 quarters**. The middle runs were mixed. We tried one order, so which markets we added could matter too. [Market selection](docs/market_selection.md) · [Full results](results/market_scale/validation)

### Forecasting a market left out of training

We left out Australia, Chile, Taiwan and Portugal one at a time. At forecast time, the model used only the local HPI history available by that date. The table compares each held-out model with the shared Transformer trained with that market included. A ratio below 1 favors the held-out model.

| Held-out market | Forecast dates | 3–6 quarters | 16–20 quarters | Against AR and no-change |
|---|---:|---:|---:|---|
| [Australia](results/market_holdout/australia/validation) | 40 | 1.008 [0.988, 1.040] | 1.013 [0.980, 1.051] | All intervals against AR and no-change include 1 in both ranges. |
| [Chile](results/market_holdout/chile/validation) | 24 | 1.021 [0.986, 1.051] | 0.901 [0.870, 0.951] | Worse than both at 3–6 quarters, better at 16–20. |
| [Taiwan](results/market_holdout/taiwan_national/validation) | 28 | 1.011 [1.005, 1.024] | 0.990 [0.979, 1.007] | Close to no-change in both ranges. Worse than AR at 16–20 quarters. |
| [Portugal](results/market_holdout/portugal/validation) | 40 | 0.956 [0.919, 0.978] | 1.009 [0.991, 1.030] | Worse than local AR and no-change in both ranges. |

Leaving a market out usually changed RMSE by only a few percent. Chile had the biggest change, with about 10% lower RMSE at 16–20 quarters. Its 95% interval includes 1 when we use 8-quarter blocks instead of 20 ([0.870, 1.049]), so the improvement is uncertain.

Chile and Taiwan used pooled AR for every 16–20-quarter forecast and for about 71% and 61% of their 3–6-quarter forecasts, respectively. For Portugal, both Transformers used the same single-thread setting. The other markets use the original shared run for comparison. In Chile and Taiwan, the first scored fold had identical training data across runs. RMSE still differed by up to about 6.5% for Chile and less than 0.02% for Taiwan across the two ranges.

## Limits of the evaluation

The results cover one historical validation period, including the 2008 crisis. The proposed test starts in 2015 and overlaps years examined in the Taiwan pilot. The longer forecasts show how the models compare, while their usefulness for real decisions depends on how much error those decisions can tolerate. The four holdout markets give examples of transfer across markets.

We still need to run per-market ARIMA, try shared pretraining followed by fine-tuning for each market, repeat the training pool comparison with the other two market orders, and run the final test.

## Run locally

Use Python 3.11, from the repository root:

```bash
conda create -n hpf python=3.11 -y
conda activate hpf
pip install -e ".[dev,dl]"
pytest -q
python scripts/build_panel.py
```

Multi-market validation:

```bash
C=configs/experiment_long.toml
python scripts/run_backtest.py --config $C --models no_change ar ar_pooled --pool national --experiment long_hpi
python scripts/run_backtest.py --config $C --models transformer transformer_per_market --pool national --experiment long_hpi
python scripts/evaluate_markets.py results/long_hpi/validation/preds_*.csv --config $C --reference no_change__national \
    --compare transformer__national transformer_per_market__national \
    --compare transformer__national ar__national \
    --compare transformer__national ar_pooled__national \
    --compare transformer__national no_change__national
```

This writes the error tables, confidence intervals and a plot if Matplotlib is installed. The command below repeats the check using the per-market fit diagnostics.

```bash
python scripts/fitted_only_check.py
```

See the Taiwan pilot notes for the earlier experiments and their commands. [Taiwan pilot](docs/taiwan_pilot.md)

### Training pool comparison

```bash
OMP_NUM_THREADS=1 python scripts/run_market_scale.py
python scripts/evaluate_markets.py results/market_scale/validation/preds_*.csv \
    --reference transformer__scale_5 \
    --compare transformer__scale_10 transformer__scale_5 \
    --compare transformer__scale_20 transformer__scale_5 \
    --compare transformer__scale_41 transformer__scale_5
```

### Holdout experiments

The holdout run trains on the other national markets and scores only the chosen market. The model uses the chosen market's HPI history at forecast time. The script takes the matching reference forecasts from the long-horizon validation run and checks that all rows line up. For Chile, Taiwan or Portugal, use the matching market ID and results folder. For the Portugal comparison, first rerun the national Transformer, AR and no-change with `OMP_NUM_THREADS=1` in a separate experiment folder. Pass that run's validation directory to `--reference-dir` when running the holdout.

```bash
OMP_NUM_THREADS=1 python scripts/run_market_holdout.py --market Australia
python scripts/evaluate_markets.py results/market_holdout/australia/validation/preds_*.csv \
    --reference no_change__national \
    --compare transformer__holdout no_change__national \
    --compare transformer__holdout ar__national \
    --compare transformer__holdout transformer__national
```

## Related work

The repository also includes a prototype for using a local language model to read Taiwan policy announcements. The forecasting results above use house price data. [Policy data and setup](data/raw/policy/README.md) · [Backtest notes](docs/backtest_framework.md) · [Experiment settings](docs/experiment_contract.md)
