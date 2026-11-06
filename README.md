# Multi-Market Housing Price Forecasting

We started with a question about Taiwan. Could one model also forecast house prices in many other markets? We use a small Transformer to explore that idea.

1. Can one model work well across different housing markets?
2. How does it compare with traditional methods and with training a separate Transformer for each market?
3. How does that comparison change as we forecast further ahead?

We test forecasts from 1 to 20 quarters ahead. We pay particular attention to 3–6 quarters for buying or selling and 16–20 quarters for construction. We also test what happens when we add training markets or leave a market out. [Early Taiwan results](docs/taiwan_pilot.md)

## Progress

Updated November 5, 2026. The results below come from validation. The final test is still ahead.

| Question | Experiment or task | Status |
|---|---|---|
| Market coverage | One shared Transformer, 1–20 quarters ahead | Validation done. 41 markets forecast, 38 reported. |
| One-task-one-model | Per-market AR with pooled fallback; no-change; pooled AR as a shared linear reference | Validation done; includes pooled fallback |
| One-task-one-model | One ARIMA per market | Template ready; fitting and forecasting code still to write |
| One-task-one-model | One Transformer per market (from scratch) | Validation done (10 folds, 5 seeds) |
| Transfer | Shared pretraining + per-market fine-tuning | Not run yet |
| Accuracy | Paired confidence intervals for the shared Transformer vs per-market Transformer, per-market AR, pooled AR and no-change | Done on validation. [Per-market Transformer comparison](results/long_hpi/validation/ratio_ci_transformer__national_vs_transformer_per_market__national.csv) |
| More training markets (#9) | Train on 5, 10, 20 and 41 eligible markets. Score the same five markets each time. | First order complete (region rotation). |
| Unseen markets (#10) | Leave one market out of training, then forecast it | All four planned markets complete. |
| Set-up | Common forecast date based on the slowest configured HPI release | Done. Multi-market validation rerun |
| Background | Taiwan pilot (2012–2017, 1–4 quarters) | Done; see [docs/taiwan_pilot.md](docs/taiwan_pilot.md) |
| Optional | Taiwan LLM policy index | Code ready; not yet run on real announcements |

## Multi-market results

**Set-up.** The target is cumulative real house price growth over the next h quarters, for h = 1–20:

```text
y(t, h) = log(HPI_real[t + h] / HPI_real[t])
```

The model uses the last eight quarters of each market's real HPI growth. Validation covers 2000Q1 to 2009Q4, with a refit every four quarters. All markets use the same forecast date for a given quarter, set to quarter-end plus the slowest configured HPI release lag (100 days in this run). Each market keeps its own release lag for training labels.

These multi-market runs score only 2000–2009 forecast dates. Forecasts reach 20 quarters ahead, so their labels use prices up to 2014. The proposed multi-market test starts in 2015 and has not been run, but it overlaps dates examined in the earlier Taiwan pilot; it is not a fully untouched period for the project.

We have predictions for 41 national markets. The results tables use the 38 with at least eight forecast dates. As more data become available, the number of markets contributing training data grows from 24 at the first refit to 38 at the last.

We measure forecast error with RMSE and average it over five seeds for each Transformer. To compare two models, we divide their RMSEs in each market and take the median across markets. If the ratio is below 1, the first model has lower error. The last two columns show RMSE itself, in cumulative log-growth units.

Per-market AR uses pooled AR when it has too few labels to fit a separate model. The per-market Transformer uses the same architecture as the shared model and fits and scales each market separately. It falls back to no-change when it has too little training history. About 23% of its scored forecasts in the 38 reported markets used this fallback. [Fit and fallback counts](results/long_hpi/validation/diag_transformer_per_market__national.csv)

| Quarters ahead | Transformer / per-market AR | Markets beating per-market AR | Transformer / no-change | Per-market AR / no-change | Transformer / pooled AR | Median RMSE (Transformer) | Median RMSE (per-market AR) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.997 | 20/38 | 0.766 | 0.785 | 1.012 | 0.017 | 0.018 |
| 4 | 0.967 | 25/38 | 0.830 | 0.868 | 0.984 | 0.053 | 0.066 |
| 5 | 0.947 | 25/38 | 0.851 | 0.891 | 0.976 | 0.065 | 0.083 |
| 6 | 0.933 | 26/38 | 0.863 | 0.902 | 0.978 | 0.077 | 0.097 |
| 8 | 0.912 | 29/38 | 0.875 | 0.956 | 0.983 | 0.104 | 0.132 |
| 12 | 0.861 | 29/38 | 0.855 | 1.032 | 0.957 | 0.152 | 0.200 |
| 16 | 0.854 | 28/38 | 0.874 | 1.069 | 0.954 | 0.191 | 0.263 |
| 20 | 0.863 | 27/38 | 0.894 | 1.093 | 0.958 | 0.235 | 0.304 |

The next table compares the shared Transformer with each baseline over two ranges. We combine the errors within each range before calculating RMSE. The brackets show 95% confidence intervals from 2,000 bootstrap draws. Each draw uses the same 20-quarter blocks for both models and all markets.

| Baseline | 3–6 quarters | 16–20 quarters |
|---|---:|---:|
| Per-market Transformer | 0.895 [0.796, 0.930] | 0.804 [0.791, 0.858] |
| Per-market AR | 0.943 [0.883, 0.977] | 0.869 [0.795, 0.949] |
| Pooled AR | 0.978 [0.918, 1.058] | 0.953 [0.917, 1.005] |
| No-change | 0.849 [0.649, 0.979] | 0.889 [0.831, 1.027] |

- **Per-market Transformer.** The shared model has about 10% lower RMSE over 3–6 quarters and 20% lower RMSE over 16–20 quarters, based on the median ratios. The 95% interval stays below 1 at every horizon from 1 to 20. This comparison includes the no-change forecasts where a separate model could not be fitted.
- **Per-market AR.** The two models are about even one quarter ahead. From 5 through 20 quarters, the Transformer's ratio stays below 1 across the full confidence interval at each horizon. Over 16–20 quarters, its median ratio is 0.869, or about 13% lower RMSE.
- **Pooled AR.** The Transformer has about 5% lower RMSE over 16–20 quarters. The interval just crosses 1 (0.917 to 1.005), so the two models could be about even.
- **No-change.** The Transformer has lower RMSE over 3–6 quarters, with the full interval below 1. Its ratio is still below 1 over 16–20 quarters, but the interval crosses 1. Per-market AR falls behind no-change from 10 quarters onward and reaches a ratio of 1.093 at 20 quarters.

The shared model also had lower error when both forecasts came from a trained Transformer. We kept dates where the per-market model fitted successfully in all five seeds. Only 24 markets had at least eight such dates, and all had long or medium HPI histories. The median ratio was **0.816 [0.796, 0.899]** for 3–6 quarters and **0.788 [0.771, 0.836]** for 16–20. The same 24 markets gave the same median ratios with the fallback forecasts included. [Comparison by sample](results/long_hpi/validation/fitted_only_ratio_ci.csv) · [Coverage by HPI history](results/long_hpi/validation/fitted_only_counts.csv)

![Median RMSE relative to no-change by forecast horizon](results/long_hpi/validation/rel_rmse_by_horizon.png)

This plot shows the shared Transformer and AR baselines relative to no-change. The shaded ranges show 3–6 and 16–20 quarters. Forecasts get less accurate further out. We still need to check whether the errors at 16–20 quarters are small enough for practical use. So far we have looked at one validation period, which includes the 2008 crisis.

See the [market errors](results/long_hpi/validation/metrics_by_market.csv), [relative errors](results/long_hpi/validation/relative_by_market.csv), [horizon summary](results/long_hpi/validation/summary_by_horizon.csv) and [history groups](results/long_hpi/validation/summary_by_history.csv). The full confidence intervals are in the comparison files for [per-market Transformer](results/long_hpi/validation/ratio_ci_transformer__national_vs_transformer_per_market__national.csv), [per-market AR](results/long_hpi/validation/ratio_ci_transformer__national_vs_ar__national.csv), [pooled AR](results/long_hpi/validation/ratio_ci_transformer__national_vs_ar_pooled__national.csv), [no-change](results/long_hpi/validation/ratio_ci_transformer__national_vs_no_change__national.csv).

### #9. Does adding markets help?

We trained on 5, 10, 20 and 41 markets in the planned region order. Every run was scored on the same five markets and dates. The table compares RMSE with the five-market run. A ratio below 1 means lower error, and brackets show 95% confidence intervals.

| Markets on the list | Markets used per fold | 3–6 quarters | 16–20 quarters |
|---|---:|---:|---:|
| 5 | 3–5 | Reference | Reference |
| 10 | 7–10 | 0.922 [0.797, 0.978] | 0.925 [0.798, 1.080] |
| 20 | 15–20 | 1.002 [0.862, 1.039] | 0.905 [0.679, 1.136] |
| 41 | 24–38 | 0.912 [0.802, 0.972] | 0.818 [0.651, 0.982] |

The 41-market run cut error by about **9% at 3–6 quarters** and **18% at 16–20 quarters**. The middle runs were mixed. We tried one order, so which markets we added could matter too. [Market selection](docs/market_selection.md) · [Full results](results/market_scale/validation)

### #10. Can the model forecast a market it left out?

We left each market out of training and still gave the model its recent HPI history at forecast time. The table compares each held-out model with the shared Transformer trained with that market included. A ratio below 1 favors the held-out model.

| Held-out market | Forecast dates | 3–6 quarters | 16–20 quarters | Against AR and no-change |
|---|---:|---:|---:|---|
| [Australia](results/market_holdout/australia/validation) | 40 | 1.008 [0.988, 1.040] | 1.013 [0.980, 1.051] | All intervals against AR and no-change include 1 in both ranges. |
| [Chile](results/market_holdout/chile/validation) | 24 | 1.021 [0.986, 1.051] | 0.901 [0.870, 0.951] | Worse than both at 3–6 quarters, better at 16–20. |
| [Taiwan](results/market_holdout/taiwan_national/validation) | 28 | 1.011 [1.005, 1.024] | 0.990 [0.979, 1.007] | Close to no-change in both ranges. Worse than AR at 16–20 quarters. |
| [Portugal](results/market_holdout/portugal/validation) | 40 | 0.956 [0.919, 0.978] | 1.009 [0.991, 1.030] | Worse than local AR and no-change in both ranges. |

Leaving a market out usually changed RMSE by only a few percent. Chile had the biggest change, with about 10% lower RMSE at 16–20 quarters. Its 95% interval includes 1 when we use 8-quarter blocks instead of 20 ([0.870, 1.049]), so the improvement is uncertain.

Chile and Taiwan used pooled AR for every 16–20-quarter forecast and for about 71% and 61% of their 3–6-quarter forecasts, respectively. Both Portugal runs used the same single-thread setting. Australia, Chile and Taiwan use the original shared run as their reference. In the first scored fold for Chile and Taiwan, leaving the market out removed zero training rows. The two runs still differed in RMSE by about 6.5% and 3.7% for Chile across the two ranges, and by less than 0.02% for Taiwan.

## Next steps

1. Run ARIMA for each market and report where it uses a fallback. Run shared pretraining with per-market fine-tuning as a separate comparison.
2. For #9, try the other two market orders and see whether the pattern holds. [Market selection](docs/market_selection.md)

## Data and backtest notes

Raw and processed source tables are in `data/raw/` and `data/processed/`. Taiwan HPI/CPI and global AnaHPI were copied from the Spring 2026 dataset. Generated panels go in `data/interim/` and can be rebuilt locally.

Taiwan HPI is currently treated as nominal and deflated with CPI; its price basis still needs confirmation. Taiwan credit / GDP uses bank consumer loans divided by the latest four quarters of nominal GDP. Its coverage is narrower than the overseas household-credit measure. Publication lags are provisional; the draft evidence is in [availability_matrix.csv](results/availability_matrix.csv). The data contain later revisions, rather than the versions originally released.

Both studies use an expanding training window and refit every four forecast dates. The multi-market run uses the first forecast date in each fold as its training cutoff and keeps a four-quarter gap between training and validation origins. Input scaling is fitted on the training fold. The Taiwan pilot uses its original release timing.

The Taiwan pilot uses 2012–2017 validation origins, a 2018 buffer and a test starting in 2019. Its confidence intervals use four-quarter circular bootstrap blocks. The multi-market setup uses 2000–2009 validation origins and a proposed test from 2015. Test runs stay locked by default.

Keep each prediction CSV beside its `.meta.json` file. The metadata record features, dates, seeds and data fingerprints so the evaluator can check that runs are comparable.

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

Other multi-market tools:

```bash
python scripts/build_market_list.py   # docs/market_list.csv and per-fold training samples
pip install statsmodels               # then fill the TODO(B) parts of src/housing_forecast/models/arima.py
python scripts/run_backtest.py --config $C --models arima --pool national --experiment long_hpi   # also writes diag_arima__national.csv
```

The earlier Taiwan runs and their commands are in [the Taiwan pilot notes](docs/taiwan_pilot.md). Plain `ar` uses per-market fits with pooled fallback; `ar_pooled` uses one shared fit. Transformer training selects the epoch count on a time split inside each fold, then retrains on that fold's full training set.

### Market-scale experiment (#9)

Each run uses the same five evaluation markets. The training lists grow from 5 to 10, 20 and 41 markets in the region-rotation order. The coverage table shows how many markets and rows actually enter training in each fold. [Market selection](docs/market_selection.md)

```bash
OMP_NUM_THREADS=1 python scripts/run_market_scale.py
python scripts/evaluate_markets.py results/market_scale/validation/preds_*.csv \
    --reference transformer__scale_5 \
    --compare transformer__scale_10 transformer__scale_5 \
    --compare transformer__scale_20 transformer__scale_5 \
    --compare transformer__scale_41 transformer__scale_5
```

### Leave one market out (#10)

The holdout run trains on the other national markets and scores only the chosen market. The model uses the chosen market's HPI history at forecast time. The script takes the matching reference forecasts from the long-horizon validation run and checks that all rows line up. For Chile, Taiwan or Portugal, use the matching market ID and results folder. For the Portugal comparison, first rerun the national Transformer, AR and no-change with `OMP_NUM_THREADS=1` in a separate experiment folder. Pass that run's validation directory to `--reference-dir` when running the holdout.

```bash
OMP_NUM_THREADS=1 python scripts/run_market_holdout.py --market Australia
python scripts/evaluate_markets.py results/market_holdout/australia/validation/preds_*.csv \
    --reference no_change__national \
    --compare transformer__holdout no_change__national \
    --compare transformer__holdout ar__national \
    --compare transformer__holdout transformer__national
```

## Optional policy index

As an optional extension, we plan to use a local LLM to identify tightening and easing signals in Taiwan policy announcements. The code and a template for a manually checked timeline are ready. We still need to collect the announcements and run the model. The forecasting experiments take priority.

Settings are in [policy.toml](configs/policy.toml), with input formats in [the policy data notes](data/raw/policy/README.md). Announcements are grouped by publication quarter; availability at each forecast date still needs checking before the index can be used as an input.

```bash
python scripts/policy_pipeline.py --backend mock extract
# Requires vLLM and a GPU, after collecting the input data
python scripts/policy_pipeline.py --backend vllm extract --limit 20
python scripts/policy_pipeline.py --backend vllm index
python scripts/policy_pipeline.py --backend vllm eval
python scripts/policy_pipeline.py --backend vllm spotcheck-export
```

## Code and notes

`src/housing_forecast/` contains data readers, backtesting, models, evaluation and policy extraction. Commands are in `scripts/`, checks in `tests/`, and outputs in `results/<experiment>/<phase>/`.

See [experiment settings](docs/experiment_contract.md), [backtest notes](docs/backtest_framework.md) and [data formats](docs/design_notes.md) for more detail. The original course milestones are November 12 for baselines, November 19 for Transformer comparisons, November 26 for analysis and December 10 for the report. Initial validation runs are done; final checks and the report are still in progress.
