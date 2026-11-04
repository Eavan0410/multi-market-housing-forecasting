# Multi-Market Housing Price Forecasting

This is a course project on housing price forecasting. We are testing whether one Transformer can predict several markets, how it compares with AR models, and how far ahead it can make useful forecasts.

We started with Taiwan and then added a multi-market experiment with forecasts from 1 to 20 quarters ahead. The model is a small Transformer trained on our data. The main results so far are below.

## Progress

Updated November 3, 2026. “Validation done” means predictions and evaluation tables are saved; final testing is still pending.

| Experiment or task | Status |
|---|---|
| Multi-market HPI-only Transformer, h=1–20 | Validation done, five seeds |
| Multi-market no-change, per-market AR and pooled AR | Validation done |
| Errors by market, horizon and history length | Tables saved |
| Separate Transformer for each market | Not run yet |
| Confidence intervals for multi-market model differences | Not done yet |
| Taiwan-only, joint and pretrain + fine-tune Transformers, h=1–4 | Validation done, five seeds each |
| Head-only, shorter and lower-learning-rate fine-tuning | Validation done |
| Overseas-only Transformer and joint model with market embeddings | Validation done |
| Taiwan HPI-only, wider pool and four-country comparison | Validation done for joint Transformer and pooled AR |
| Taiwan HPI-only, Taiwan-only Transformer and pretrain + fine-tune | Not run yet |
| Data availability matrix | Draft saved; timing assumptions still need checking |
| LLM policy index | Code available; real-data run pending |
| Final settings, test runs and report | Still to do |

## Multi-market results

These validation runs use forecast dates in 2000Q1–2009Q4. Each input window contains eight quarters of real HPI log growth. The target is cumulative growth over the next h quarters:

```text
y(t, h) = log(HPI_real[t + h] / HPI_real[t])
```

Predictions cover 41 national markets. We report the 38 with at least eight forecast dates each; their sample sizes range from 8 to 40. The same market/date pairs are used at every horizon. City indices are left out.

The Transformer uses five seeds. We calculate each seed's errors first, then average them. AR and no-change use one seed. Per-market AR uses pooled coefficients when a market has too few training labels.

The table shows the median error ratio across markets. For example, 0.95 means the median ratio is 5% below the comparison model. Each ratio is calculated within the same market and horizon before taking the median. Lower is better.

| Horizon (quarters) | Transformer / no-change | Transformer / per-market AR | Transformer / pooled AR | Markets beating pooled AR |
|---|---:|---:|---:|---:|
| 1 | 0.782 | 1.009 | 1.015 | 11/38 |
| 4 | 0.837 | 0.965 | 0.996 | 20/38 |
| 6 | 0.867 | 0.924 | 0.983 | 22/38 |
| 12 | 0.862 | 0.875 | 0.951 | 24/38 |
| 16 | 0.878 | 0.860 | 0.936 | 24/38 |
| 20 | 0.894 | 0.866 | 0.948 | 25/38 |

The short-term results are close to pooled AR. At h=20, Transformer has lower RMSE in 25 of 38 markets, and the median ratio is about 5.2% below pooled AR. Its advantage over per-market AR is larger. This suggests the shared Transformer may be more useful further ahead, although pooling data also helps the simpler AR model.

It does not win everywhere. At h=20, its RMSE is about 1.21 times pooled AR in Taiwan, 1.47 times in Germany and 1.60 times in Mexico. These runs use each market's earlier history, so they do not test forecasting a completely unseen market.

![Multi-market validation results](results/long_hpi/validation/market_comparison.png)

Left: median RMSE relative to no-change. Right: median Transformer/AR RMSE ratio. Values below 1 favour Transformer in the right panel. The shaded ranges are 3–6 and 16–20 quarters, motivated by buying/selling and construction decisions.

These are early results. We have not calculated confidence intervals for the multi-market differences. Long-horizon targets overlap, and some markets have few forecast dates. A lower relative error also does not mean the absolute error is small enough for a real decision. The next checks are uncertainty, results in other periods, and a comparison with separate Transformers.

The proposed test starts in 2015, which overlaps Taiwan periods we have already examined. We still need to review that split before calling it an independent final test.

Results: [errors by market](results/long_hpi/validation/metrics_by_market.csv), [relative errors](results/long_hpi/validation/relative_by_market.csv), [horizon summary](results/long_hpi/validation/summary_by_horizon.csv), and [history groups](results/long_hpi/validation/summary_by_history.csv). History-group differences are descriptive; they do not show that history length caused the gains.

## Earlier Taiwan experiments

These runs evaluate Taiwan's national real HPI at 24 forecast dates in 2012Q1–2017Q4, with horizons 1–4. They use a different period from the multi-market study, so their scores should be read separately.

### Four features

Inputs are real HPI log growth, long-term bond yields, real GDP quarterly growth and the four-quarter change in household credit / GDP. The overseas markets are Australia, Japan, South Korea and the United States. Errors are in cumulative log-growth units. Lower is better.

| Model / training data | RMSE h=1 | h=2 | h=3 | h=4 | MAE h=4 |
|---|---:|---:|---:|---:|---:|
| No-change | 0.02313 | 0.03780 | 0.05059 | 0.06297 | 0.04404 |
| Taiwan AR | 0.02440 | 0.04553 | 0.06639 | 0.08593 | 0.07524 |
| Overseas AR | 0.02260 | 0.03791 | 0.04614 | 0.05270 | 0.03960 |
| Pooled AR: overseas + Taiwan | 0.02120 | 0.03476 | 0.04304 | 0.05140 | 0.03953 |
| Transformer: Taiwan-only | 0.03042 | 0.05256 | 0.07207 | 0.08314 | 0.07242 |
| Transformer: joint | 0.02102 | 0.03596 | 0.04831 | 0.05846 | 0.04576 |
| Transformer: pretrain + fine-tune | 0.02373 | 0.04295 | 0.05957 | 0.07457 | 0.06284 |
| Fine-tune: head-only | 0.02185 | 0.03775 | 0.05044 | 0.06096 | 0.04794 |
| Fine-tune: at most 20 epochs | 0.02196 | 0.03784 | 0.05032 | 0.06095 | 0.04951 |
| Fine-tune: lower learning rate | 0.02175 | 0.03767 | 0.05025 | 0.06064 | 0.04749 |
| Transformer: zero-shot | 0.02147 | 0.03750 | 0.05006 | 0.06057 | 0.04412 |
| Joint + market embeddings | 0.02086 | 0.03657 | 0.04959 | 0.06072 | 0.04869 |

Joint training has lower RMSE than Taiwan-only training and default fine-tuning at every horizon. The three fine-tuning variants improve on default fine-tuning, but still trail joint training. Market embeddings help slightly at h=1 and do not improve h=2–4.

Pooled AR has lower RMSE than the joint Transformer at h=2–4. At h=4, the Transformer-minus-pooled-AR RMSE difference is +0.00707, with a 95% block-bootstrap interval of [-0.00451, 0.01942]. The interval includes zero, so this comparison is inconclusive.

Results: [metrics](results/fundamentals/validation/metrics.csv), [confidence intervals](results/fundamentals/validation/metric_ci.csv), and [comparisons with pooled AR](results/fundamentals/validation/bootstrap_vs_ar_pooled__overseas_plus_target.csv).

### HPI-only and four-country comparison

Using HPI growth alone lets us include more AnaHPI countries. The four-country comparison uses the same four overseas markets as above, plus Taiwan, to help separate the effects of features and country coverage.

| Model / training data | RMSE h=1 | h=2 | h=3 | h=4 | MAE h=4 |
|---|---:|---:|---:|---:|---:|
| No-change | 0.02313 | 0.03780 | 0.05059 | 0.06297 | 0.04404 |
| Taiwan AR | 0.02440 | 0.04553 | 0.06639 | 0.08593 | 0.07524 |
| Pooled AR: wider pool + Taiwan | 0.02173 | 0.03575 | 0.04417 | 0.05136 | 0.03929 |
| Joint Transformer: wider pool + Taiwan | 0.02273 | 0.03807 | 0.04689 | 0.05230 | 0.03926 |
| Pooled AR: four countries + Taiwan | 0.02138 | 0.03551 | 0.04410 | 0.05179 | 0.03996 |
| Joint Transformer: four countries + Taiwan | 0.02080 | 0.03604 | 0.04833 | 0.05611 | 0.04285 |

At h=4, the wider-pool Transformer is close to pooled AR: RMSE 0.05230 versus 0.05136. Their RMSE difference is +0.00094, with a 95% interval of [-0.00264, 0.00525]. This does not show an improvement over pooled AR.

With the same four overseas markets, the HPI-only joint model has lower RMSE than the four-feature model at h=1 and h=4, and nearly the same RMSE at h=2–3. Within HPI-only runs, the wider pool helps the Transformer at h=3–4 but hurts at h=1–2. More countries do not help at every horizon. These feature and country-pool differences have not been tested directly with paired confidence intervals.

Results: [metrics](results/hpi_only/validation/metrics.csv), [confidence intervals](results/hpi_only/validation/metric_ci.csv), and [comparisons with wider-pool AR](results/hpi_only/validation/bootstrap_vs_ar_pooled__overseas_plus_target.csv).

## Data and backtest notes

Raw and processed source tables are in `data/raw/` and `data/processed/`. Taiwan HPI/CPI and global AnaHPI were copied from the Spring 2026 dataset. Generated panels go in `data/interim/` and can be rebuilt locally.

Taiwan HPI is currently treated as nominal and deflated with CPI; its price basis still needs confirmation. Taiwan credit / GDP uses bank consumer loans divided by the latest four quarters of nominal GDP. Its coverage is narrower than the overseas household-credit measure. Publication lags are provisional; the draft evidence is in [availability_matrix.csv](results/availability_matrix.csv). The data contain later revisions, rather than the versions originally released.

Both studies use an expanding training window and refit every four forecast dates. All countries share a calendar cutoff. Unpublished training labels are hidden, and input scaling is fitted on the training fold. The long-horizon setup keeps a four-quarter gap and masks unavailable longer-horizon labels separately.

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
python scripts/run_backtest.py --config $C --models transformer --pool national --experiment long_hpi
python scripts/evaluate_markets.py results/long_hpi/validation/preds_*.csv --config $C --reference no_change__national
```

This writes market-level tables, summaries and a relative-RMSE plot when Matplotlib is installed. The two-panel comparison above also includes direct Transformer/AR ratios calculated from `relative_by_market.csv`.

Taiwan baselines and the three main Transformer settings:

```bash
python scripts/run_backtest.py --models no_change ar --experiment fundamentals
python scripts/run_backtest.py --models ar --pool overseas --experiment fundamentals
python scripts/run_backtest.py --models ar_pooled --pool overseas_plus_target --experiment fundamentals
python scripts/run_backtest.py --models transformer --pool target_only --experiment fundamentals
python scripts/run_backtest.py --models transformer --pool overseas_plus_target --experiment fundamentals
python scripts/run_backtest.py --models transformer_ft --pool overseas_plus_target --experiment fundamentals
python scripts/evaluate.py results/fundamentals/validation/preds_*.csv --reference no_change
```

Use `ar_pooled` for a shared AR fit. Plain `ar` fits each market separately where enough labels are available. Transformer training selects the epoch count on a time split inside each fold, then retrains on that fold's full training set.

HPI-only country comparison:

```bash
python scripts/run_backtest.py --models no_change ar --features hpi_qoq --experiment hpi_only
python scripts/run_backtest.py --models transformer ar_pooled --pool overseas_plus_target --features hpi_qoq --experiment hpi_only
python scripts/run_backtest.py --models transformer ar_pooled --pool overseas_plus_target --features hpi_qoq --experiment hpi_only --countries Australia Japan "South Korea" "United States"
python scripts/evaluate.py results/hpi_only/validation/preds_*.csv --reference ar_pooled__overseas_plus_target
```

## Optional policy index

The policy module is an extra experiment using a local LLM to extract tightening/easing signals from Taiwan announcements. The code and manual-timeline template are available, but announcements still need collecting and there are no saved real-model results. Forecasting experiments come first.

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
