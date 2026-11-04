# Multi-Market Housing Price Forecasting

This project compares a Taiwan-only Transformer with joint multi-market training and overseas pretraining followed by fine-tuning on Taiwan.

The data pipeline, backtest and evaluator are working, with no-change, direct AR, transfer AR and pooled AR baselines available. The Transformer comparisons are still to come. Extracting policy signals with an LLM is an optional extension after the forecasting experiments.

## Data and target

The main dataset covers Australia, Japan, South Korea, the United States and Taiwan at quarterly frequency. The target is Taiwan's national real house price index. Each forecast predicts cumulative log growth over the next 1–4 quarters:

```text
y(t, h) = log(HPI_real[t + h] / HPI_real[t])
```

The current feature set is:

- Real HPI quarterly log growth
- Long-term government bond yield, averaged over a complete quarter
- Seasonally adjusted real GDP quarterly growth
- Four-quarter change in household credit / GDP

Taiwan's credit measure uses bank consumer loans divided by the sum of the latest four quarters of nominal GDP. It is a substitute for the overseas household-credit series; taking changes does not remove that difference in coverage. Taiwan HPI is currently treated as nominal and deflated with CPI, pending confirmation of its basis.

An HPI-only setting can also use the additional countries in the OECD AnaHPI file. Those countries drop out of the current four-feature setting because they lack the other inputs. City-level Taiwan HPI is retained for separate analysis.

## Run locally

Use Python 3.11 and run these commands from the repository root:

```bash
conda create -n hpf python=3.11 -y
conda activate hpf
pip install -e ".[dev]"
pytest -q

python scripts/build_panel.py
python scripts/run_backtest.py --models no_change ar --experiment fundamentals
python scripts/evaluate.py results/fundamentals/validation/preds_*.csv --reference no_change
```

To try the HPI-only setting:

```bash
python scripts/run_backtest.py --features hpi_qoq --experiment hpi_only
python scripts/evaluate.py results/hpi_only/validation/preds_*.csv --reference no_change
```

There are two ways to use the overseas data with an AR model:

```bash
# Transfer AR: trained on the four overseas markets only, then used to forecast Taiwan
python scripts/run_backtest.py --models ar --pool overseas --experiment fundamentals

# Pooled AR: one AR fitted on the overseas markets and Taiwan together
python scripts/run_backtest.py --models ar_pooled --pool overseas_plus_target --experiment fundamentals
```

These add `preds_ar__overseas.csv` and `preds_ar_pooled__overseas_plus_target.csv` to the same experiment folder. Use `ar_pooled` for the pooled run: plain `ar` fits each series separately, so `--models ar --pool overseas_plus_target` gives the same Taiwan forecasts as the Taiwan-only AR. Neither is the planned joint Transformer.

## Transformer runs

The Transformer models need PyTorch (`pip install torch`). They use the same backtest, features and output format as the baselines:

```bash
# Taiwan-only
python scripts/run_backtest.py --models transformer --pool target_only --experiment fundamentals
# Joint training on overseas markets and Taiwan
python scripts/run_backtest.py --models transformer --pool overseas_plus_target --experiment fundamentals
# Pretrain on overseas markets, then fine-tune on Taiwan
python scripts/run_backtest.py --models transformer_ft --pool overseas_plus_target --experiment fundamentals
```

Each run trains five seeds. The model picks its number of epochs on the last two years of each training fold and is then retrained on the whole fold. Compare the joint and fine-tuned runs with `ar_pooled__overseas_plus_target`, which uses the same training pool.

Each prediction CSV has a `.meta.json` file recording its features, training pool, cutoffs and input-data fingerprints. The evaluator rejects incompatible manifests and checks that models and seeds cover the same forecast rows. Keep the metadata beside the CSVs. Files without metadata produce a warning because their settings cannot be verified.

## Backtest

Validation origins run from 2012Q1 to 2017Q4. Origins in 2018 form a buffer, and the test period starts in 2019Q1. The test period stays locked until the final configuration freeze; use validation for model and feature choices.

Training uses an expanding window, with one refit every four forecast origins. The last training origin is four quarters before the first evaluation origin. All countries share the same calendar cutoff, unpublished training labels are masked, and scaling is fitted within each training fold.

The evaluator reports RMSE, MAE and direction accuracy by horizon. Metrics are calculated for each seed and then averaged. Confidence intervals use paired circular blocks of four origin quarters, keeping all horizons and seeds together in each draw. The bootstrap holds the fitted seeds fixed; across-seed standard deviations describe training variability separately.

Publication lags are still placeholders until the availability matrix is confirmed. The data are revised series rather than real-time vintages, so the backtest does not reconstruct exactly what was known historically.

## Files

```text
configs/experiment.toml    Features, horizons, periods, lags and seeds
data/raw/                  Overseas and Taiwan source tables
data/processed/           Four-country processed tables
data/interim/             Generated panels, ignored by Git
src/housing_forecast/
  data/                    Readers and quarterly transformations
  backtesting/             Windows, folds, scaling, runner and run metadata
  models/                  No-change and AR baselines
  evaluation/              Metrics and block bootstrap
  policy/                  Placeholder for policy extraction
scripts/                   Build, backtest and evaluation commands
tests/                     Data, leakage, alignment and evaluation checks
results/                   Outputs grouped by experiment and phase
```

Source tables in `data/raw/` and `data/processed/` are tracked. Generated panels go to `data/interim/`; checkpoints and large training artifacts are ignored. The Taiwan HPI/CPI and global AnaHPI tables were copied from the Spring 2026 dataset, so the build no longer needs a sibling data folder.

See the [experiment contract](docs/experiment_contract.md) for the forecast definitions, [framework notes](docs/backtest_framework.md) for adding models, and [design notes](docs/design_notes.md) for the data interfaces and merge rules. Runtime settings live in [experiment.toml](configs/experiment.toml).

## Next steps

| Date | Deliverable |
|---|---|
| October 29 | Forecast definitions, CSV interfaces, backtest and bootstrap evaluator |
| November 12 | Baselines and an initial HPI-and-rates transfer experiment |
| November 19 | Prediction CSVs for the three Transformer variants |
| November 26 | Model comparisons, confidence intervals and feature ablations |
| December 10 | Final report and reproducibility materials |

The Chinese project plan remains the execution reference. Policy extraction is optional and comes after the core forecasting work.
