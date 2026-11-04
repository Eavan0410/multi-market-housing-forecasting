# Multi-Market Housing Price Forecasting

This is a course project on forecasting Taiwan housing prices. We compare a Transformer trained on Taiwan data with two ways of using overseas data: joint training, and pretraining followed by fine-tuning on Taiwan.

Validation runs are complete for the three main Transformer settings, the AR baselines, fine-tuning variants, and HPI-only country-pool comparisons. Results and current conclusions are below. Final settings and held-out testing are still pending. An optional LLM policy module is implemented locally, but has no saved real-data results yet.

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

## Progress and validation results

Updated November 3, 2026. Here, **validation complete** means prediction CSVs, run metadata and evaluation tables are saved locally. It does not mean the model settings are final or that the held-out test has been run.

| Experiment / task | Status | What is available |
|---|---|---|
| No-change, Taiwan AR, overseas AR and pooled AR | Validation complete | Four-feature baseline comparisons |
| Taiwan-only Transformer | Validation complete | Five seeds, four features |
| Joint Transformer | Validation complete | Four overseas markets plus Taiwan, five seeds, four features |
| Overseas pretraining + Taiwan fine-tuning | Validation complete | Default fine-tuning run, five seeds, four features |
| Fine-tuning variants | Validation complete | Head-only, shorter training and lower learning rate, five seeds each |
| Zero-shot Transformer | Validation complete | Trained on overseas markets only, evaluated on Taiwan |
| Joint Transformer with market embeddings | Validation complete | Five seeds, four features |
| HPI-only, wider AnaHPI pool | Validation complete | Joint Transformer and pooled AR, plus Taiwan AR and no-change baselines |
| HPI-only, four-country control | Validation complete | Joint Transformer and pooled AR; four overseas markets plus Taiwan |
| HPI-only Taiwan-only Transformer and pretrain + fine-tune | Not run | No saved predictions in this setting yet |
| Data availability matrix | Draft available | CSV and Excel saved; release lags and Taiwan HPI basis still need confirmation |
| LLM policy index | Code available; real-data experiment pending | No announcement dataset or model results saved yet |
| Final settings and held-out test | Pending | No test results saved; test period remains locked by default |
| Final report | Pending | Validation tables are available for the report |

### Four-feature experiments

These runs use HPI growth, bond yields, GDP growth and the change in credit / GDP. The overseas pool is Australia, Japan, South Korea and the United States. Every run is evaluated on Taiwan's national index at the same 24 validation origins, 2012Q1–2017Q4. Transformers use five seeds; AR and no-change use one. Errors below are in cumulative log-growth units, with lower values better. Transformer metrics are averaged over seeds, rather than computed from averaged predictions.

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

Joint training has lower RMSE than Taiwan-only training and default pretrain + fine-tune at every horizon. The three lighter fine-tuning variants improve on default fine-tuning, but still have higher RMSE than joint training at every horizon. Market embeddings give a slightly lower h=1 RMSE than the plain joint model, but do not improve h=2–4.

Pooled AR remains a strong baseline: its RMSE is lower than the plain joint Transformer at h=2–4. At h=4 the joint model's RMSE is 0.05846, compared with 0.05140 for pooled AR. Their paired RMSE difference is +0.00707, with a 95% block-bootstrap interval of [-0.00451, 0.01942]. This does not establish a clear difference at h=4. So far, using overseas data looks useful relative to the Taiwan-only Transformer, but we have not shown that a Transformer beats pooled AR.

Full results: [metrics](results/fundamentals/validation/metrics.csv), [metric confidence intervals](results/fundamentals/validation/metric_ci.csv), and [paired comparisons with pooled AR](results/fundamentals/validation/bootstrap_vs_ar_pooled__overseas_plus_target.csv).

### HPI-only and the four-country control

These runs use only `hpi_qoq`. The wider pool includes eligible overseas countries from AnaHPI. The four-country control limits overseas training data to Australia, Japan, South Korea and the United States; Taiwan is included in both joint pools. Both settings use the same 24 Taiwan validation origins as above.

| Model / training data | RMSE h=1 | h=2 | h=3 | h=4 | MAE h=4 |
|---|---:|---:|---:|---:|---:|
| No-change | 0.02313 | 0.03780 | 0.05059 | 0.06297 | 0.04404 |
| Taiwan AR | 0.02440 | 0.04553 | 0.06639 | 0.08593 | 0.07524 |
| Pooled AR: wider pool + Taiwan | 0.02173 | 0.03575 | 0.04417 | 0.05136 | 0.03929 |
| Joint Transformer: wider pool + Taiwan | 0.02273 | 0.03807 | 0.04689 | 0.05230 | 0.03926 |
| Pooled AR: four countries + Taiwan | 0.02138 | 0.03551 | 0.04410 | 0.05179 | 0.03996 |
| Joint Transformer: four countries + Taiwan | 0.02080 | 0.03604 | 0.04833 | 0.05611 | 0.04285 |

At h=4 the wider-pool Transformer is close to pooled AR: RMSE 0.05230 versus 0.05136, and MAE 0.03926 versus 0.03929. The paired RMSE difference is +0.00094, with a 95% interval of [-0.00264, 0.00525]. The result does not show an improvement over pooled AR. In the wider pool, pooled AR also has lower RMSE at h=1–3.

The four-country control helps separate changing the inputs from changing the country pool. With the same four overseas markets, the HPI-only joint model has lower RMSE than the four-feature joint model at h=1 and h=4, and nearly the same RMSE at h=2 and h=3. Within HPI-only runs, the wider pool lowers Transformer RMSE at h=3–4 but raises it at h=1–2 compared with the four-country control. More countries therefore do not help at every horizon. These are descriptive comparisons; the saved bootstrap tables do not directly test the feature change or the two Transformer country pools against each other.

Full results: [metrics](results/hpi_only/validation/metrics.csv), [metric confidence intervals](results/hpi_only/validation/metric_ci.csv), and [paired comparisons with wider-pool AR](results/hpi_only/validation/bootstrap_vs_ar_pooled__overseas_plus_target.csv).

These conclusions are preliminary: there are only 24 validation origins, release lags remain provisional, and the held-out test has not been evaluated. Next we need to confirm the data assumptions, settle the final settings and then run the locked test.

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

These add `preds_ar__overseas.csv` and `preds_ar_pooled__overseas_plus_target.csv` to the same experiment folder. Use `ar_pooled` for the pooled run: plain `ar` fits each series separately, so `--models ar --pool overseas_plus_target` gives the same Taiwan forecasts as the Taiwan-only AR. These AR runs provide baselines for the joint Transformer.

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

To reproduce the HPI-only country-pool comparison:

```bash
# Wider pool
python scripts/run_backtest.py --models transformer ar_pooled --pool overseas_plus_target --features hpi_qoq --experiment hpi_only
# Same four overseas markets as the four-feature runs
python scripts/run_backtest.py --models transformer ar_pooled --pool overseas_plus_target --features hpi_qoq --experiment hpi_only --countries Australia Japan "South Korea" "United States"
python scripts/evaluate.py results/hpi_only/validation/preds_*.csv --reference ar_pooled__overseas_plus_target
```

## LLM policy index (optional extension)

This extension uses a local open-source model to turn Taiwan policy announcements into a quarterly tightening/easing index. The planned evaluation compares extracted events with a manually collected policy timeline. The code and timeline template are available; collecting the announcements and running the real model are still pending. Settings are in `configs/policy.toml`; input formats are described in `data/raw/policy/README.md`.

```bash
# dry run of the whole pipeline with keyword rules instead of a model
python scripts/policy_pipeline.py --backend mock extract
# on a GPU machine with vLLM installed (pip install vllm)
python scripts/policy_pipeline.py --backend vllm extract --limit 20
python scripts/policy_pipeline.py --backend vllm index
python scripts/policy_pipeline.py --backend vllm eval
python scripts/policy_pipeline.py --backend vllm spotcheck-export
```

Announcements are grouped by publication quarter. Before using the index for forecasting, we still need to check its availability at each forecast date. This extension is exploratory and kept separate from the core experiments.

Each prediction CSV has a `.meta.json` file recording its features, training pool, cutoffs and input-data fingerprints. The evaluator rejects incompatible manifests and checks that models and seeds cover the same forecast rows. Keep the metadata beside the CSVs. Files without metadata produce a warning because their settings cannot be verified.

## Backtest

Validation origins run from 2012Q1 to 2017Q4. Origins in 2018 form a buffer, and the test period starts in 2019Q1. The test period stays locked until the final configuration freeze; use validation for model and feature choices.

Training uses an expanding window, with one refit every four forecast origins. The last training origin is four quarters before the first evaluation origin. All countries share the same calendar cutoff, unpublished training labels are masked, and scaling is fitted within each training fold.

The evaluator reports RMSE, MAE and direction accuracy by horizon. Metrics are calculated for each seed and then averaged. Confidence intervals use paired circular blocks of four origin quarters, keeping all horizons and seeds together in each draw. The bootstrap holds the fitted seeds fixed; across-seed standard deviations describe training variability separately.

A draft availability matrix is saved in `results/availability_matrix.csv` and `.xlsx`. Publication lags are still placeholders until its timing assumptions are confirmed. The data are revised series rather than real-time vintages, so the backtest does not reconstruct exactly what was known historically.

## Files

```text
configs/experiment.toml    Features, horizons, periods, lags and seeds
data/raw/                  Overseas and Taiwan source tables
data/processed/           Four-country processed tables
data/interim/             Generated panels, ignored by Git
src/housing_forecast/
  data/                    Readers and quarterly transformations
  backtesting/             Windows, folds, scaling, runner and run metadata
  models/                  No-change, AR and Transformer models
  evaluation/              Metrics and block bootstrap
  policy/                  Policy extraction and quarterly index
scripts/                   Build, backtest and evaluation commands
tests/                     Data, leakage, alignment and evaluation checks
results/                   Outputs grouped by experiment and phase
```

Source tables in `data/raw/` and `data/processed/` are tracked. Generated panels go to `data/interim/`; checkpoints and large training artifacts are ignored. The Taiwan HPI/CPI and global AnaHPI tables were copied from the Spring 2026 dataset, so the build no longer needs a sibling data folder.

See the [experiment contract](docs/experiment_contract.md) for the forecast definitions, [framework notes](docs/backtest_framework.md) for adding models, and [design notes](docs/design_notes.md) for the data interfaces and merge rules. Runtime settings live in [experiment.toml](configs/experiment.toml).

## Course schedule

| Date | Planned work | Current status |
|---|---|---|
| October 29 | Forecast definitions and backtest | Framework implemented; data assumptions still provisional |
| November 12 | Baselines and initial transfer experiment | Validation runs complete |
| November 19 | Three main Transformer settings | Validation runs complete; final settings pending |
| November 26 | Errors, confidence intervals and feature choices | Initial comparisons complete, including HPI-only and country controls; final choices pending |
| December 10 | Report and rerun instructions | Run commands available; final report pending |

These are the course milestones; some initial runs are already available. Policy extraction is optional, and the forecasting experiments come first.
