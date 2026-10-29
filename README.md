# Multi-Market Housing Price Forecasting with LLM Policy Signals

A deep learning research project combining cross-market Transformer transfer learning with structured LLM inference for housing-policy signals.

**Status:** Backtest framework, evaluation (with block bootstrap) and shared CSV contracts are in place, with "no change" and AR benchmarks running on the validation period. Transformer models and policy extraction are under development; no final results are claimed yet.

## Research questions

- Does pretraining on other housing markets improve forecasts for Taiwan compared with training on Taiwan alone?
- How does transfer learning compare with joint multi-market training and simple forecasting baselines?
- Do policy signals extracted from public announcements add predictive value beyond economic fundamentals?

The initial transfer experiment uses HPI and interest rates. The broader dataset targets at least three markets including Taiwan, quarterly observations, and approximately 6–8 comparable features selected using economic reasoning and data availability.

## Technical highlights

The following components describe the planned implementation. Deep learning and forecasting evaluation are the core deliverables; the LLM inference pipeline is an optional extension.

### Deep Learning and Transfer Learning

- **Multi-horizon time-series Transformer:** Encode multivariate quarterly histories and directly predict 1–4 quarters ahead in one forward pass.
- **Cross-market pretraining and fine-tuning:** Learn from source markets, then adapt to Taiwan. Compare against Taiwan-only training and joint multi-market training under identical information cutoffs.
- **Market-aware representations:** Evaluate country embeddings to capture market-specific responses while retaining a shared feature space.
- **Controlled model comparisons:** Benchmark against historical mean, AR, Ridge, and GRU; use feature-group ablations and paired uncertainty estimates to assess gains.

### LLM Inference and Structured Extraction

- **vLLM batch inference:** Process Chinese-language policy announcements into structured records for quarterly policy and credit-condition signals.
- **JSON Schema constrained decoding:** Constrain output structure and validate field types, required fields, and permitted values. Schema validity and factual extraction accuracy will be evaluated separately.
- **Prefix caching:** Reuse shared instruction and schema prefixes where supported; measure cache-enabled versus cache-disabled performance with matched workloads.
- **Rule + LLM validation:** Check extracted fields against deterministic rules and evidence spans in the source document, with human-audited samples for quality measurement.
- **Inference benchmarking:** Report documents/second, output tokens/second, end-to-end latency distributions, peak GPU memory, and extraction quality. Record hardware, model and engine versions, prompt/output lengths, batch settings, and cache conditions.

### Evaluation and Research Infrastructure

- **Shared data and prediction contracts:** Country-specific mapping files, common feature definitions, publication-lag metadata, and standardized prediction CSVs.
- **Leakage-aware backtesting:** Expanding windows, synchronized calendar cutoffs across markets, training-window-only preprocessing, and checks for training-label overlap.
- **Statistical evaluation:** RMSE, MAE, direction accuracy, and paired bootstrap confidence intervals that account for time dependence.
- **Policy-signal ablation:** Compare the same forecasting model with and without policy signals to isolate their incremental contribution.

## Architecture

```text
Economic series -> Country mappings -> Quarterly feature panel
                                                |
                                      Expanding-window splits
                                                |
                         Baselines / Taiwan Transformer /
                         Joint Transformer / Pretrain + fine-tune
                                                |
                                      Prediction CSVs
                                                |
                              Shared evaluator -> Tables and plots

Public policy announcements -> Structured extraction -> Evidence checks
                                      |
                              Quarterly policy signals
                                      |
                            Optional forecast ablations
```

## Repository layout

`policy/` is still a placeholder; the other packages are implemented.

```text
configs/                   Experiment and model configurations
data/
  raw/                     Local source data (not tracked)
  processed/               Local feature panels (not tracked)
  mappings/                Country-specific mapping definitions
docs/                      Data contracts and methodology notes
src/housing_forecast/
  data/                    Loading, harmonization, availability handling
  backtesting/             Time splits and leakage checks
  models/                  Baseline and Transformer adapters
  evaluation/              Metrics and paired bootstrap comparisons
  policy/                  Optional LLM extraction and aggregation
scripts/                   Experiment and benchmark entry points
tests/                     Contract and leakage-prevention tests
results/                   Small, reviewable evaluation summaries
```

## Evaluation protocol

All models will share forecast origins, target definitions, horizons, and scoring rules. Each market uses the same calendar cutoff. Preprocessing and model selection use only the training portion of each fold; source-market pretraining must obey the same information cutoff.

The project plan specifies a four-quarter separation between training and testing. The implementation will define this using forecast-origin and label dates and verify that no training label crosses the permitted boundary. Economic observations and policy announcements must also respect their availability dates.

Bootstrap comparisons will resample paired forecast errors while accounting for serial dependence and overlapping forecast horizons. Policy-signal contribution will be measured by comparing the same forecasting model with and without those signals, separately from comparisons against GRU.

## Roadmap

- **October 29:** Freeze forecast definitions and shared CSV contracts; implement backtesting and evaluation with bootstrap.
- **November 12:** Standardized multi-market data, baseline results, and an initial HPI-and-rates transfer experiment.
- **November 19:** Deliver final prediction CSVs for the three Transformer variants.
- **November 26:** Complete model comparisons, confidence intervals, feature-group ablations, and the transfer-learning conclusion.
- **December 10:** Complete the report, reproducibility documentation, and handoff materials.

The Chinese project plan is the execution reference. Policy extraction, sensitivity analysis, and other extensions remain optional and must not delay core forecasting deliverables.

## Setup and usage

```bash
conda create -n hpf python=3.11 -y && conda activate hpf
pip install -e ".[dev]"
pytest -v

# 1. Build the panel (B's processed files + Spring 2026 Raw for Taiwan and other AnaHPI countries)
python scripts/build_panel.py --b-dir data/processed --spring-raw "../26spring data"
# 2. Backtest on the validation period (the test period is locked)
python scripts/run_backtest.py --models no_change ar --pool target_only
# 3. Metrics + paired block-bootstrap CIs against "no change"
python scripts/evaluate.py results/validation/preds_*.csv --reference no_change
```

- Experiment contract (draft for the freeze): [`docs/experiment_contract.md`](docs/experiment_contract.md), values in [`configs/experiment.toml`](configs/experiment.toml)
- Framework design and how to add a model: [`docs/backtest_framework.md`](docs/backtest_framework.md)

## Data and reproducibility

Raw datasets, model checkpoints, credentials, and large generated artifacts are excluded from version control. Future dataset documentation will record source URLs, access dates, units, transformations, coverage, publication lags, and redistribution restrictions.

Published results will link to configurations and evaluation artifacts.
