# Multi-Market Housing Forecasting

Cross-market housing price forecasting with transfer learning, multi-horizon Transformers, and an optional LLM policy-signal pipeline.

**Status:** Initial project setup. Models, backtesting, and policy extraction are under development; no forecasting or inference-performance results are claimed yet.

## Research questions

- Does pretraining on other housing markets improve forecasts for Taiwan compared with training on Taiwan alone?
- How does transfer learning compare with joint multi-market training and simple forecasting baselines?
- Do policy signals extracted from public announcements add predictive value beyond economic fundamentals?

The initial transfer experiment uses HPI and interest rates. The broader dataset targets at least three markets including Taiwan, quarterly observations, and approximately 6–8 comparable features selected using economic reasoning and data availability.

## Planned technical highlights

- **Multi-horizon forecasting:** A compact Transformer predicts 1–4 quarters ahead in one forward pass. Experiments compare Taiwan-only training, joint multi-market training, and source-market pretraining followed by Taiwan fine-tuning.
- **Comparable data contracts:** Shared feature definitions, country-specific mapping files, publication-lag metadata, and standardized prediction CSVs connect data preparation, modeling, and evaluation.
- **Time-aware evaluation:** Expanding-window backtests, synchronized calendar cutoffs across markets, training-window-only preprocessing, and explicit label-overlap checks.
- **Statistical comparisons:** Historical-mean, AR, Ridge, and GRU baselines; RMSE, MAE, direction accuracy, feature-group ablations, and bootstrap confidence intervals for paired model differences.
- **Optional policy extraction:** Schema-constrained LLM outputs, rule-based validation, source evidence spans, and human-audited extraction quality. A vLLM batch pipeline and prefix-caching experiments are planned if the core forecast milestones are met.
- **Reproducible performance measurement:** Record model versions, hardware, batch settings, token lengths, throughput, latency, memory use, and extraction quality when benchmarking the policy pipeline.

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

The directories below are placeholders for the planned implementation.

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

## Data and reproducibility

Raw datasets, model checkpoints, credentials, and large generated artifacts are excluded from version control. Future dataset documentation will record source URLs, access dates, units, transformations, coverage, publication lags, and redistribution restrictions.

Runnable setup instructions and experiment commands will be added alongside their implementations. Published results will link to configurations and evaluation artifacts.
