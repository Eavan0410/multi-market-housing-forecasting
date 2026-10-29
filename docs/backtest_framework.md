# Backtest framework

## Pipeline

```text
data/processed/ (B, 4 countries) ────┐
data/raw/taiwan/ (HPI, CPI, macro) ──┼─> scripts/build_panel.py ──> data/interim/panel_quarterly.csv
data/raw/global/ (AnaHPI, HPI only) ─┘                              data/interim/taiwan_cpi_monthly.csv
                                                              │
                         prepare_panel      (deflate Taiwan, hpi_qoq, Taiwan cpi_yoy,
                                             household_credit_gdp_d4)
                                                              │
                         build_samples      (labels, availability shifts, windows)
                                                              │  SampleSet: X (n, lookback, F), y (n, 4)
                         make_splits        (validation folds; test locked)
                                                              │
                         run_backtest       (pool, label masking, leakage check,
                                             window-only scaling, seeds)
                                                              │
                         results/<phase>/preds_<model>.csv   (shared format)
                                                              │
                         scripts/evaluate.py (metrics.csv, bootstrap_vs_<ref>.csv)
```

## Modules

| File | Role |
|---|---|
| `data/contract.py` | panel and prediction CSV schemas, validators, row-alignment check |
| `data/b_adapter.py` | B's per-country processed files → panel; adds other AnaHPI countries and Taiwan from Spring Raw |
| `data/readers.py` | readers for the raw tables copied from Spring 2026 (AnaHPI, Taiwan HPI, Taiwan CPI) |
| `data/taiwan_raw.py` | B's Taiwan raw files → quarterly interest_rate, gdp_growth, household_credit_gdp |
| `data/transforms.py` | backward-looking row-wise transforms (deflation, log growth, monthly → quarterly, credit / GDP, 4-quarter change) |
| `backtesting/labels.py` | cumulative log-growth labels, matched by calendar quarter |
| `backtesting/infoset.py` | `as_of_date`, release-lag shifts, input windows → `SampleSet` |
| `backtesting/splits.py` | folds, test lock, label publication mask, leakage assertions |
| `backtesting/scaling.py` | standardizer fitted on the training window |
| `backtesting/runner.py` | runs any `Forecaster` over all folds and seeds |
| `models/base.py` | `Forecaster` interface, seeding |
| `models/baselines.py` | `NoChange`, `ARDirect` |
| `evaluation/metrics.py` | RMSE, MAE, direction accuracy, per-seed table |
| `evaluation/bootstrap.py` | paired circular block bootstrap |

## Adding a model

```python
from housing_forecast.models.base import Forecaster

class MyModel(Forecaster):
    name = "my_model"
    stochastic = True       # runs once per seed in configs/experiment.toml
    needs_scaling = True    # runner standardizes X with training-window stats

    def fit(self, train, seed):
        # train.X: (n, lookback, F); train.y: (n, 4), NaN = label hidden or missing -> mask in the loss
        # train.meta: series_id, country, geography_level, origin, as_of_date
        ...

    def predict(self, test):
        return y_pred      # (len(test), 4)
```

Register it in `models/baselines.REGISTRY` (or a new registry) and run `scripts/run_backtest.py --models my_model --pool ...`. A pretrain → fine-tune model gets the `overseas_plus_target` pool and splits it inside `fit` with `train.meta["country"] != "Taiwan"`.

## Known limitations

- Release lags are placeholders until B's availability matrix is done.
- `real_income_growth` is dropped from the feature set (starts 2007Q2; missing for Korea and Taiwan).
- Inputs: `hpi_qoq`, `interest_rate`, `gdp_growth`, `household_credit_gdp_d4`. `cpi_yoy` waits for Japan's CPI after 2021Q2; `unemployment_rate` waits for Taiwan's file. Adding a variable that Taiwan lacks would drop all Taiwan samples.
- With these inputs only series that have all four variables remain (B's 4 countries and Taiwan); the extra AnaHPI countries drop out. Use `--features hpi_qoq` for the larger HPI-only pool.
- Changing the feature list changes which origins have complete windows, so compare models only within the same feature setting (enforced, see "Results layout").
- Taiwan HPI and CPI are copies of the Spring 2026 Raw tables, stored as `data/raw/taiwan/taiwan_hpi_quarterly_raw.csv` (from `full_HPI_quarterly(city+taiwan).csv`) and `data/raw/taiwan/taiwan_cpi_monthly_raw.csv` (from `CPI_monthly.csv`), unchanged. The 43 extra AnaHPI countries (HPI only) come from `data/raw/global/anahpi_quarterly_raw.csv`, an unchanged copy of Spring 2026 Raw `Global/Raw/AnaHPI_quarter_data.csv`. The pipeline no longer reads anything outside the repository.

## Results layout

`run_backtest.py` writes one folder per experiment, so different settings never overwrite each other:

```text
results/<experiment>/<phase>/preds_<label>.csv         shared prediction format
results/<experiment>/<phase>/preds_<label>.meta.json   settings that produced it
```

`<experiment>` defaults to the feature list joined by `+` (override with `--experiment`). `<label>` is the model name, plus `__<pool>` when the pool is not `target_only` (e.g. `ar__overseas_plus_target`), so Taiwan-only, joint and pretrained runs of one model sit side by side.

The manifest records the features, window, periods, release lags, HPI basis and the SHA-256 of the panel and CPI files. `evaluate.py` stops if the files it scores differ in any of these; the pool and the model may differ. Files without a manifest (for example a CSV from another script) are scored with a warning.

## Feature definitions

| Feature | Definition | Taiwan source |
|---|---|---|
| `hpi_qoq` | log q/q growth of real HPI | Spring 2026 Raw HPI, deflated by CPI |
| `interest_rate` | long-term (10-year) government bond yield, quarterly mean, % p.a. | B, monthly |
| `gdp_growth` | real GDP growth, q/q, seasonally adjusted, % | B, quarterly |
| `household_credit_gdp_d4` | 4-quarter change of household credit / GDP, percentage points | B: bank consumer loans (end of quarter) / last 4 quarters of nominal GDP × 100 (substitute for BIS household credit) |

`household_credit_gdp_d4` uses the change rather than the level so that a constant gap between Taiwan's bank-only consumer loans and BIS household credit (all lenders) drops out. The change does not remove differences in how the two series move, so Taiwan's series stays a substitute variable and is reported as such. Macro history before a country's HPI starts is kept, so 4-quarter changes are available from the first HPI quarter.
| `cpi_yoy` | CPI % change over 4 quarters | computed from the CPI index |

## Evaluation acceptance checks

Point metrics and bootstrap both average per-seed metrics, rather than scoring
seed-averaged predictions. The same sampled origin blocks are used for every
seed and model. Seeds are held fixed in the bootstrap; report across-seed
standard deviations separately from the time-block confidence intervals.

Every seed must contain the same forecast rows. Matching rows must agree on
`y_true` (rtol 1e-10, atol 1e-12), `target_quarter`, `as_of_date`, `target_type`
and `split_id`. Duplicate forecast rows across folds and non-finite values
are rejected. The evaluator requires all configured seeds for each model,
except those explicitly listed in `[run].deterministic_models`, which require
only the first seed. Keep that list up to date when registering models.
