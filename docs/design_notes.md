# Design notes

Background for the methods section and for whoever takes over the code. The code itself keeps short docstrings; the reasons live here.

## 1. Fair-test rules and where the code enforces them

| Rule (project plan) | Where | How |
|---|---|---|
| 1. Expanding window: train only on the past, test on later periods | `backtesting/splits.py`: `make_splits`, `train_mask` | Each fold trains on all origins up to a cutoff and tests the next `refit_every` origins; the training window only grows. |
| 2. Gap of 4 quarters between training and test, so training labels never overlap the test period | `make_splits`, `check_no_leakage` | Last training origin = first test origin − `gap_quarters`. Every fold asserts that no visible training target falls after the first test origin. |
| 3. One calendar cutoff for every country | `make_splits`, `train_mask`, `mask_unpublished_labels` | Folds are defined on calendar quarters and applied to all series. A training label is kept only if it was published by the fold's cutoff date (the `as_of_date` of the first test origin); later labels are hidden. |
| 4. Standardization, selection and fitting only inside the training window; respect release lags | `backtesting/runner.py`, `backtesting/scaling.py`, `backtesting/infoset.py` | The runner fits `WindowStandardizer` and the model on the fold's training set only. Each feature is shifted back by whole quarters until it is public at the forecast date (`delay_quarters`). Row-wise transforms that only look backwards (deflation, log growth, monthly → quarterly, 4-quarter change) are done once beforehand. |
| 5. Same splits and metrics for every model; differences reported with CIs | `runner.py`, `data/contract.py`, `backtesting/manifest.py`, `evaluation/` | Every model runs through the same `run_backtest`. `check_alignment` requires identical forecast rows, labels, dates and folds across models and seeds. Run manifests stop the evaluator from mixing different information sets. Metrics are per seed, then averaged; CIs come from a circular block bootstrap over origins. |

Periods (forecast origins, same dates for every country):

```text
[ dev: training only ][ validation 2012Q1–2017Q4 ][ buffer 2018 ][ test 2019Q1– (locked) ]
```

The test period is locked in code (`PeriodLockedError`) until the configuration freeze; `--unlock-test` is for the final run only.

Tests for each rule: `tests/test_backtest_rules.py`.

## 2. Hand-off tables

### Panel (data → backtest)

Long format, one row per series and quarter. Raw observations only: no scaling, no filling, no interpolation; missing values stay empty.

| Column | Meaning |
|---|---|
| `series_id` | unique series name, e.g. `Australia`, `Taiwan_national`, `Taiwan_Taipei` |
| `country` | common English name; Taiwan cities keep `country = Taiwan` |
| `geography_level` | `national` or `city` |
| `quarter` | `2010Q1` |
| `hpi` and other variables | one column per variable |

Checked by `validate_panel`: required columns, quarter format, no duplicate `(series_id, quarter)`, one country per series.

### Predictions (every model → evaluator)

One row per model, series, origin, horizon and seed.

| Column | Meaning |
|---|---|
| `model` | model label; `<model>__<pool>` when the pool is not `target_only` |
| `series_id` | evaluated series (main table: `Taiwan_national`) |
| `origin_quarter` | t, the base quarter |
| `as_of_date` | date the forecast is made: end of t + HPI release lag |
| `target_quarter` | t + h |
| `horizon` | h, 1–4 |
| `y_true` | log(HPI_real[t+h] / HPI_real[t]) |
| `y_pred` | the model's forecast of `y_true` |
| `target_type` | `cum_log_growth_real` |
| `seed` | training seed (deterministic models: first seed only) |
| `split_id` | fold, e.g. `val_2012Q1` |

Checked by `validate_predictions` and `check_alignment`: no empty or non-finite values, `target_quarter = origin_quarter + horizon`, `as_of_date` after the origin quarter, no forecast repeated across folds, and the same rows and definitions for every model and seed. Models from other scripts (e.g. the GRU) must write exactly these columns.

## 3. B's processed files and how they are merged

`data/processed/<ISO>_processed.csv`, stacked:

```text
country, feature, REF_AREA, FREQ, TIME_PERIOD, SOURCE_PERIOD, OBS_VALUE, UNIT_MEASURE, ADJUSTMENT, OBS_STATUS
```

- `FREQ = Q`: `SOURCE_PERIOD` is `1970-Q1` or a quarter-start date (`1977-10-01`).
- `FREQ = M`: `SOURCE_PERIOD` is `1956-01`.

Merge rules (`data/b_adapter.py`):

1. Native quarterly values are used as they are.
2. Monthly values become a quarterly mean, only for quarters with all three months.
3. If a quarter has both, the native quarterly value wins (monthly only fills gaps).
4. Nothing is filled or interpolated.
5. Country names are unified (`Korea` → `South Korea`).

Taiwan files in `data/raw/taiwan/` (`data/taiwan_raw.py`) are found by their `feature` value, not by file name. Monthly rates use the quarterly mean; the consumer loan balance (a stock) uses the value of the quarter's last month. Credit / GDP = end-of-quarter balance ÷ sum of the last four quarters of nominal GDP × 100, matching the BIS definition. Taiwan's macro series are national, so the seven cities share them, and history before the HPI starts is kept so that 4-quarter changes exist from the first HPI quarter.
