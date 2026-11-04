# Experiment settings (draft)

These notes cover the Taiwan h=1–4 experiments. The [README](../README.md#multi-market-results) has the newer multi-market h=1–20 results.

Shared settings are in [`configs/experiment.toml`](../configs/experiment.toml). This page explains the main choices. Items marked **TODO** still need team confirmation.

## Target

- Series: Taiwan national real HPI (`Taiwan_national`). The 7 cities are reported separately and are not treated as independent samples.
- Label: cumulative log growth `y(t,h) = log(HPI_{t+h} / HPI_t)`, h = 1, 2, 3, 4, on the **real** HPI level.
- HPI basis: OECD AnaHPI is already real. Taiwan HPI is treated as nominal and deflated in our pipeline with the quarterly mean of monthly CPI (`hpi_real = hpi / CPI * 100`). **TODO (B):** confirm the Taiwan HPI basis.
- Main horizon: h = 4. **TODO (team).**

## Three times per forecast

| Field | Meaning |
|---|---|
| `origin_quarter` | t, the base quarter |
| `as_of_date` | the day the forecast is made = end of quarter t + Taiwan HPI release lag |
| `target_quarter` | t + h |

Every variable carries an observation quarter and an availability date (observation quarter end + release lag). At `as_of_date` a model sees only values whose availability date has passed. Release lags are placeholders in the config until its timing evidence is confirmed (**TODO B**). We use the latest revised data, not real-time vintages, which limits how closely the backtest represents forecasts made at the time.

## Periods (origin quarters, same calendar dates for every country)

| Period | Origins | Use |
|---|---|---|
| Dev | before 2012Q1 | training only |
| Validation | 2012Q1–2017Q4 | all model / feature / country / tuning choices |
| Buffer | 2018Q1–2018Q4 | not scored, so validation labels never reach the test period |
| Test | 2019Q1–latest | locked in code until the 11/19 freeze (`--unlock-test`) |

## Backtest rules

1. Expanding window; one refit per 4 test origins.
2. Last training origin = first test origin − 4 quarters, so no training label overlaps the test period.
3. One calendar cutoff for all countries. A training label is used only if it was published by the fold's cutoff date (the `as_of_date` of the first test origin).
4. Standardization, feature selection and model fitting happen inside each training window only. Row-wise transforms that look only backwards (deflation, log growth, monthly → quarterly mean, 4-quarter changes) are done once beforehand. Input windows then use forward fill up to 2 quarters.
5. All models share the same folds, the same information set and the same prediction format, and must be row-aligned (checked by `check_alignment`).

## Pools

- Taiwan-only: `Taiwan_national`.
- Pretraining pool: overseas countries only; no Taiwan national or city series.
- Joint: overseas plus `Taiwan_national`.

## Metrics and uncertainty

- RMSE, MAE and direction accuracy per horizon. A "no change" forecast of 0 counts as a wrong direction call.
- Paired circular block bootstrap over origin quarters (block length 4, 2000 draws, 95% CI). All horizons and series of one origin are resampled together.
- Seeds: `[0, 1, 2, 3, 4]`. Deterministic models use seed 0 only; metrics are computed separately for each seed and then averaged. Each bootstrap draw resamples the same origin blocks for all models and seeds, computes each seed's metric, averages those metrics within each model, and takes the paired difference. Predictions are not averaged before scoring. CIs condition on the fitted seeds and describe origin-sampling uncertainty; seed standard deviations separately describe training variability.
- Evaluation checks complete row coverage within every seed and agreement across models on labels, dates, target definitions and folds. `[run].deterministic_models` lists models that require only the first configured seed; other models require the full seed list. Adding a deterministic model requires updating that list.

## To confirm before 11/19

The final Transformer settings, the tuning budget, and the feature list (`[features].use`). Current draft list: `hpi_qoq`, `interest_rate`, `gdp_growth`, `household_credit_gdp_d4` (definitions in `docs/backtest_framework.md`).
