# Backtest framework

## Pipeline

```text
B's processed files ─┐
Spring 2026 Raw ─────┴─> scripts/build_panel.py ──> data/processed/panel_quarterly.csv
                                                     data/processed/taiwan_cpi_monthly.csv
                                                              │
                         prepare_hpi_panel  (deflate Taiwan, hpi_qoq)
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
| `data/raw_dev.py` | readers for Spring 2026 Raw (AnaHPI, Taiwan HPI, Taiwan CPI) |
| `data/transforms.py` | backward-looking row-wise transforms (deflation, log growth, monthly → quarterly) |
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
- Only `hpi_qoq` is used as an input for now. B's other variables are in the panel but Taiwan has no mapped equivalents yet; adding them to `[features].use` before that would drop all Taiwan samples.
- Taiwan HPI and CPI still come from Spring 2026 Raw until B delivers the Taiwan series.

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
