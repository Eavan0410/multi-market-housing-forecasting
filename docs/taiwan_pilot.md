# Taiwan pilot (background)

These were the first experiments, before the project moved to many markets. They evaluate only Taiwan's national real HPI, at 24 forecast dates in 2012Q1–2017Q4 and horizons 1–4, so their scores are not comparable with the multi-market results.



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

Results: [metrics](../results/fundamentals/validation/metrics.csv), [confidence intervals](../results/fundamentals/validation/metric_ci.csv), and [comparisons with pooled AR](../results/fundamentals/validation/bootstrap_vs_ar_pooled__overseas_plus_target.csv).

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

Results: [metrics](../results/hpi_only/validation/metrics.csv), [confidence intervals](../results/hpi_only/validation/metric_ci.csv), and [comparisons with wider-pool AR](../results/hpi_only/validation/bootstrap_vs_ar_pooled__overseas_plus_target.csv).

## Reproduce the pilot

Run these commands from the repository root after the setup in the [README](../README.md#run-locally).

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

