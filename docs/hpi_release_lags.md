# HPI release timing

[hpi_release_lags.csv](hpi_release_lags.csv) lists the 41 markets with validation forecasts. Its country names and source files match the forecasting panel. Fill the evidence fields as sources are checked. An empty lag is unknown, rather than zero.

| Field | What to enter |
|---|---|
| `series_id`, `country`, `source_file` | Prefilled identifiers. Keep these aligned with the forecasting data. |
| `lag_days` | Whole calendar days from the observed quarter-end to publication. Leave blank until supported. |
| `reference_period` | The quarter the observation describes, such as `2025Q2`. |
| `release_date` | The date that observation was published, in `YYYY-MM-DD` format. |
| `evidence_url` | A publisher release, schedule or archived record supporting the timing. |
| `status` | One of the four statuses below. |
| `notes` | What was checked, the source series, historical coverage and any remaining uncertainty. |

Use `unverified` while researching, with possible dates or assumptions in `notes`. Use `release_example` for a dated observation, supplying its reference quarter, release date and measured lag. Use `schedule_estimate` for an official schedule that has not been checked against historical releases. Use `verified` only when the evidence supports availability for the exact source series and historical scope described in `notes`.

A national statistical office's release date may differ from the date the value entered the OECD dataset. Record which one the evidence establishes. A page update date or a revision date alone does not establish first publication. One recent release example does not establish a constant lag for 2000–2009.

Rebuild the matrix after updating the table:

```bash
python scripts/build_availability_matrix.py
```

The matrix contains 41 HPI rows and the existing macro indicators for five countries. Its HPI overlap is calculated across all 41 markets. Macro overlap keeps its original country groups. Unverified blank rows preserve the timing evidence already recorded for the original five countries. The builder reads the lag table but never overwrites it. At the end of each run it lists any supported lag (status other than `unverified`) that is longer than the lag assumed in `configs/experiment.toml` or `configs/experiment_long.toml`.

This table records evidence separately from the experiment settings. The default 100-day HPI lag and Taiwan's 60-day lag remain assumptions in the configurations. A supported lag above 100 days needs review of the data source, reference period and evidence type before changing the timing rule. If the common forecast date changes, rebuild the information sets and rerun the affected comparisons.
