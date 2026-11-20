# IMF iMaPP

Source: https://www.elibrary-areaer.imf.org/Macroprudential/Documents/iMaPP_database-2024-12-02.zip

The December 2024 vintage covers 135 economies from 1990 through December 2023. It includes Taiwan. Cite the IMF integrated Macroprudential Policy Database, originally constructed by Alam et al. (2019), IMF Working Paper 19/66.

The first experiment adds one quarterly feature: the net sum of `LTV`, `DSTI` and `LoanR_HH` actions in the `MaPP` sheet. Positive values indicate net tightening, negative values net easing. These instrument groups concern secured lending and household credit and are not exclusively residential mortgage measures. The feature counts policy changes, not the level of the LTV cap or the amount of credit affected.

Require three complete months and all three instrument values. Missing data remain missing; zero means a recorded quarter without net changes. Sum action flows rather than averaging them. Avoid double-counting the aggregate `LoanR` and its household subcategory.

The database uses implementation dates and includes historical revisions. We assume a policy quarter is available by quarter-end. This is an exploratory reconstruction, not a real-time vintage backtest. The runner accepts `--policy-lag-days` to check a different timing assumption. Both arms use the same training and forecast samples, with no forward filling. The frozen HPI-only final-test configuration is separate.

```bash
pip install -e ".[imapp]"
python scripts/build_imapp.py
OMP_NUM_THREADS=1 python scripts/run_imapp_experiment.py
```

`--prepare-only` writes coverage without training. `--models no_change ar_pooled` runs the linear baselines only. Results are written to `results/imapp/validation`.
