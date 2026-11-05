# Market selection (B fills this in before any results are looked at)

Source table: [market_list.csv](market_list.csv) — one row per national market with its HPI
start/end, history group, region, large quarterly jumps in real HPI (|log change| > 0.10), the
number of 2000Q1–2009Q4 validation origins and how many folds it actually trains in. Rebuild it with `python scripts/build_market_list.py`.

Rules
- Choose only from data properties (history length, region, data quality). Do not look at
  model results; do not leave a market out because a model does badly on it.
- Markets with no validation origins (HPI starting 2008 or later) cannot be evaluated in the
  current validation window.
- Being on a pool's list is not the same as adding training data. A market only contributes
  samples to a fold once it has labels published before that fold's cutoff; see the columns
  `folds_with_training_data`, `train_samples_first_fold`, `train_samples_last_fold` and the
  per-fold table [market_fold_samples.csv](market_fold_samples.csv). Under the current timing
  rules, the 10 markets whose HPI starts in 2007 or later add no training data in any
  validation fold, Latvia and Lithuania only in the last fold, and markets starting in
  2000–2005 only in later folds; the number of markets that actually train rises from 24 in
  the first fold to 38 in the last.
- The data-scale test reports both the nominal number of markets in a pool and the number that
  actually contribute training samples in each fold. Rebuild the table after the timing rules
  change.
- Write one line of reasons for every choice.

## 1. Evaluation markets for the data-scale test (about 5)

Cover long, medium and short histories and several regions; every training pool will include them.

| Market | History group | Region | Reason |
|---|---|---|---|
| | | | |

## 2. Order in which other markets are added (pools of 10, 20, all)

Three orders fixed in advance, e.g. one by region rotation, one by history length, one random
(write the seed). List the markets added to reach 10 and 20; "all" adds the rest.

| Order | Rule | Added to reach 10 | Added to reach 20 |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## 3. Leave-one-market-out targets (3–5)

| Market | History group | Region | Reason |
|---|---|---|---|
| | | | |
