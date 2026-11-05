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
| Germany | Long (by 1979) | Western Europe | Its HPI begins in 1970Q1 with 222 continuous quarters, giving the evaluation set a long-history European market. |
| Australia | Long (by 1979) | Oceania | Its HPI begins in 1970Q1 with 223 continuous quarters, providing a long-history market outside Europe. |
| South Korea | Medium (1980–1999) | Asia | Its HPI begins in 1986Q1 with 159 continuous quarters, representing a medium-history Asian market. |
| Mexico | Short (2000+) | Latin America | Its HPI begins in 2005Q1 with 82 continuous quarters, providing a short-history Latin American market. |
| Taiwan (`Taiwan_national`) | Short (2000+) | Asia | Its HPI begins in 2001Q1 with 99 continuous quarters, representing the short-history transfer target in Asia. |

## 2. Order in which other markets are added (pools of 10, 20, all)

Three orders fixed in advance, e.g. one by region rotation, one by history length, one random
(write the seed). List the markets added to reach 10 and 20; "all" adds the rest.

| Order | Rule | Added to reach 10 | Added to reach 20 |
|---|---|---|---|
| 1. Region rotation | Add markets from different regions in a fixed rotation to increase geographic diversity, breaking ties by training-data availability and HPI coverage. | South Africa, Canada, Israel, Norway, Italy | Russia, Japan, Colombia, United Kingdom, New Zealand, United States, Denmark, Spain, Bulgaria, Indonesia |
| 2. Long-history first | Add markets in descending order of valid HPI quarters to test data depth; break ties by training-fold availability and market name. | Japan, South Africa, United Kingdom, Canada, Norway | United States, Belgium, Denmark, Finland, France, Ireland, Italy, Netherlands, Sweden, Switzerland |
| 3. Seeded random | Use a reproducible random order (NumPy `default_rng`, seed `20261104`) to avoid a history- or region-based preference. | France, Switzerland, Slovenia, Colombia, Poland | Spain, Chile, United Kingdom, Luxembourg, Lithuania, New Zealand, Israel, Portugal, Norway, Indonesia |

**Pool counting:** The five evaluation markets above are included in every data-scale training pool, as specified by this template. Thus 10 = 5 fixed + 5 added, 20 = 5 fixed + 15 added, and all = 41 nominal eligible markets (the remaining 21 markets are added at this stage). The three full addition orders are frozen as follows; the first 15 entries in each order are shown in the table above.

- **Order 1, remaining 21 (positions 16–36):** Chile, Belgium, Finland, Portugal, Croatia, France, Sweden, Greece, Estonia, Ireland, Iceland, Poland, Netherlands, Slovak Republic, Switzerland, Latvia, Austria, Lithuania, Luxembourg, Hungary, Slovenia.
- **Order 2, remaining 21 (positions 16–36):** New Zealand, Spain, Colombia, Portugal, Israel, Greece, Austria, Iceland, Russia, Indonesia, Chile, Bulgaria, Croatia, Estonia, Poland, Slovak Republic, Latvia, Lithuania, Hungary, Slovenia, Luxembourg.
- **Order 3, remaining 21 (positions 16–36):** Iceland, United States, Austria, Italy, Finland, Canada, Latvia, Bulgaria, Slovak Republic, Denmark, Greece, Estonia, Ireland, Japan, South Africa, Sweden, Russia, Croatia, Hungary, Belgium, Netherlands.

**Feasibility note (data only):** In the 41-market eligible set, Hungary, Luxembourg and Slovenia have zero usable training folds under the current timing rules. The all-market pool therefore has 41 nominal members but 38 contributing markets in the last fold. They remain listed; do not remove them based on model outcomes. All three orderings were specified without using prediction results.

## 3. Leave-one-market-out targets (3–5)

| Market | History group | Region | Reason |
|---|---|---|---|
| Germany | Long (by 1979) | Western Europe | With HPI starting in 1970Q1, it is a long-history European target for assessing transfer to an unseen market. |
| South Korea | Medium (1980–1999) | Asia | With HPI starting in 1986Q1, it tests transfer to a medium-history Asian market. |
| Mexico | Short (2000+) | Latin America | With HPI starting in 2005Q1 and only two usable training folds under current timing, it tests transfer to a short-history Latin American market. |
| Taiwan (`Taiwan_national`) | Short (2000+) | Asia | With HPI starting in 2001Q1, it tests transfer to a short-history Asian target whose nominal HPI requires separate real-price conversion. |
