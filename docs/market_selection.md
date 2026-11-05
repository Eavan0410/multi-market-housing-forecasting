# Market selection
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
| United States | Long (by 1979) | North America | HPI begins 1970Q1; 223 continuous quarters, 40 validation origins, and 10 usable training folds provide long-history coverage in North America. |
| South Africa | Long (by 1979) | Africa | HPI begins 1966Q1; 238 continuous quarters, 40 validation origins, and 10 usable training folds provide long-history coverage in Africa. |
| South Korea | Medium (1980–1999) | Asia | HPI begins 1986Q1; 159 continuous quarters, 40 validation origins, and 10 usable training folds provide medium-history coverage in Asia. |
| Indonesia | Short (2000+) | Asia | HPI begins 2002Q1; 94 continuous quarters, 24 validation origins, and 5 usable training folds provide short-history coverage in Asia. |
| Russia | Short (2000+) | Eastern Europe | HPI begins 2001Q1; 97 continuous quarters, 28 validation origins, and 6 usable training folds provide short-history coverage in Eastern Europe. |

## 2. Order in which other markets are added (pools of 10, 20, all)

Three data-only orders are fixed in advance: region rotation, long-history first, and seeded random. The five evaluation markets remain in every pool.

| Order | Rule | Added to reach 10 | Added to reach 20 |
|---|---|---|---|
| 1 | Rotate across the least represented macroregion; break ties using first-fold training samples, HPI history, and market ID. | Australia, Colombia, Israel, New Zealand, Chile | United Kingdom, Canada, Mexico, Norway, Japan, Belgium, Taiwan, Denmark, Finland, France |
| 2 | Rank by HPI quarters (descending); break ties by usable folds, first-fold samples, and market ID. | Japan, United Kingdom, Australia, Canada, Norway | Belgium, Denmark, Finland, France, Germany, Ireland, Italy, Netherlands, Sweden, Switzerland |
| 3 | Randomly permute sorted market IDs with NumPy `default_rng(20261104)`. | Finland, Switzerland, Slovak Republic, Chile, New Zealand | Spain, Canada, Taiwan, Lithuania, Latvia, Mexico, Ireland, Norway, Netherlands, Hungary |

**Pool sizes:** 10 = 5 fixed evaluation markets + 5 additions; 20 = the same 5 + the first 15 additions; all = 5 + all 36 additions (41 nominal markets). “Added to reach 20” lists the *next 10* markets. The remaining 21 markets enter at the “all” stage in the printed order.

**Selection basis:** The 41-market universe is defined by `validation_origins_2000_2009 > 0`. Evaluation and LOMO targets were previously chosen using HPI history, region and common data-availability requirements (at least 20 validation origins, 5 usable folds and no missing internal quarters); Taiwan is the predetermined transfer target. No model errors were used in the revised rules. Because prior H=20 results were already viewed, this revised design should be frozen for future tests, not described as an originally blind selection.

<details>
<summary>Python code to reproduce the three training-pool orders


```python
from collections import Counter
import numpy as np
import pandas as pd

SEED = 20261104
market = pd.read_csv("market_list.csv")

# The five evaluation markets are fixed by the table above.
evaluation = ["United States", "South Africa", "South Korea", "Indonesia", "Russia"]
eligible = market[market["validation_origins_2000_2009"] > 0].copy()
remaining = eligible[~eligible["series_id"].isin(evaluation)].to_dict("records")
assert len(eligible) == 41 and len(remaining) == 36

# Collapse European subregions only for the region-balancing algorithm.
def macro(region):
    return "Europe" if region.endswith("Europe") else region

lookup = eligible.set_index("series_id").to_dict("index")

# Order 1: region rotation; deterministic region and within-region tie breaks.
region_ties = ["Oceania", "Latin America", "Middle East", "Europe",
               "Asia", "North America", "Africa"]
counts = Counter(macro(lookup[m]["region"]) for m in evaluation)
by_region = {}
for row in remaining:
    by_region.setdefault(macro(row["region"]), []).append(row)
for rows in by_region.values():
    rows.sort(key=lambda r: (-r["train_samples_first_fold"],
                             -r["n_quarters"], r["series_id"]))
region_order = []
while any(by_region.values()):
    available = [r for r in region_ties if by_region.get(r)]
    region = min(available, key=lambda r: (counts[r], region_ties.index(r)))
    chosen = by_region[region].pop(0)["series_id"]
    region_order.append(chosen)
    counts[region] += 1

# Order 2: longest HPI coverage first, with explicit tie breaks.
history_order = [r["series_id"] for r in sorted(
    remaining,
    key=lambda r: (-r["n_quarters"], -r["folds_with_training_data"],
                   -r["train_samples_first_fold"], r["series_id"])
)]

# Order 3: seeded permutation; sort IDs before the draw.
random_order = list(np.random.default_rng(SEED).permutation(
    sorted(r["series_id"] for r in remaining)
))

orders = {"1 - Region rotation": region_order,
          "2 - Long-history first": history_order,
          "3 - Seeded random": random_order}

for name, order in orders.items():
    assert len(order) == 36 and len(set(order)) == 36
    assert set(order) == {r["series_id"] for r in remaining}
    print("\n" + name)
    print("Added to reach 10:", ", ".join(order[:5]))
    print("Added to reach 20:", ", ".join(order[5:15]))
    print("Added at all stage:", ", ".join(order[15:]))
    print("Pool sizes:", len(evaluation + order[:5]),
          len(evaluation + order[:15]), len(evaluation + order))
```

</details>

## 3. Leave-one-market-out targets (3–5)

| Market | History group | Region | Reason |
|---|---|---|---|
| Australia | Long (by 1979) | Oceania | HPI begins 1970Q1; 40 validation origins and 10 usable training folds provide long-history Oceania representation. |
| Portugal | Medium (1980–1999) | Southern Europe | HPI begins 1988Q1; 40 validation origins and 10 usable training folds provide medium-history Southern Europe representation. |
| Chile | Short (2000+) | Latin America | HPI begins 2002Q1; 24 validation origins and 5 usable training folds provide short-history Latin America representation. |
| Taiwan | Short (2000+) | Asia | Project-designated transfer target with HPI from 2001Q1, 28 validation origins, and 6 training folds; its nominal HPI requires CPI deflation. |

Each LOMO target is excluded from cross-market pretraining for its held-out run; any permitted target-market fine-tuning uses only historical pre-test observations.
