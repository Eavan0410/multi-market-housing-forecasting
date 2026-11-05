
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261104

MARKET_LIST_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "market_list.csv"
)


market = pd.read_csv(MARKET_LIST_PATH)

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
