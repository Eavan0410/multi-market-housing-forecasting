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


EVALUATION_MARKETS = ("United States", "South Africa", "South Korea", "Indonesia", "Russia")


def macro(region):
    return "Europe" if region.endswith("Europe") else region


def market_orders(market):
    eligible = market[market["validation_origins_2000_2009"] > 0].copy()
    remaining = eligible[~eligible["series_id"].isin(EVALUATION_MARKETS)].to_dict("records")
    if eligible["series_id"].nunique() != 41 or len(eligible) != 41 or len(remaining) != 36:
        raise ValueError("expected 41 unique markets including the five evaluation markets")

    lookup = eligible.set_index("series_id").to_dict("index")

    region_ties = ["Oceania", "Latin America", "Middle East", "Europe",
                   "Asia", "North America", "Africa"]
    counts = Counter(macro(lookup[m]["region"]) for m in EVALUATION_MARKETS)
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

    history_order = [r["series_id"] for r in sorted(
        remaining,
        key=lambda r: (-r["n_quarters"], -r["folds_with_training_data"],
                       -r["train_samples_first_fold"], r["series_id"])
    )]

    random_order = list(np.random.default_rng(SEED).permutation(
        sorted(r["series_id"] for r in remaining)
    ))

    orders = {"1 - Region rotation": region_order,
              "2 - Long-history first": history_order,
              "3 - Seeded random": random_order}
    return orders


def main():
    orders = market_orders(pd.read_csv(MARKET_LIST_PATH))
    for name, order in orders.items():
        assert len(order) == 36 and len(set(order)) == 36
        print("\n" + name)
        print("Added to reach 10:", ", ".join(order[:5]))
        print("Added to reach 20:", ", ".join(order[5:15]))
        print("Added at all stage:", ", ".join(order[15:]))
        print("Pool sizes:", len(EVALUATION_MARKETS) + len(order[:5]),
              len(EVALUATION_MARKETS) + len(order[:15]), len(EVALUATION_MARKETS) + len(order))


if __name__ == "__main__":
    main()
