"""CPI coverage and outlier check for the 41 forecast markets (the CPI version of
docs/national_hpi_assessment_48countries.csv).

    python scripts/build_cpi_assessment.py

Writes docs/cpi_assessment.csv, one row per market. Flags are screening marks only; no data
are changed or removed.

Rules (on year-on-year CPI inflation in percent, at the native frequency):
  big_jumps       d = yoy_t - yoy_{t-1}; flag when |d - median(d)| > 5 * MAD(d), with MAD
                  scaled by 1.4826 and computed on that market's own history. Comparing a
                  market with itself means high-inflation countries are not flagged for
                  being high, only for sudden changes.
  extreme_values  yoy > 100 or yoy < -5, the same absolute rule for every market. These values
                  would dominate the input scaling, which is shared by all markets.
Each flag is split into "in HPI period" (quarters the forecasting data can use) and earlier.
likely_causes_in_hpi_period matches flag dates to known episodes (oil shocks, 2008, the
2021-23 energy shock, Russia 2014-15 and 2022). It is a date match, not a verified cause;
anything outside those windows is counted as "not identified".

cpi_quarters_lost: HPI quarters whose input window cannot have CPI because CPI starts later.
Training-sample counts per fold come from build_market_list.py with a CPI config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from housing_forecast.data.cpi import load_cpi_raw, native_yoy, quarterly_yoy  # noqa: E402
from housing_forecast.data.readers import load_taiwan_cpi_monthly  # noqa: E402

JUMP_K = 5.0
EXTREME_HIGH, EXTREME_LOW = 100.0, -5.0

# Causes checked by hand for flags inside the HPI period.
CAUSES = {
    "Ireland": "Extreme values: 2009 deflation after the financial crisis. Before 1997 the OECD series mixes quarterly and monthly observations, so some 1970s-80s jumps compare months three months apart. 1975 has no complete quarter.",
    "Japan": "OECD's Japan CPI stops in 2021-06 (the four_countries copy too). scripts/download_cpi_gaps.py adds the BIS long CPI series, which runs to date.",
    "New Zealand": "Quarterly index only (FRED, OECD archive); main file starts 1990. Run scripts/download_cpi_gaps.py for the full history.",
    "Russia": "BIS long CPI; main file starts 2002 while HPI starts 2001Q1. Run scripts/download_cpi_gaps.py (from 1995).",
    "Australia": "Quarterly CPI stored as one month per quarter (Feb/May/Aug/Nov); read as quarterly.",
    "Chile": "BIS long CPI (OECD file has no Chile). Year-on-year rates after 2019 computed from the official BIS index.",
    "Mexico": "BIS long CPI (OECD file has no Mexico). Year-on-year rates after 2019 computed from the official BIS index.",
    "Taiwan": "DGBAS monthly CPI index (data/raw/taiwan); year-on-year rate computed from the index.",
}


EPISODES = [  # (first year, last year, label) for flags inside the HPI period
    (1973, 1976, "1973-74 oil shock"),
    (1979, 1983, "1979-80 oil shock and disinflation"),
    (1990, 1991, "1990 Gulf War oil spike"),
    (2008, 2009, "2008 commodity spike and crisis"),
    (2021, 2023, "2021-23 energy shock and its base effect"),
]
COUNTRY_EPISODES = {"Russia": [(2014, 2016, "2014-15 rouble collapse"), (2022, 2023, "2022 sanctions")]}


def likely_causes(country: str, flagged: pd.Series) -> str:
    years = [p.year for p in flagged.index]
    found, unknown = [], 0
    for y in years:
        label = next((lab for a, b, lab in COUNTRY_EPISODES.get(country, []) + EPISODES if a <= y <= b), None)
        if label is None:
            unknown += 1
        elif label not in found:
            found.append(label)
    return "; ".join(found + ([f"{unknown} not identified"] if unknown else []))


def jump_flags(s: pd.Series) -> pd.Series:
    """Robust jumps on changes between adjacent periods of one series."""
    ordinal = pd.Series([p.ordinal for p in s.index], index=s.index)
    d = s.diff()[ordinal.diff().eq(1)].dropna()
    if len(d) < 12:
        return d.iloc[0:0]
    med = d.median()
    mad = 1.4826 * (d - med).abs().median()
    if mad == 0:
        return d.iloc[0:0]
    return d[(d - med).abs() > JUMP_K * mad]


def fmt(s: pd.Series, unit: str = "", limit: int = 12) -> str:
    items = [f"{p} ({v:+.1f}{unit})" for p, v in s.items()]
    return " ".join(items[:limit]) + (f" ... (+{len(items) - limit} more)" if len(items) > limit else "")


def taiwan_native() -> pd.DataFrame:
    tw = load_taiwan_cpi_monthly("data/raw/taiwan/taiwan_cpi_monthly_raw.csv")
    s = tw.set_index(pd.PeriodIndex(tw["month"], freq="M"))["cpi"].astype(float)
    s = ((s / s.shift(12) - 1) * 100).dropna()
    return pd.DataFrame({"country": "Taiwan", "freq": "M", "period": s.index, "cpi_yoy": s.to_numpy(),
                         "from_index": True, "source": "DGBAS (Taiwan)"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", default="docs/hpi_release_lags.csv", help="the 41 forecast markets")
    ap.add_argument("--market-list", default="docs/market_list.csv", help="HPI start and end per market")
    ap.add_argument("--out", default="docs/cpi_assessment.csv")
    args = ap.parse_args()

    markets = pd.read_csv(args.markets)[["series_id", "country"]]
    hpi = pd.read_csv(args.market_list).set_index("series_id")
    native = pd.concat([native_yoy(load_cpi_raw()), taiwan_native()], ignore_index=True)
    quarterly = quarterly_yoy(native)

    rows = []
    for sid, country in markets.itertuples(index=False):
        h0 = pd.Period(hpi.loc[sid, "hpi_start"], "Q")
        h1 = pd.Period(hpi.loc[sid, "hpi_end"], "Q")
        n = native[native["country"].eq(country)]
        q = quarterly[quarterly["country"].eq(country)]
        if q.empty:
            rows.append({"series_id": sid, "country": country, "cpi_source": "missing",
                         "notes": CAUSES.get(country, "No CPI found.")})
            continue
        qp = pd.PeriodIndex(q["quarter"], freq="Q")
        span = pd.period_range(qp.min(), qp.max(), freq="Q")
        # The first usable HPI origin needs CPI from the quarter after HPI starts (first growth rate).
        lost = max(0, (qp.min() - (h0 + 1)).n)

        # Jumps are measured between adjacent periods only. Monthly rows spaced three months
        # apart (quarterly CPI stored as months) are checked as a quarterly series.
        parts = []
        for freq, g in n.groupby("freq"):
            s = g.set_index("period")["cpi_yoy"].sort_index()
            if freq == "M":
                step = pd.Series([p.ordinal for p in s.index], index=s.index)
                dense = (step.diff().eq(1) | step.diff(-1).eq(-1))
                sparse = s[~dense]
                if len(sparse):
                    sq = pd.Series(sparse.to_numpy(), index=pd.PeriodIndex([p.asfreq("Q") for p in sparse.index], freq="Q"))
                    parts.append(("Q", sq[~sq.index.duplicated()]))
                s = s[dense]
            parts.append((freq, s))
        jumps = pd.concat([jump_flags(s) for _, s in parts]) if parts else pd.Series(dtype=float)
        extreme = pd.concat([s[(s > EXTREME_HIGH) | (s < EXTREME_LOW)] for _, s in parts])

        def in_hpi(s: pd.Series) -> pd.Series:
            qs = pd.PeriodIndex([p.asfreq("Q") for p in s.index], freq="Q") if len(s) else pd.PeriodIndex([], freq="Q")
            return s[(qs >= h0) & (qs <= h1)]

        jh, eh = in_hpi(jumps), in_hpi(extreme)
        freqs = sorted(n["freq"].unique())
        rows.append({
            "series_id": sid,
            "country": country,
            "cpi_source": n["source"].iloc[0],
            "frequency": " + ".join({"M": "Monthly", "Q": "Quarterly"}[f] for f in freqs)
                         + (" (one month per quarter in some years)" if country in ("Australia", "Ireland") else ""),
            "cpi_start": str(qp.min()),
            "cpi_end": str(qp.max()),
            "n_quarters": len(qp),
            "missing_quarters_inside": len(span) - len(qp),
            "yoy_from_index_share": round(float(n["from_index"].mean()), 2),
            "hpi_start": str(h0),
            "hpi_end": str(h1),
            "cpi_quarters_lost": lost,
            "cpi_ends_before_hpi": bool(qp.max() < h1),
            "max_yoy_in_hpi_period": round(float(in_hpi(pd.concat([s for _, s in parts])).max()), 1),
            "n_big_jumps_in_hpi_period": len(jh),
            "big_jumps_in_hpi_period": fmt(jh, "pp"),
            "likely_causes_in_hpi_period": likely_causes(country, pd.concat([jh, eh])),
            "n_big_jumps_before_hpi": len(jumps) - len(jh),
            "n_extreme_in_hpi_period": len(eh),
            "extreme_values_in_hpi_period": fmt(eh, "%"),
            "n_extreme_before_hpi": len(extreme) - len(eh),
            "notes": CAUSES.get(country, ""),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(out[["series_id", "cpi_source", "cpi_start", "cpi_end", "cpi_quarters_lost", "cpi_ends_before_hpi",
               "max_yoy_in_hpi_period", "n_big_jumps_in_hpi_period", "n_extreme_in_hpi_period"]].to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
