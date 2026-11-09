# HPI evidence audit — 2026-11-09

**Scope:** HPI-only open items in `TODO_B.md`: release timing (41 validation markets), Taiwan HPI timing, and large quarterly real-HPI log changes (15 flagged markets from the broader 48-market assessment). This is an evidence audit, **not** a claim that first OECD AnaHPI vintages or every historical outlier have been verified.

## 1. Publication timing — evidence versus model assumptions

- The existing `docs/hpi_release_lags.csv` contains **41** market rows: **8** `release_example` entries, all **27–85 calendar days**, and **33** `unverified` entries without supported numerical `lag_days`.
- The two experiment TOMLs currently assume `hpi = { default = 100, Taiwan = 60 }`. The eight examples do **not** validate the 100-day bound for the remaining markets or for historical OECD *real* AnaHPI vintages.
- For **Bulgaria**, [official NSI HPI metadata](https://www.nsi.bg/en/metadata/house-price-index-hpi-227) says first results are published approximately **85 days after quarter-end**. Its source coverage and compilation have changed over time: 2005–2008 estimates, 2009–2014 preliminary data, regular harmonised HPI since 2015, and a source change in 2022. This is useful **nominal national HPI schedule evidence**, not an independently confirmed earliest release of the corresponding *real, seasonally adjusted* OECD dataset. Do not fill `lag_days` with 85 as `verified` without source matching.
- For **Poland**, the [Statistics Poland residential-premises index description](https://stat.gov.pl/en/metainformation/description-of-economic-indicators/description-of-economic-indicators-cso/price-index-of-residential-premises/) is a candidate methodology source; its publication-timeliness wording could not be directly retrieved during this audit. **Do not assign a numerical Polish lag** or treat a four-month upper limit as independently verified. Match the national source and earliest national/OECD real-series publication separately.
- For European countries, [Eurostat's 2025Q2 release on 2025-10-03](https://ec.europa.eu/eurostat/en/web/products-eurostat-news/w/ddn-20251003-1) establishes availability of a nominal EU HPI publication **95 days after 2025Q2**, not the earliest individual national release and not the OECD real data vintage.
- For **Taiwan**, the [2025Q2 notice](https://www.ncscre.nccu.edu.tw/node/8689) (2025-07-21, 21 days after quarter-end) and [2025Q3 notice](https://www.ncscre.nccu.edu.tw/node/8956) (2025-10-27, 27 days) are dated publisher examples. Neither establishes first publication or historical 2000–2009 timing; retain the provisional Taiwan 60-day setting.

**Decision:** No automatic config change or model rerun is supported by this audit. Investigate Poland's exact-source historic timing and OECD real-series availability before treating the 100-day assumption as adequate. A historical/source-matched lag over the configured bound should trigger coordinated TOML changes, reconstruction of the information set, and reruns of affected experiments.

## 2. All 15 flagged markets — review register

Values below are **project-screening values**, not independently reconstructed values. They are absolute quarterly **log changes of the OECD real, seasonally adjusted HPI**. The reported largest change is included as a priority case for each market.

| Market | Largest flagged quarter | Log-QoQ change | Evidence check and decision |
|---|---|---:|---|
| South Africa | 1968Q3 | +10.37% | Original pre-1970 national source and index construction not established; retain and investigate |
| United Kingdom | 1972Q3 | +10.08% | Historical original series and real-deflator / seasonal-adjustment contribution not established |
| Ireland | 1979Q2 | +10.72% | Early historical source reconstruction and revisions need verification |
| Italy | 1974Q1 | +14.15% | Validate early historical nominal source/deflator, then compare nearby 1973Q4 and 1974Q2 |
| New Zealand | 1974Q1 | +10.78% | Original nominal source, CPI deflator and seasonal methodology require verification |
| Spain | 1987Q1 | +10.83% | Original pre-harmonisation national source and backfilled OECD series require verification |
| Colombia | 1988Q3 | +10.45% | Original index definition and earliest coverage require verification |
| Russia | 2011Q1 | -14.63% | Compare original Russian housing series and OECD deflation/SA, including 2006Q2 and 2024Q1 |
| Bulgaria | 2020Q4 | +15.50% | [NSI methodology](https://www.nsi.bg/en/metadata/house-price-index-hpi-227) confirms historical source/coverage changes, including **2022**. No official point-by-point validation of the **2020Q4** +15.50% real-SA jump yet; retain flag |
| Estonia | 2009Q1 | -22.25% | [Statistics Estonia's retrospective annual series](https://www.stat.ee/en/news/the-dwelling-price-index-rose-by-52-in-2025) reports a **-37.2% nominal year-on-year price change in 2009**; this independently supports a severe downturn but does **not** verify the exact Q1 2009 OECD real-SA -22.25% value. Retain observation and annotate |
| Poland | 2006Q3 | +21.86% | [Statistics Poland metadata](https://stat.gov.pl/en/metainformation/description-of-economic-indicators/description-of-economic-indicators-cso/price-index-of-residential-premises/) describes the modern transaction-based index, but not the specific OECD **2006Q3** historical backfill. Unresolved |
| Latvia | 2009Q1 | -19.02% | [CSB HPI description](https://stat.gov.lv/en/statistics-themes/economy/consumer-prices/press-releases/22927-changes-dwelling-prices-1st-quarter) confirms quality-adjusted transaction-price coverage; no archival 2008–09 quarterly source match yet |
| Lithuania | 2009Q1 | -24.63% | Exact original national HPI time series and OECD deflation/SA match still required |
| Romania | 2010Q1 | -11.10% | Part of the **48-market data-quality audit**, but not the 41 validation-market lag table; verify historical source before interpreting |
| Türkiye | 2022Q3 | +13.38% | Part of the **48-market data-quality audit**, but not the 41 validation-market lag table; confirm real deflator and original index first |

**Coverage clarification:** The 15 outlier markets come from `docs/national_hpi_assessment_48countries.csv`, whereas `docs/hpi_release_lags.csv` contains **41** evaluation markets. Romania and Türkiye are not among those 41, so do not conflate these counts.

**Current evidence status:** The 2009 housing downturn in Estonia is corroborated at the *annual national nominal HPI* level; the specific 2009Q1 OECD real-SA log change remains unverified. Bulgaria's documented methodology changes demonstrate the need for break checks. For the other cases, the current official sources do not yet verify every flagged quarter or distinguish genuine market moves from source revisions, seasonal adjustment and deflation. **Do not delete, winsorize or exclude** flagged observations on this evidence alone.

## 3. Exact checks required for closure

1. **Release lag:** For each unverified market, identify original national index used by OECD, determine when each quarter's observation was first released (not current webpage update), and determine when the inflation-adjusted OECD series first became available in the validation period.
2. **Outliers:** For *every flagged quarter*, compare three values where available: original national nominal HPI change, inflation deflator change, and OECD real seasonally adjusted HPI log change. Record methodological breaks and revisions, keeping the quarter identifier and source URL.
3. **Timing and experiment sensitivity:** Treat `default=100` and `Taiwan=60` as assumptions pending historical confirmation. Change both TOML configs only on adequate corresponding evidence and rerun dependent backtests if information-set dates change.
4. **Documentation:** Keep `TODO_B.md` at `Partial` for the outlier investigation and HPI release verification, with this audit as evidence of progress; no item is `Verified` for the whole historical scope.

## Files and references

- `docs/national_hpi_assessment_48countries.csv` — 15 screening flags and quarter-level notes.
- `docs/market_list.csv` — 41 evaluation markets and their validated training-window coverage.
- `docs/hpi_release_lags.csv` — per-market lag table (8 release examples, 33 unverified).
- `docs/hpi_release_lags.md` — statuses, definitions and evidence requirements.
- `configs/experiment.toml`, `configs/experiment_long.toml` — current lag assumptions.

**Do not mistake** nominal national HPI dates or current OECD observations for the first historical release of the exact OECD real AnaHPI series.
