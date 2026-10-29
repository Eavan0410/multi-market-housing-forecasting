"""Rebuild the four-country availability matrix from original observations.

Install dependencies: python3 -m pip install pandas openpyxl
Run: python3 scripts/build_availability_matrix.py

The only generated files are results/availability_matrix.csv and
results/availability_matrix.xlsx. Rerunning replaces those two outputs.
No processed data, frequency conversion, filling or scaling is used.

Coverage means distinct source months or quarters containing finite numbers.
Monthly complete_quarters requires all three months; directly reported
quarterly complete_quarters simply counts valid quarterly observations.
Common coverage uses the intersection of observed periods in all four markets.

Public-source metadata was checked on 2026-10-28. Release examples below are
explicitly scoped to a reference period. News releases and national schedules
do not establish first OECD database availability or constant historical lags.
The raw files contain no release-date/vintage fields; no last-update timestamp
is used as a first-release date. Research included the OECD calendar, dataset
pages, indicator/methodology pages and the OECD API documentation:
https://www.oecd.org/en/data/insights/statistical-releases/release-dates-for-oecd-statistics-news-releases.html
https://www.oecd.org/en/data/insights/data-explainers/2024/09/api.html
"""

import argparse
from decimal import Decimal
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = {
    "AUS": "Australia",
    "USA": "United States",
    "KOR": "South Korea",
    "JPN": "Japan",
}
VARIABLES = [
    "hpi", "cpi_yoy", "gdp_growth", "household_credit_gdp",
    "interest_rate", "real_income_growth", "unemployment_rate",
]
FREQUENCIES = {"M": "Monthly", "Q": "Quarterly"}
COLUMNS = [
    "country", "variable", "frequency", "first_period", "last_period",
    "valid_observations", "gaps_inside_span", "complete_quarters",
    "common_availability", "common_valid_observations", "unit",
    "real_nominal", "seasonal_adj", "transformation", "variable_definition",
    "source", "publication_lag", "publication_lag_source", "publication_lag_status",
]
CREDIT_SERIES = {
    "AUS": "QAUHAM770A", "USA": "QUSHAM770A",
    "KOR": "QKRHAM770A", "JPN": "QJPHAM770A",
}
INTEREST_REFERENCE = "https://www.oecd.org/en/data/indicators/long-term-interest-rates.html"
HOUSING_REFERENCE = "https://www.oecd.org/en/data/indicators/housing-prices.html"
HOUSING_FAQ = (
    "https://www.oecd.org/en/data/insights/data-explainers/2024/07/"
    "Residential-Property-Price-Indices-and-related-housing-indicators-Frequently-Asked-Questions.html"
)

# These mappings apply to the actual raw dimension codes, not filename guesses.
UNITS = {"IX": "Index points", "PA": "% per annum", "PC": "%", "PT_LF_SUB": "% of labour force"}
ADJUSTMENTS = {"S": "SA", "Y": "SA", "N": "NSA", "_Z": "Not applicable"}
REAL_BASES = {"L", "LR", "V", "PYP"}

# Only release examples whose country and reference period were checked are
# used. The G20 GDP release reports individual country observations, not just
# the aggregate; each of the four countries appears in its Q2 2025 discussion.
GDP_NEWS = (
    "https://www.oecd.org/en/data/insights/statistical-releases/2025/09/"
    "g20-gdp-growth-second-quarter-2025.html"
)
UNEMPLOYMENT_NEWS = (
    "https://www.oecd.org/content/dam/oecd/en/data/insights/statistical-releases/"
    "2025/10/labour-market-situation-oecd-10-2025.pdf"
)
CPI_NEWS = (
    "https://www.oecd.org/content/dam/oecd/en/data/insights/statistical-releases/"
    "2025/11/consumer-prices-oecd-11-2025.pdf"
)
INCOME_NEWS = (
    "https://www.oecd.org/content/dam/oecd/en/data/insights/statistical-releases/"
    "2025/8/Growth-and-economic-well-being-oecd-08-2025.pdf"
)
PUBLICATION_EXAMPLES = {}
for country_code in COUNTRIES:
    PUBLICATION_EXAMPLES[(country_code, "gdp_growth", "Q")] = (
        "2025Q2", "2025-09-15", GDP_NEWS, "OECD news-release proxy", "OECD news release",
    )
    PUBLICATION_EXAMPLES[(country_code, "unemployment_rate", "M")] = (
        "2025-08", "2025-10-16", UNEMPLOYMENT_NEWS,
        "OECD news-release proxy", "OECD news release",
    )
for country_code in ["USA", "KOR"]:
    PUBLICATION_EXAMPLES[(country_code, "cpi_yoy", "M")] = (
        "2025-09", "2025-11-05", CPI_NEWS, "OECD news-release proxy", "OECD news release",
    )
PUBLICATION_EXAMPLES[("USA", "real_income_growth", "Q")] = (
    "2025Q1", "2025-08-07", INCOME_NEWS, "OECD news-release proxy", "OECD news release",
)
PUBLICATION_EXAMPLES[("AUS", "cpi_yoy", "Q")] = (
    "2025Q3", "2025-10-29",
    "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/sep-quarter-2025",
    "Official schedule estimate", "ABS national release",
)
PUBLICATION_EXAMPLES[("JPN", "cpi_yoy", "M")] = (
    "2021-06", "2021-07-20", "https://www.stat.go.jp/english/info/news/20210831.html",
    "Official schedule estimate", "Statistics Bureau national release",
)
# Research exclusions are intentional:
# - OECD housing metadata gives definitions, not historical first-seen dates.
# - The IRLT definition identifies 10-year bonds but gives no publication lag.
# - The BIS calendar/FAQ describes quarterly credit releases, but does not
#   establish historical first FRED availability of these local observations.
#   https://data.bis.org/release-calendar
#   https://data.bis.org/faq?stats-faqs=7
# - The August 2025 income release lacks current Japanese income and does not
#   present an Australian observation; its release date is not assigned to them.
# - The November 2025 CPI release uses COICOP 2018 for Australia/Japan, whereas
#   the local export is COICOP 1999. No date is transferred to those raw series.
# - No quarterly CPI availability date is inferred from monthly publication.


def metadata_value(frame, column):
    """Read one series' documented metadata; stop if definitions conflict."""
    if column not in frame:
        return "Unknown"
    values = frame[column].astype(str).str.strip()
    values = values[values.ne("")].unique()
    if len(values) > 1:
        raise ValueError(f"Conflicting series metadata in {column}: {list(values)}")
    return values[0] if len(values) else "Unknown"


def read_raw(variable, raw_dir):
    """Step 1: read the original layout and retain only the four country codes."""
    path = raw_dir / f"{variable}_raw.csv"
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    code_column = "country_code" if variable == "household_credit_gdp" else "REF_AREA"
    raw = raw.loc[raw[code_column].isin(COUNTRIES)].copy()
    raw["country_code"] = raw[code_column]

    if variable == "household_credit_gdp":
        # Each credit row has one country-specific value column selected by
        # series_id. Blanks in other countries' columns are structural blanks.
        expected = raw["country_code"].map(CREDIT_SERIES)
        if not raw["series_id"].eq(expected).all():
            raise ValueError("Credit series_id does not match the country code.")
        raw["raw_value"] = ""
        for code, series in CREDIT_SERIES.items():
            selected = raw["country_code"].eq(code)
            other_series = [name for name in CREDIT_SERIES.values() if name != series]
            if raw.loc[selected, other_series].ne("").any().any():
                raise ValueError("Unexpected credit values in another country's column.")
            raw.loc[selected, "raw_value"] = raw.loc[selected, series]
        dates = pd.to_datetime(raw["observation_date"], format="%Y-%m-%d", errors="raise")
        if not (dates.dt.is_quarter_start).all():
            raise ValueError("Unexpected credit observation-date convention.")
        # Official FRED series pages confirm quarterly end-of-quarter series;
        # their CSV observation_date labels the quarter using its first day.
        raw["FREQ"] = "Q"
        raw["period"] = dates.dt.to_period("Q").astype(str)
    else:
        if not raw["FREQ"].isin(FREQUENCIES).all():
            raise ValueError(f"Unsupported source frequency in {path.name}.")
        raw["raw_value"] = raw["OBS_VALUE"]
        raw["period"] = ""
        for frequency in raw["FREQ"].unique():
            selected = raw["FREQ"].eq(frequency)
            periods = raw.loc[selected, "TIME_PERIOD"]
            pattern = r"\d{4}-\d{2}" if frequency == "M" else r"\d{4}-Q[1-4]"
            if not periods.str.fullmatch(pattern).all():
                raise ValueError(f"Invalid source period in {path.name} ({frequency}).")
            if frequency == "M":
                dates = pd.to_datetime(periods, format="%Y-%m", errors="raise")
                raw.loc[selected, "period"] = dates.dt.to_period("M").astype(str)
            else:
                raw.loc[selected, "period"] = periods.str.replace("-", "", regex=False)

    numeric = pd.to_numeric(raw["raw_value"], errors="coerce")
    raw["valid"] = numeric.notna() & ~numeric.isin([float("inf"), -float("inf")])
    # Zero and negative numbers remain valid. Do not turn blanks into zero.
    return raw


def valid_periods(group):
    """Step 2: deduplicate periods and flag conflicting values explicitly."""
    valid = group.loc[group["valid"]]
    duplicates = valid.loc[valid.duplicated("period", keep=False)]
    for period, rows in duplicates.groupby("period"):
        # Decimal avoids hiding small conflicts through float rounding;
        # numerically identical strings such as 1 and 1.0 count only once.
        values = {Decimal(value.strip()) for value in rows["raw_value"]}
        if len(values) > 1:
            code = rows["country_code"].iloc[0]
            raise ValueError(f"Conflicting duplicate observations for {code}, {period}: {values}")
    return set(valid["period"])


def coverage_statistics(periods, frequency):
    """Step 3: count observed periods, internal gaps and complete quarters."""
    if not periods:
        return {"first_period": "", "last_period": "", "valid_observations": 0,
                "gaps_inside_span": 0, "complete_quarters": 0}
    ordered = sorted(periods)
    span_length = len(pd.period_range(ordered[0], ordered[-1], freq=frequency))
    if frequency == "M":
        quarter_labels = pd.PeriodIndex(ordered, freq="M").asfreq("Q").astype(str)
        month_counts = pd.Series(quarter_labels).value_counts()
        complete = int(month_counts.eq(3).sum())
    else:
        # Direct quarterly values do not imply checking three monthly inputs.
        complete = len(ordered)
    return {
        "first_period": ordered[0], "last_period": ordered[-1],
        "valid_observations": len(ordered),
        "gaps_inside_span": span_length - len(ordered), "complete_quarters": complete,
    }


def measurement_metadata(variable, group, code):
    """Step 4: separate units, price basis, adjustment and transformation."""
    filename = f"data/raw/{variable}_raw.csv"
    if group.empty:
        return {
            "unit": "Unknown", "real_nominal": "Unknown", "seasonal_adj": "Unknown",
            "transformation": "Unknown",
            "variable_definition": "No observations supplied for this country; measurement metadata unavailable.",
            "source": f"{filename}; no {code} observations",
        }
    if variable == "household_credit_gdp":
        series = metadata_value(group, "series_id")
        return {
            "unit": "% of GDP", "real_nominal": "Not applicable", "seasonal_adj": "NSA",
            "transformation": "Ratio to GDP",
            "variable_definition": (
                "Total credit to households and NPISHs, adjusted for breaks; end-of-quarter "
                "outstanding credit as a percentage of GDP. Source dates label quarter starts."
            ),
            "source": f"BIS via FRED; {series}; https://fred.stlouisfed.org/series/{series}; {filename}",
        }

    # Check dimensions separately for each country-frequency group. Multiple
    # definitions cannot silently become one row merely because values match.
    for column in ["MEASURE", "TRANSACTION", "PRICE_BASE", "UNIT_MULT", "BASE_PER"]:
        metadata_value(group, column)
    measure = metadata_value(group, "MEASURE")
    unit_code = metadata_value(group, "UNIT_MEASURE")
    unit = UNITS.get(unit_code, metadata_value(group, "Unit of measure"))
    seasonal = ADJUSTMENTS.get(metadata_value(group, "ADJUSTMENT"), "Unknown")
    transform_code = metadata_value(group, "TRANSFORMATION")
    transform = {"GY": "Year-over-year percentage change", "G1": "Quarter-over-quarter percentage change"}.get(transform_code, "Unknown")
    basis = metadata_value(group, "PRICE_BASE")
    real_nominal = "Real" if basis in REAL_BASES else "Unknown"
    definition = metadata_value(group, "Measure")

    if variable == "hpi":
        if measure != "RHP":
            raise ValueError(f"Expected real HPI (RHP), found {measure} for {code}.")
        real_nominal, transform = "Real", "Level"
        base = metadata_value(group, "BASE_PER")
        unit = f"Index points (base period {base})" if base != "Unknown" else "Index points"
        definition = "Real house price index: nominal house prices divided by the household consumption expenditure deflator."
    elif variable == "cpi_yoy":
        real_nominal = "Not applicable"
        definition = "National all-items consumer price index; growth over one year; COICOP 1999 source export."
    elif variable == "gdp_growth":
        definition = f"{metadata_value(group, 'Transaction')}; {metadata_value(group, 'Price base')}; {metadata_value(group, 'Transformation')}."
    elif variable == "interest_rate":
        if measure != "IRLT":
            # A policy-rate substitution or other unexpected measure is an
            # explicit error, never relabeled as a 10-year government yield.
            raise ValueError(f"Interest-rate mismatch for {code}: expected IRLT, found {measure}.")
        transform = "Level (monthly average yield)"
        # The local metadata and the cited indicator definition do not state a
        # real/nominal price-basis code. Leave that classification Unknown.
        real_nominal = "Unknown"
        definition = "10-year government bond yield (long-term interest rate; not policy rate)"
    elif variable == "real_income_growth":
        definition = f"{definition}; {metadata_value(group, 'Transformation')}; includes households and NPISHs."
    elif variable == "unemployment_rate":
        real_nominal, transform = "Not applicable", "Level"
        definition = (
            f"{definition}; {metadata_value(group, 'Sex')}; {metadata_value(group, 'Age')}; "
            "percentage of labour force."
        )
    structure = metadata_value(group, "STRUCTURE_ID")
    agency, flow_version = structure.split(":", 1)
    flow, version = flow_version.removesuffix(")").split("(")
    data_url = f"https://data-explorer.oecd.org/vis?df[ag]={agency}&df[id]={flow}&df[vs]={version}"
    source = f"OECD; {structure}; {data_url}; {filename}"
    if variable == "interest_rate":
        source += f"; definition: {INTEREST_REFERENCE}"
    elif variable == "hpi":
        source += f"; definition: {HOUSING_REFERENCE}; metadata: {HOUSING_FAQ}"
    return {
        "unit": unit, "real_nominal": real_nominal, "seasonal_adj": seasonal,
        "transformation": transform, "variable_definition": definition, "source": source,
    }


def publication_metadata(code, variable, frequency, periods):
    """Step 5: attach traceable timing examples, with their precise limitations."""
    example = PUBLICATION_EXAMPLES.get((code, variable, frequency))
    if example is None or example[0] not in periods:
        # A generic calendar URL does not prove a numerical lag. Leave the
        # evidence cell blank when no supporting timing evidence was verified.
        return {"publication_lag": "Not verified", "publication_lag_source": "",
                "publication_lag_status": "Not verified"}
    period, release_date, url, status, provider = example
    period_end = pd.Period(period, freq=frequency).end_time.normalize()
    days = (pd.Timestamp(release_date) - period_end).days
    return {
        "publication_lag": (
            f"{days} calendar days ({provider} example: {period} -> {release_date}; "
            "first OECD database availability not verified; not a constant historical lag)"
        ),
        "publication_lag_source": url, "publication_lag_status": status,
    }


def build_matrix(raw_dir):
    """Build one row per country-variable-source-frequency, then intersect sets."""
    records = []
    for variable in VARIABLES:
        raw = read_raw(variable, raw_dir)
        # Frequencies must actually occur in the target countries' raw file.
        # An absent country's row (for example income growth) records zero
        # coverage, keeping all four markets visible for that observed group.
        for frequency in sorted(raw["FREQ"].unique()):
            groups = {
                code: raw.loc[raw["country_code"].eq(code) & raw["FREQ"].eq(frequency)]
                for code in COUNTRIES
            }
            periods_by_country = {code: valid_periods(group) for code, group in groups.items()}
            common = set.intersection(*periods_by_country.values())
            interval = f"{min(common)}–{max(common)}" if common else ""
            for code, group in groups.items():
                periods = periods_by_country[code]
                records.append({
                    "country": COUNTRIES[code], "variable": variable,
                    "frequency": FREQUENCIES[frequency],
                    **coverage_statistics(periods, frequency),
                    "common_availability": interval, "common_valid_observations": len(common),
                    **measurement_metadata(variable, group, code),
                    **publication_metadata(code, variable, frequency, periods),
                })
    matrix = pd.DataFrame(records, columns=COLUMNS)
    # Display countries in the requested order and preserve project feature names.
    order = {country: index for index, country in enumerate(COUNTRIES.values())}
    matrix = matrix.sort_values("country", key=lambda column: column.map(order), kind="stable").reset_index(drop=True)
    if set(matrix["country"]) != set(COUNTRIES.values()):
        raise ValueError("Unexpected country coverage.")
    if matrix.duplicated(["country", "variable", "frequency"]).any():
        raise ValueError("Duplicate matrix rows.")
    return matrix


def export_matrix(matrix, output_dir):
    """Step 6: write identical CSV/Excel data using one simply formatted sheet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "availability_matrix.csv"
    excel_path = output_dir / "availability_matrix.xlsx"
    matrix.to_csv(csv_path, index=False, encoding="utf-8")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        matrix.to_excel(writer, sheet_name="Availability Matrix", index=False)
        sheet = writer.sheets["Availability Matrix"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="244062")
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        widths = [20, 25, 13, 16, 16, 19, 18, 19, 28, 24, 32, 18, 17, 42, 75, 80, 80, 80, 28]
        for column, width in zip(sheet.columns, widths):
            sheet.column_dimensions[column[0].column_letter].width = width
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            sheet.row_dimensions[row[0].row].height = 85
        sheet.row_dimensions[1].height = 32
    # Keep blanks as blanks so absent periods and unknown URLs match exactly.
    csv_data = pd.read_csv(csv_path, keep_default_na=False)
    excel_data = pd.read_excel(excel_path, sheet_name="Availability Matrix", keep_default_na=False)
    pd.testing.assert_frame_equal(csv_data, excel_data)
    return csv_path, excel_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    matrix = build_matrix(args.raw_dir)
    paths = export_matrix(matrix, args.output_dir)
    print(f"Created {len(matrix)} rows for four countries, with {len(matrix.columns)} columns.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
