"""Build HPI and macro-variable coverage with source and publication-date evidence.

Count observed periods and complete quarters. Flag conflicting observations and
keep estimated release lags distinct from verified historical availability."""

import argparse
from decimal import Decimal
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COUNTRIES = {
    "AUS": "Australia",
    "USA": "United States",
    "KOR": "South Korea",
    "JPN": "Japan",
}
COUNTRIES = {**ORIGINAL_COUNTRIES, "TWN": "Taiwan"}
TAIWAN_VARIABLES = {"hpi", "cpi_yoy", "gdp_growth", "interest_rate", "household_credit_gdp"}
TAIWAN_FILES = {
    "hpi": "taiwan_hpi_quarterly_raw.csv",
    "cpi_yoy": "taiwan_cpi_monthly_raw.csv",
    "gdp_growth": "taiwan_gdp_growth_qoq_raw.csv",
    "interest_rate": "taiwan_interest_rate_raw.csv",
    "household_credit_gdp": "taiwan_household_credit_raw.csv",
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

# Official Taiwan evidence is recorded here so rebuilding needs no internet.
# These are dated examples or schedules, not one assumed lag for all history.
TAIWAN_GDP_REVISION = "https://eng.stat.gov.tw/News_Content.aspx?n=2317&s=235721"
TAIWAN_CREDIT_RELEASE = "https://www.cbc.gov.tw/tw/cp-302-185273-9a588-1.html"
TAIWAN_RATE_SCHEDULE = (
    "https://www.stat.gov.tw/News_NoticeCalendar_Content_temp.aspx?MetaI_D=218&n=3717&year=2026"
)
TAIWAN_RATE_SOURCE = "https://data.gov.tw/dataset/10840"
TAIWAN_CREDIT_SOURCE = "https://data.gov.tw/dataset/6549"
TAIWAN_HPI_SOURCE = "https://www.sinyinews.com.tw/quarterly"
TAIWAN_HPI_EXAMPLE = "https://www.sinyi.com.tw/expertise/house-price-index/40"
TAIWAN_HPI_NOTICES = [
    # NCCU is a co-producer. Its dated notices establish availability by these
    # dates; they do not prove that Sinyi had not published the values earlier.
    ("2025Q2", "164.39", "2025-07-21", "https://www.ncscre.nccu.edu.tw/node/8689"),
    ("2025Q3", "168.73", "2025-10-27", "https://www.ncscre.nccu.edu.tw/node/8956"),
]
TAIWAN_CPI_SCHEDULE = (
    "https://www.stat.gov.tw/News_NoticeCalendar_Content_temp.aspx?MetaI_D=166&n=3717&year=2026"
)
TAIWAN_CPI_RELEASE = "https://www.stat.gov.tw/News_Content.aspx?n=3703&s=235357"
TAIWAN_CPI_BOOK = "https://ws.dgbas.gov.tw/001/Upload/464/relfile/10862/235361/pricebook.pdf"

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
for country_code in ORIGINAL_COUNTRIES:
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
    """read the relocated raw files for the original four countries."""
    path = raw_dir / "four_countries" / f"{variable}_raw.csv"
    if not path.exists() and variable == "real_income_growth":
        # The merged remote commit deleted this source file. Keep the original
        # variable's quarterly rows visible with zero coverage; do not reuse
        # old matrix counts or processed income observations.
        return pd.DataFrame(columns=["country_code", "FREQ", "period", "raw_value", "valid"])
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    code_column = "country_code" if variable == "household_credit_gdp" else "REF_AREA"
    raw = raw.loc[raw[code_column].isin(ORIGINAL_COUNTRIES)].copy()
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


def taiwan_months(periods):
    """Parse source labels like Jul88 without relying on the machine's locale.

    Use the standard explicit 1969-2068 two-digit-year window: 88 means 1988,
    and 25 means 2025. No date rows or missing economic values are generated.
    """
    parts = periods.str.extract(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{2})$")
    if parts.isna().any().any():
        raise ValueError("Unexpected Taiwan monthly date format.")
    month_numbers = {name: index for index, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )}
    years = parts[1].astype(int)
    years = years.where(years.lt(69), years + 1900).where(years.ge(69), years + 2000)
    months = parts[0].map(month_numbers).astype(str).str.zfill(2)
    return years.astype(str) + "-" + months


def read_taiwan_file(raw_dir, filename, expected_feature, frequency):
    """Read the Taiwan raw layout and keep numeric validity explicit."""
    path = raw_dir / "taiwan" / filename
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not raw["REF_AREA"].eq("TWN").all():
        raise ValueError(f"Unexpected country in {filename}.")
    if not raw["feature"].eq(expected_feature).all() or not raw["FREQ"].eq(frequency).all():
        raise ValueError(f"Unexpected feature or frequency in {filename}.")
    raw["country_code"] = "TWN"
    if frequency == "M":
        raw["period"] = taiwan_months(raw["TIME_PERIOD"])
    else:
        if not raw["TIME_PERIOD"].str.fullmatch(r"\d{4}Q[1-4]").all():
            raise ValueError(f"Unexpected quarterly date in {filename}.")
        raw["period"] = raw["TIME_PERIOD"]
    raw["raw_value"] = raw["OBS_VALUE"]
    numeric = pd.to_numeric(raw["raw_value"], errors="coerce")
    raw["valid"] = numeric.notna() & ~numeric.isin([float("inf"), -float("inf")])
    return raw


def taiwan_observations(periods, values, frequency):
    """Put supplied cells into the shared layout without filling missing values."""
    raw = pd.DataFrame({"period": periods, "raw_value": values})
    raw["country_code"] = "TWN"
    raw["FREQ"] = frequency
    numeric = pd.to_numeric(raw["raw_value"], errors="coerce")
    raw["valid"] = numeric.notna() & ~numeric.isin([float("inf"), -float("inf")])
    return raw


def read_year_grid(path, labels, frequency):
    """Reshape a Year-by-month/quarter table, keeping each source cell intact.

    labels explicitly lists the observation columns. For CPI this deliberately
    excludes Index, which is an annual summary rather than another month.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if set(table.columns) != {"Year", *labels, *(["Index"] if frequency == "M" else [])}:
        raise ValueError(f"Unexpected columns in {path.name}.")
    if not table["Year"].str.fullmatch(r"\d{4}").all():
        raise ValueError(f"Unexpected year in {path.name}.")
    cells = table.melt(id_vars="Year", value_vars=labels, var_name="label", value_name="value")
    numbers = cells["label"].map({label: index for index, label in enumerate(labels, 1)})
    suffix = numbers.astype(str).str.zfill(2) if frequency == "M" else numbers.astype(str)
    periods = cells["Year"] + ("-" if frequency == "M" else "Q") + suffix
    return taiwan_observations(periods, cells["value"], frequency)


def read_taiwan_hpi(raw_dir):
    """Select Taiwan's HPI, never average city indices or count both files twice."""
    path = raw_dir / "taiwan" / TAIWAN_FILES["hpi"]
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not {"Period", "Year", "Season", "台灣"}.issubset(table.columns):
        raise ValueError("Taiwan HPI source lacks dates or its Taiwan column.")
    if not table["Period"].str.fullmatch(r"\d{4}/Q[1-4]").all():
        raise ValueError("Unexpected period format in Taiwan HPI source.")
    if not table["Period"].eq(table["Year"] + "/" + table["Season"]).all():
        raise ValueError("Taiwan HPI date columns disagree.")
    raw = taiwan_observations(table["Period"].str.replace("/", "", regex=False), table["台灣"], "Q")
    other = read_year_grid(raw_dir / "taiwan" / "taiwan_hpi_national_quarterly_raw.csv", ["Q1", "Q2", "Q3", "Q4"], "Q")
    # The wide file's blank 2025Q4 must not become an extra observation.
    values = []
    for frame in [raw, other]:
        valid_periods(frame)
        valid = frame.loc[frame["valid"]].drop_duplicates("period")
        values.append({row.period: Decimal(row.raw_value) for row in valid.itertuples()})
    if values[0] != values[1]:
        raise ValueError("Taiwan HPI values differ between the two supplied files.")
    return raw


def read_taiwan_cpi(raw_dir):
    """Derive monthly YoY only when both actual CPI index inputs are present.

    Use exact calendar months, not the twelfth previous nonmissing row: gaps
    must never cause us to compare the wrong years. Nothing is interpolated.
    """
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    levels = read_year_grid(raw_dir / "taiwan" / TAIWAN_FILES["cpi_yoy"], labels, "M")
    source_periods = valid_periods(levels)
    values = levels.loc[levels["valid"]].drop_duplicates("period").set_index("period")["raw_value"]
    records = []
    for month, value in values.items():
        previous = str(pd.Period(month, freq="M") - 12)
        if previous not in values.index:
            continue
        denominator = Decimal(values.loc[previous])
        if denominator <= 0:
            continue
        growth = (Decimal(value) / denominator - 1) * 100
        records.append({"period": month, "raw_value": str(growth)})
    data = pd.DataFrame(records, columns=["period", "raw_value"])
    raw = taiwan_observations(data["period"], data["raw_value"], "M")
    # Retain original-level coverage as notes; the matrix row counts YoY pairs.
    raw["source_valid_months"] = len(source_periods)
    raw["source_first_month"] = min(source_periods) if source_periods else "Unknown"
    raw["source_last_month"] = max(source_periods) if source_periods else "Unknown"
    return raw


def taiwan_credit_quarters(credit, nominal_gdp):
    """Count quarters with the exact raw inputs needed for the agreed ratio.

    Credit is a stock, so use only the balance in March, June, September or
    December. Do not average or add monthly balances. Every denominator must
    have four actual, consecutive, finite nominal GDP quarters in NT$ million.
    A positive denominator is required. Both inputs use the same currency unit.
    """
    if metadata_value(credit, "UNIT_MEASURE") != "NT$ million":
        raise ValueError("Taiwan credit currency/unit differs from the agreed formula.")
    if metadata_value(nominal_gdp, "UNIT_MEASURE") != "NT$ million":
        raise ValueError("Taiwan nominal GDP currency/unit differs from the credit unit.")
    valid_periods(credit)
    valid_periods(nominal_gdp)
    credit_values = credit.loc[credit["valid"]].drop_duplicates("period").set_index("period")["raw_value"]
    gdp_values = nominal_gdp.loc[nominal_gdp["valid"]].drop_duplicates("period").set_index("period")["raw_value"]
    records = []
    for month, value in credit_values.items():
        period = pd.Period(month, freq="M")
        if period.month not in [3, 6, 9, 12]:
            continue
        quarter = period.asfreq("Q")
        required = [str(quarter - offset) for offset in [3, 2, 1, 0]]
        if not all(key in gdp_values.index for key in required):
            continue
        denominator = sum(Decimal(gdp_values.loc[key]) for key in required)
        if denominator <= 0:
            continue
        ratio = Decimal(value) / denominator * 100
        records.append({
            "country_code": "TWN", "FREQ": "Q", "period": str(quarter),
            "raw_value": str(ratio), "valid": ratio.is_finite(),
        })
    # These rows describe derived-ratio availability, not directly reported
    # raw ratio observations. That distinction is retained in the matrix.
    return pd.DataFrame(records, columns=["country_code", "FREQ", "period", "raw_value", "valid"])


def read_taiwan(variable, raw_dir):
    """Read Taiwan's supplied formats and map them to the project feature names."""
    if variable == "hpi":
        return read_taiwan_hpi(raw_dir)
    if variable == "cpi_yoy":
        return read_taiwan_cpi(raw_dir)
    if variable == "gdp_growth":
        return read_taiwan_file(raw_dir, TAIWAN_FILES[variable], "gdp_growth_qoq", "Q")
    if variable == "interest_rate":
        return read_taiwan_file(raw_dir, TAIWAN_FILES[variable], "interest_rate", "M")
    credit = read_taiwan_file(raw_dir, TAIWAN_FILES[variable], "consumer_credit_balance", "M")
    gdp = read_taiwan_file(raw_dir, "taiwan_nominal_gdp_raw.csv", "nominal_gdp", "Q")
    return taiwan_credit_quarters(credit, gdp)


def valid_periods(group):
    """deduplicate periods and flag conflicting values explicitly."""
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
    """count observed periods, internal gaps and complete quarters."""
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
    """separate units, price basis, adjustment and transformation."""
    if code == "TWN":
        return taiwan_metadata(variable, group)
    filename = f"data/raw/four_countries/{variable}_raw.csv"
    if group.empty:
        return {
            "unit": "Unknown", "real_nominal": "Unknown", "seasonal_adj": "Unknown",
            "transformation": "Unknown",
            "variable_definition": (
                "Source file removed by the merged remote update; raw coverage cannot be established."
                if variable == "real_income_growth" else
                "No observations supplied for this country; measurement metadata unavailable."
            ),
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


def taiwan_metadata(variable, group):
    """Use matched official series; leave unsupported classifications unknown."""
    filename = f"data/raw/taiwan/{TAIWAN_FILES[variable]}"
    source = filename
    common = " Common availability uses five countries and measures period coverage, not definition equivalence."
    if variable == "hpi":
        return {
            "unit": "Index points (2016Q1=100)", "real_nominal": "Nominal", "seasonal_adj": "NSA",
            "transformation": "Level",
            "variable_definition": (
                "Sinyi housing price index, Taiwan aggregate (台灣), not a city index or "
                "an average of cities. All valid observations cross-checked against "
                "taiwan_hpi_national_quarterly_raw.csv; the duplicate file is not counted twice. Publisher "
                "identifies a hedonic index of residential transactions, excluding presales. "
                "Nominal: the index is built from transaction prices with no inflation adjustment, and "
                "a central bank study uses a separately deflated 'real Sinyi index'; the pipeline deflates "
                "it with Taiwan CPI. Not seasonally adjusted, unlike OECD AnaHPI; quarter-of-year growth "
                "differences are not significant (F = 1.24, p = 0.3, 2001-2025), so no adjustment is applied. "
                "No added history before 2001Q1 and no filled 2025Q4 value." + common
            ),
            "source": (
                f"{source}; data/raw/taiwan/taiwan_hpi_national_quarterly_raw.csv (cross-check); "
                "Sinyi Realty / NCCU Sinyi Real Estate Research and Development Center; "
                f"publisher methodology and history: {TAIWAN_HPI_SOURCE}; "
                f"matching 2025Q3 Taiwan and city values: {TAIWAN_HPI_EXAMPLE}; "
                "real Sinyi index in CBC Quarterly vol. 37 no. 3: "
                "https://www.cbc.gov.tw/public/Attachment/5111316213471.pdf"
            ),
        }
    if variable == "cpi_yoy":
        return {
            "unit": "% (YoY)", "real_nominal": "Not applicable", "seasonal_adj": "Unknown",
            "transformation": "Year-over-year percentage change (derived from raw index levels)",
            "variable_definition": (
                "Monthly CPI growth = (CPI index this month / CPI index same month one year "
                "earlier - 1) * 100. Coverage counts months with both actual finite inputs "
                "and a positive year-earlier index; no filling, monthly aggregation or scaling. "
                f"Original index coverage: {metadata_value(group, 'source_valid_months')} valid months, "
                f"{metadata_value(group, 'source_first_month')}–{metadata_value(group, 'source_last_month')}. "
                "Annual Index column excluded. The September 2025 and September 2024 "
                "levels (109.82 and 108.46) match DGBAS headline CPI, base 2021=100. "
                "This verifies the series match for that example, not every historical "
                "value or vintage. Seasonal adjustment is not documented in the CSV. "
                "Rates are calculated from supplied rounded levels, not directly reported "
                "YoY figures; later revisions/rebasing may change historical inputs." + common
            ),
            "source": (
                f"{source}; DGBAS headline CPI (2021=100); September 2024/2025 levels "
                f"matched to Price Statistics Monthly: {TAIWAN_CPI_BOOK}; "
                f"dated September 2025 release: {TAIWAN_CPI_RELEASE}"
            ),
        }
    if variable == "household_credit_gdp":
        source += (
            f"; CBC consumer-loan subtotal (447 valid balances matched): {TAIWAN_CREDIT_SOURCE}; "
            f"DGBAS nominal GDP: {TAIWAN_GDP_REVISION}; "
            "data/raw/taiwan/taiwan_nominal_gdp_raw.csv; ratio calculated locally"
        )
        return {
            "unit": "% of GDP", "real_nominal": "Not applicable", "seasonal_adj": "Unknown",
            "transformation": "Ratio to GDP (derived from raw inputs)",
            "variable_definition": (
                "Consumer-credit balance in the quarter's final month / total nominal GDP for "
                "that quarter and the previous three quarters * 100; both inputs NT$ million. "
                "Coverage and complete_quarters count quarters with all required raw inputs, "
                "not directly reported ratios. Consumer credit may differ from BIS total "
                "household/NPISH credit; no monthly filling or averaging." + common
            ),
            "source": source,
        }
    unit = metadata_value(group, "UNIT_MEASURE")
    if variable == "gdp_growth":
        adjustment = metadata_value(group, "ADJUSTMENT")
        seasonal = {"Seasonally adjusted": "SA", "Not seasonally adjusted": "NSA"}.get(adjustment, "Unknown")
        return {
            "unit": "% (QoQ)" if unit == "%" else unit,
            "real_nominal": "Unknown", "seasonal_adj": seasonal,
            "transformation": "Quarter-over-quarter percentage change",
            "variable_definition": (
                "GDP growth; raw feature gdp_growth_qoq, supplied directly as quarterly "
                f"percentage changes; source adjustment: {adjustment}. No conversion from "
                "YoY, annualisation/deannualisation, filling or scaling performed. The "
                "blank 1982Q1 cell is excluded from coverage. Price basis (real/nominal), "
                "annualisation convention, original publisher/table identifier and "
                "release vintage are not documented in the new CSV; equivalence to "
                "the OECD definitions is not fully established." + common
            ),
            "source": f"{source}; upstream-supplied QoQ series; original statistical publisher/table not verified",
        }
    return {
        "unit": unit, "real_nominal": "Unknown", "seasonal_adj": "Unknown",
        "transformation": "Level (monthly weighted-average yield)",
        "variable_definition": (
            "10-year government bond secondary-market yield; monthly weighted average, "
            "remaining maturity close to ten years. All 359 valid raw observations "
            "match CBC open-data table EG43M01, column 十年期政府公債次級市場利率(%). "
            "CBC compiles this series using Taipei Exchange data; not a policy rate." + common
        ),
        "source": f"{source}; CBC / Taipei Exchange; {TAIWAN_RATE_SOURCE}; {TAIWAN_RATE_SCHEDULE}",
    }


def taiwan_publication_metadata(variable, periods, group):
    """Separate first estimates, later revisions, schedules and derived ratios."""
    if variable == "cpi_yoy":
        lag = (
            "DGBAS schedule: 5 working days after month-end, adjusted for Lunar New Year, "
            "long holidays or special circumstances. This is not 5 calendar days."
        )
        urls = [TAIWAN_CPI_SCHEDULE]
        status = "Official schedule estimate" if periods else "Not verified"
        # Match the computed rate to the two verified raw index inputs before
        # attaching the historical example; never change the raw CPI values.
        values = group.loc[group["period"].eq("2025-09") & group["valid"], "raw_value"]
        expected_growth = (Decimal("109.82") / Decimal("108.46") - 1) * 100
        if len(values) and all(Decimal(value) == expected_growth for value in values):
            end = pd.Period("2025-09", freq="M").end_time.normalize()
            days = (pd.Timestamp("2025-10-08") - end).days
            lag += (
                f" Verified example: September 2025 -> 2025-10-08 ({days} calendar days). "
                "The two index inputs match DGBAS; derived YoY rounds to its reported 1.25%."
            )
            urls.extend([TAIWAN_CPI_RELEASE, TAIWAN_CPI_BOOK])
            status = "Verified"
        lag += (
            " Historical first-release dates for every exact input vintage remain unverified; "
            "revised or rebased historical values cannot be assumed available at first release."
        )
        return {"publication_lag": lag, "publication_lag_source": "; ".join(urls),
                "publication_lag_status": status}
    if variable == "hpi":
        examples, urls = [], []
        for period, expected_value, date, url in TAIWAN_HPI_NOTICES:
            values = group.loc[group["period"].eq(period) & group["valid"], "raw_value"]
            if not len(values) or not all(Decimal(value) == Decimal(expected_value) for value in values):
                continue
            end = pd.Period(period, freq="Q").end_time.normalize()
            days = (pd.Timestamp(date) - end).days
            examples.append(f"{period} available by {date} ({days} calendar days after quarter-end)")
            urls.append(url)
        return {
            "publication_lag": (
                "Partially verified: dated NCCU/Sinyi notices establish " + "; ".join(examples) + ". "
                "These are availability-by bounds, not verified earliest Sinyi release dates "
                "or a fixed historical schedule. Later article/update timestamps are not "
                "used as first-release dates."
                if examples else "Not verified: no matching dated HPI release example in the supplied observations."
            ),
            "publication_lag_source": "; ".join(urls),
            "publication_lag_status": "Partially verified" if examples else "Not verified",
        }
    if variable == "interest_rate" and periods:
        return {
            "publication_lag": (
                "About 23 calendar days after month-end: CBC's published monthly bond-rate "
                "schedule (2026 metadata). Saturday releases move to Friday; Sunday/holiday "
                "releases move to the next business day. Historical first-release dates "
                "have not been verified for every observation."
            ),
            "publication_lag_source": TAIWAN_RATE_SCHEDULE,
            "publication_lag_status": "Official schedule estimate",
        }
    if variable == "gdp_growth":
        # The previous evidence matched the removed YoY series. Neither those
        # values nor their release dates establish this QoQ file's vintage.
        return {
            "publication_lag": (
                "Not verified for the supplied seasonally adjusted QoQ series. Original "
                "publisher/table and release-vintage fields are absent. The removed "
                "YoY series' dated release examples are not transferred to this file."
            ),
            "publication_lag_source": "", "publication_lag_status": "Not verified",
        }
    if "2025Q3" not in periods:
        return {"publication_lag": "Not verified", "publication_lag_source": "",
                "publication_lag_status": "Not verified"}
    # This ratio is calculated locally, so it has no separate agency release.
    # A dated credit release alone cannot establish the ratio's availability:
    # all four nominal-GDP inputs must also have been published at that vintage.
    return {
        "publication_lag": (
            "Partially verified: September 2025 consumer credit was released 2025-10-27 "
            "(27 calendar days after quarter-end). Credit/GDP must wait until both the "
            "quarter-end credit balance and all four nominal-GDP quarters are available. "
            "The first availability date of the exact GDP input vintage in this file "
            "is not verified; 27 days must not be used as the ratio's lag. "
            "No constant historical lag established."
        ),
        "publication_lag_source": f"{TAIWAN_CREDIT_RELEASE}; {TAIWAN_GDP_REVISION}",
        "publication_lag_status": "Partially verified",
    }


def publication_metadata(code, variable, frequency, periods, group):
    """attach traceable timing examples, with their precise limitations."""
    if code == "TWN":
        return taiwan_publication_metadata(variable, periods, group)
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


def hpi_release_evidence(path, markets):
    evidence = pd.read_csv(path, dtype=str, keep_default_na=False).map(lambda x: x.strip())
    required = {"series_id", "country", "source_file", "lag_days", "reference_period",
                "release_date", "evidence_url", "status", "notes"}
    if not required <= set(evidence.columns):
        raise ValueError(f"{path} is missing columns {sorted(required - set(evidence.columns))}")
    if evidence["series_id"].duplicated().any() or set(evidence["series_id"]) != set(markets):
        raise ValueError("The release-lag table must contain each evaluation market exactly once.")
    statuses = {"unverified", "release_example", "schedule_estimate", "verified"}
    for row in evidence.to_dict("records"):
        sid, status = row["series_id"], row["status"]
        country = "Taiwan" if sid == "Taiwan_national" else sid
        source = ("data/raw/taiwan/taiwan_hpi_quarterly_raw.csv" if sid == "Taiwan_national" else
                  "data/raw/four_countries/hpi_raw.csv" if sid in ORIGINAL_COUNTRIES.values() else
                  "data/raw/global/anahpi_quarterly_raw.csv")
        if row["country"] != country or row["source_file"] != source:
            raise ValueError(f"{sid}: country/source_file must match the forecasting data source.")
        if status not in statuses:
            raise ValueError(f"{sid}: status must be one of {sorted(statuses)}.")
        if row["evidence_url"] and not row["evidence_url"].startswith(("https://", "http://")):
            raise ValueError(f"{sid}: evidence_url must be an HTTP or HTTPS URL.")
        if status == "unverified":
            if row["lag_days"]:
                raise ValueError(f"{sid}: record unverified lag estimates in notes, not lag_days.")
            continue
        if not row["lag_days"].isdigit() or not row["evidence_url"].startswith(("https://", "http://")):
            raise ValueError(f"{sid}: a non-negative integer lag_days and evidence URL are required.")
        if status == "release_example":
            quarter = row["reference_period"]
            date = row["release_date"]
            if not quarter or not date:
                raise ValueError(f"{sid}: a release example needs reference_period and release_date.")
            end = pd.Period(quarter, freq="Q").end_time.normalize()
            lag = (pd.to_datetime(date, format="%Y-%m-%d") - end).days
            if lag != int(row["lag_days"]):
                raise ValueError(f"{sid}: lag_days does not match the supplied quarter and release date.")
        if status == "verified" and not row["notes"]:
            raise ValueError(f"{sid}: describe the verified source and historical scope in notes.")
    return evidence.set_index("series_id")


def add_national_hpi(matrix, raw_dir, market_list, release_lags):
    listed = pd.read_csv(market_list)
    markets = listed.loc[listed["validation_origins_2000_2009"] > 0, "series_id"]
    if len(markets) != 41 or markets.duplicated().any():
        raise ValueError("Expected 41 unique evaluation markets in the market list.")
    evidence = hpi_release_evidence(release_lags, markets)
    raw = pd.read_csv(raw_dir / "global" / "anahpi_quarterly_raw.csv", dtype=str, keep_default_na=False)
    existing = set(matrix.loc[matrix["variable"].eq("hpi"), "country"])
    extra = []
    all_periods = {}
    hpi_raw = read_raw("hpi", raw_dir)
    for code, country in ORIGINAL_COUNTRIES.items():
        all_periods[country] = valid_periods(hpi_raw.loc[hpi_raw["country_code"].eq(code)])
    all_periods["Taiwan"] = valid_periods(read_taiwan("hpi", raw_dir))
    for sid in markets:
        if sid == "Taiwan_national" or sid in existing:
            continue
        group = raw.loc[raw["Country"].eq(sid) & raw["MEASURE"].eq("RHP") & raw["FREQ"].eq("Q")].copy()
        if group.empty:
            raise ValueError(f"No quarterly real HPI observations found for {sid}.")
        group["country_code"] = group["REF_AREA"]
        group["raw_value"] = group["Value"]
        group["period"] = group["Period"].str.replace("-", "", regex=False)
        numeric = pd.to_numeric(group["raw_value"], errors="coerce")
        group["valid"] = numeric.notna() & ~numeric.isin([float("inf"), -float("inf")])
        periods = valid_periods(group)
        all_periods[sid] = periods
        if metadata_value(group, "UNIT_MEASURE") != "IX":
            raise ValueError(f"Unexpected HPI units for {sid}.")
        extra.append({
            "country": sid, "variable": "hpi", "frequency": "Quarterly",
            **coverage_statistics(periods, "Q"),
            "unit": f"Index points (base period {metadata_value(group, 'BASE_PER')})",
            "real_nominal": "Real",
            "seasonal_adj": ADJUSTMENTS.get(metadata_value(group, "ADJUSTMENT"), "Unknown"),
            "transformation": "Level",
            "variable_definition": "OECD real national house price index, as supplied in the raw AnaHPI export.",
            "source": "data/raw/global/anahpi_quarterly_raw.csv; " + HOUSING_REFERENCE,
            "publication_lag": "Not verified", "publication_lag_source": "",
            "publication_lag_status": "Not verified",
        })
    matrix = pd.concat([matrix, pd.DataFrame(extra, columns=COLUMNS)], ignore_index=True)
    common = set.intersection(*all_periods.values())
    hpi = matrix["variable"].eq("hpi")
    matrix.loc[hpi, "common_availability"] = f"{min(common)}–{max(common)}" if common else ""
    matrix.loc[hpi, "common_valid_observations"] = len(common)
    matrix["common_valid_observations"] = matrix["common_valid_observations"].astype(int)
    for sid, row in evidence.iterrows():
        country = row["country"]
        selected = hpi & matrix["country"].eq(country)
        if row["status"] != "unverified":
            labels = {"release_example": "Release example", "schedule_estimate": "Official schedule estimate",
                      "verified": "Verified"}
            text = f"{row['lag_days']} calendar days after quarter-end ({row['status']})"
            if row["reference_period"]:
                text += f"; reference quarter {row['reference_period']}"
            if row["release_date"]:
                text += f"; release date {row['release_date']}"
            if row["status"] == "release_example":
                text += "; a dated example does not establish a fixed historical release lag"
            matrix.loc[selected, ["publication_lag", "publication_lag_source", "publication_lag_status"]] = [
                text, row["evidence_url"], labels[row["status"]]]
        elif row["evidence_url"]:
            original = matrix.loc[selected, "publication_lag_source"].iloc[0]
            matrix.loc[selected, "publication_lag_source"] = "; ".join(x for x in [original, row["evidence_url"]] if x)
        if row["notes"]:
            matrix.loc[selected, "publication_lag"] += "; " + row["notes"]
    # HPI overlap covers 41 markets; macro overlap retains its original country groups.
    matrix.loc[hpi, "variable_definition"] = matrix.loc[hpi, "variable_definition"].str.replace(
        " Common availability uses five countries and measures period coverage, not definition equivalence.",
        "", regex=False) + " Common HPI coverage is the intersection of all 41 evaluation markets."
    if matrix.loc[hpi, "country"].nunique() != 41 or matrix.duplicated(["country", "variable", "frequency"]).any():
        raise ValueError("Unexpected or duplicate HPI market coverage.")
    return matrix[COLUMNS]


def build_matrix(raw_dir, market_list=ROOT / "docs/market_list.csv", release_lags=ROOT / "docs/hpi_release_lags.csv"):
    """Build one row per country-variable-source-frequency, then intersect sets."""
    records = []
    for variable in VARIABLES:
        raw = read_raw(variable, raw_dir)
        taiwan = read_taiwan(variable, raw_dir) if variable in TAIWAN_VARIABLES else None
        # Frequencies must actually occur in the target countries' raw file.
        # An absent country's row (for example income growth) records zero
        # coverage, keeping all four markets visible for that observed group.
        frequencies = set(raw["FREQ"])
        if taiwan is not None:
            frequencies.update(taiwan["FREQ"])
        if variable == "real_income_growth" and not frequencies:
            # Preserve the original quarterly indicator contract when its
            # entire raw source is missing; these are zero-coverage rows.
            frequencies.add("Q")
        for frequency in sorted(frequencies):
            groups = {
                code: raw.loc[raw["country_code"].eq(code) & raw["FREQ"].eq(frequency)]
                for code in ORIGINAL_COUNTRIES
            }
            # Taiwan CPI stays monthly. Do not invent a quarterly Taiwan row
            # just because other countries also supplied quarterly CPI.
            if taiwan is not None and taiwan["FREQ"].eq(frequency).any():
                groups["TWN"] = taiwan.loc[taiwan["FREQ"].eq(frequency)]
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
                    **publication_metadata(code, variable, frequency, periods, group),
                })
    matrix = pd.DataFrame(records, columns=COLUMNS)
    order = {country: index for index, country in enumerate(COUNTRIES.values())}
    matrix = matrix.sort_values("country", key=lambda column: column.map(order), kind="stable").reset_index(drop=True)
    if set(matrix["country"]) != set(COUNTRIES.values()):
        raise ValueError("Unexpected country coverage.")
    if matrix.duplicated(["country", "variable", "frequency"]).any():
        raise ValueError("Duplicate matrix rows.")
    return add_national_hpi(matrix, raw_dir, market_list, release_lags)


def export_matrix(matrix, output_dir):
    """write the matrix to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "availability_matrix.csv"
    matrix.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


CONFIGS = (ROOT / "configs/experiment.toml", ROOT / "configs/experiment_long.toml")


def lags_above_config(release_lags, configs=CONFIGS):
    """Supported HPI lags that are longer than the lag assumed in an experiment config.

    Only rows with evidence (any status except unverified) are checked. The configs are
    read, never changed; a longer lag means the timing rule needs review before rerunning.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return None
    evidence = pd.read_csv(release_lags, dtype=str, keep_default_na=False)
    evidence = evidence[evidence["status"].str.strip().ne("unverified") & evidence["lag_days"].str.strip().ne("")]
    found = []
    for config in configs:
        with open(config, "rb") as f:
            table = tomllib.load(f)["release_lag_days"]["hpi"]
        for row in evidence.itertuples():
            assumed = int(table.get(row.country, table["default"]))
            if int(row.lag_days) > assumed:
                found.append((Path(config).name, row.series_id, int(row.lag_days), assumed, row.status))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--market-list", type=Path, default=ROOT / "docs/market_list.csv")
    parser.add_argument("--release-lags", type=Path, default=ROOT / "docs/hpi_release_lags.csv")
    args = parser.parse_args()
    matrix = build_matrix(args.raw_dir, args.market_list, args.release_lags)
    csv_path = export_matrix(matrix, args.output_dir)
    print(f"Created {len(matrix)} rows with HPI coverage for 41 markets and macro coverage for five countries.")
    print(csv_path)
    above = lags_above_config(args.release_lags)
    if above is None:
        print("Note: install tomli (Python 3.10) to compare release lags with the configs.")
    elif above:
        print("\nWARNING: supported HPI release lags longer than the configured lag:")
        for config, sid, lag, assumed, status in above:
            print(f"  {config}: {sid} {lag} days ({status}) > {assumed} days assumed")
        print("Review the evidence, then update [release_lag_days] hpi in both configs and rerun "
              "the affected comparisons if the timing rule changes.")
    else:
        print("All supported HPI release lags are within the configured lags.")


if __name__ == "__main__":
    main()
