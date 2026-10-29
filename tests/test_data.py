import numpy as np
import pandas as pd

from housing_forecast.data.b_adapter import _to_quarter
from housing_forecast.data.transforms import deflate_to_real, log_growth, monthly_to_quarterly_mean


def test_to_quarter_formats():
    s = pd.Series(["1970-Q1", "1977-10-01", "1956-02"])
    assert _to_quarter(s).tolist() == ["1970Q1", "1977Q4", "1956Q1"]


def test_monthly_to_quarterly_needs_full_quarter():
    m = pd.DataFrame({"country": "Taiwan", "month": ["2025-01", "2025-02", "2025-03", "2025-04"], "cpi": [1, 2, 3, 4.0]})
    q = monthly_to_quarterly_mean(m, "cpi")
    assert q["quarter"].tolist() == ["2025Q1"] and q["cpi"].item() == 2.0     # 2025Q2 incomplete -> dropped


def test_deflate_only_nominal(cfg):
    panel = pd.DataFrame({"series_id": ["Taiwan_national", "Japan"], "country": ["Taiwan", "Japan"],
                          "geography_level": "national", "quarter": "2020Q1", "hpi": [110.0, 110.0]})
    cpi = pd.DataFrame({"country": ["Taiwan"], "quarter": ["2020Q1"], "cpi": [110.0]})
    out = deflate_to_real(panel, cpi, cfg).set_index("series_id")
    assert out.loc["Taiwan_national", "hpi_real"] == 100.0       # nominal -> deflated
    assert out.loc["Japan", "hpi_real"] == 110.0                 # AnaHPI already real


def test_log_growth_uses_calendar_not_rows():
    panel = pd.DataFrame({"series_id": "A", "quarter": ["2020Q3", "2020Q1", "2020Q2"], "x": [4.0, 1.0, 2.0]})
    out = log_growth(panel, "x", 1, "g").set_index("quarter")
    assert np.isnan(out.loc["2020Q1", "g"])
    assert np.isclose(out.loc["2020Q3", "g"], np.log(2))


# ---- Taiwan raw files and derived features ---------------------------------------------

def test_parse_period_quarter_and_month():
    from housing_forecast.data.taiwan_raw import parse_period
    out = parse_period(pd.Series(["1982Q2", "Jul88"]))
    assert str(out[0]) == "1982Q2" and str(out[1]) == "1988-07"


def test_monthly_to_quarterly_last_needs_quarter_end():
    from housing_forecast.data.transforms import monthly_to_quarterly
    m = pd.DataFrame({"country": "Taiwan",
                      "month": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
                      "x": [1, 2, 3, 4, 5, np.nan]})
    q = monthly_to_quarterly(m, "x", how="last")
    assert q["quarter"].tolist() == ["2025Q1"] and q["x"].item() == 3     # Q2 has no June value


def test_credit_to_gdp_uses_four_calendar_quarters():
    from housing_forecast.data.transforms import credit_to_gdp
    df = pd.DataFrame({"country": "Taiwan",
                       "quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q2"],
                       "credit": [50.0] * 5, "gdp": [25.0] * 5})
    out = credit_to_gdp(df, "credit", "gdp", "ratio").set_index("quarter")["ratio"]
    assert out["2020Q4"] == 50.0                        # 50 / (4 * 25) * 100
    assert np.isnan(out["2020Q3"]) and np.isnan(out["2021Q2"])   # 2021Q1 missing -> no sum


def test_change_over_is_calendar_based():
    from housing_forecast.data.transforms import change_over
    df = pd.DataFrame({"series_id": "A", "quarter": ["2021Q1", "2020Q1", "2020Q2"], "x": [5.0, 2.0, 9.0]})
    out = change_over(df, "x", 4, "d4").set_index("quarter")["d4"]
    assert out["2021Q1"] == 3.0 and np.isnan(out["2020Q2"])


def test_derived_feature_takes_source_lag(cfg):
    from housing_forecast.backtesting.infoset import features_from_config
    cfg["features"]["use"] = ["hpi_qoq", "household_credit_gdp_d4", "interest_rate"]
    src = {f.name: f.source for f in features_from_config(cfg)}
    assert src == {"hpi_qoq": "hpi", "household_credit_gdp_d4": "household_credit_gdp",
                   "interest_rate": "interest_rate"}


def test_macro_history_before_hpi_is_kept_for_4q_change(cfg):
    """Taiwan HPI starts 2001Q1; 2001's credit change needs 2000's ratio."""
    from housing_forecast.data.b_adapter import attach_national_macro
    from housing_forecast.data.transforms import change_over
    hpi = pd.DataFrame({"series_id": "Taiwan_national", "country": "Taiwan", "geography_level": "national",
                        "quarter": ["2001Q1", "2001Q2"], "hpi": [100.0, 101.0]})
    macro = pd.DataFrame({"country": "Taiwan",
                          "quarter": ["2000Q1", "2000Q2", "2001Q1", "2001Q2"],
                          "household_credit_gdp": [40.0, 41.0, 43.0, 45.0]})
    panel = attach_national_macro(hpi, macro)
    out = change_over(panel, "household_credit_gdp", 4, "d4").set_index("quarter")
    assert out.loc["2001Q1", "d4"] == 3.0 and out.loc["2001Q2", "d4"] == 4.0
    assert np.isnan(out.loc["2000Q1", "hpi"])          # earlier rows carry no HPI, so no labels


def test_cities_share_national_macro():
    from housing_forecast.data.b_adapter import attach_national_macro
    hpi = pd.DataFrame({"series_id": ["Taiwan_national", "Taiwan_Taipei"], "country": "Taiwan",
                        "geography_level": ["national", "city"], "quarter": "2001Q1", "hpi": [1.0, 2.0]})
    macro = pd.DataFrame({"country": "Taiwan", "quarter": ["2001Q1"], "interest_rate": [3.0]})
    out = attach_national_macro(hpi, macro).set_index("series_id")
    assert out.loc["Taiwan_Taipei", "interest_rate"] == 3.0
    assert out.loc["Taiwan_Taipei", "geography_level"] == "city"
