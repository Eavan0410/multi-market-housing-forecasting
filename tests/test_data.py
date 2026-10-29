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
