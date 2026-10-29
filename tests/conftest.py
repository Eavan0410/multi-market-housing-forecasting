import copy

import numpy as np
import pandas as pd
import pytest

from housing_forecast.config import load_config


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


def synthetic_panel(seed: int = 0) -> pd.DataFrame:
    """Taiwan (nominal, + 1 city) and two overseas countries (real), 1995Q1–2025Q3."""
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("1995Q1", "2025Q3", freq="Q").astype(str)
    rows = []
    for sid, country, level, start in [
        ("Taiwan_national", "Taiwan", "national", "2001Q1"),
        ("Taiwan_Taipei", "Taiwan", "city", "2001Q1"),
        ("Japan", "Japan", "national", "1995Q1"),
        ("United States", "United States", "national", "1995Q1"),
    ]:
        g = np.cumsum(rng.normal(0.01, 0.02, len(quarters)))
        hpi = 100 * np.exp(g)
        hpi[quarters < start] = np.nan
        rows.append(pd.DataFrame({"series_id": sid, "country": country, "geography_level": level,
                                  "quarter": quarters, "hpi": hpi}))
    return pd.concat(rows, ignore_index=True)


def synthetic_cpi() -> pd.DataFrame:
    months = pd.period_range("1995-01", "2025-09", freq="M")
    return pd.DataFrame({"country": "Taiwan", "month": months.astype(str),
                         "cpi": np.linspace(70, 110, len(months))})
