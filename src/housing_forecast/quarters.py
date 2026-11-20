"""Quarter-string conversion and calendar arithmetic."""

from __future__ import annotations

import pandas as pd


def to_period(quarters) -> pd.PeriodIndex:
    """'2010Q1' strings (or a Series of them) -> quarterly PeriodIndex."""
    return pd.PeriodIndex(quarters, freq="Q")


def quarter_end(periods) -> pd.DatetimeIndex:
    """Last calendar day of each quarter (e.g. 2010Q1 -> 2010-03-31)."""
    idx = pd.PeriodIndex(periods, freq="Q") if not isinstance(periods, pd.PeriodIndex) else periods
    return idx.end_time.normalize()
