"""Quarter helpers (季度工具).

All tables in this project write quarters as strings like "2010Q1".
Inside the code we turn them into pandas Period objects, because a Period
knows calendar arithmetic:

    pd.Period("2010Q4", freq="Q") + 1   ->  Period("2011Q1")

So "t + h quarters" is just `period + h`, and we never count rows by hand.
"""

from __future__ import annotations

import pandas as pd


def to_period(quarters) -> pd.PeriodIndex:
    """'2010Q1' strings (or a Series of them) -> quarterly PeriodIndex."""
    return pd.PeriodIndex(quarters, freq="Q")


def quarter_end(periods) -> pd.DatetimeIndex:
    """Last calendar day of each quarter (e.g. 2010Q1 -> 2010-03-31)."""
    idx = pd.PeriodIndex(periods, freq="Q") if not isinstance(periods, pd.PeriodIndex) else periods
    return idx.end_time.normalize()
