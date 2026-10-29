"""Row-wise transforms that only look backwards in time (no fitting, so no leakage).

Anything that must be FITTED (standardization, feature selection, regressions)
lives in backtesting/scaling.py and the models, and is fitted per training window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_forecast.config import hpi_basis
from housing_forecast.quarters import to_period


def monthly_to_quarterly_mean(monthly: pd.DataFrame, value_col: str, min_months: int = 3) -> pd.DataFrame:
    """Monthly long table (country, month 'YYYY-MM', value) -> quarterly mean.

    A quarter is kept only if it has at least `min_months` months, so a partly
    published quarter is not treated as complete.
    """
    df = monthly.copy()
    df["quarter"] = pd.PeriodIndex(pd.to_datetime(df["month"]), freq="Q").astype(str)
    g = df.groupby(["country", "quarter"])[value_col]
    out = g.mean().to_frame(value_col)
    out["n_months"] = g.count()
    out = out[out["n_months"] >= min_months].drop(columns="n_months")
    return out.reset_index()


def deflate_to_real(panel: pd.DataFrame, cpi_quarterly: pd.DataFrame, cfg: dict,
                    hpi_col: str = "hpi", out_col: str = "hpi_real") -> pd.DataFrame:
    """Add `hpi_real`. Nominal series: hpi / CPI * 100; real series: copied as is.

    The CPI base year does not matter: labels are log ratios of hpi_real, so a
    constant rescaling cancels out.
    """
    df = panel.copy()
    basis = df["country"].map(lambda c: hpi_basis(cfg, c))
    nominal = basis.eq("nominal")
    df = df.merge(cpi_quarterly.rename(columns={"cpi": "_cpi"}), on=["country", "quarter"], how="left")
    df[out_col] = np.where(nominal.to_numpy(), df[hpi_col] / df["_cpi"] * 100.0, df[hpi_col])
    missing_cpi = nominal.to_numpy() & df[hpi_col].notna().to_numpy() & df["_cpi"].isna().to_numpy()
    if missing_cpi.any():
        # Leave empty rather than guess: a nominal value without CPI cannot be made real.
        df.loc[missing_cpi, out_col] = np.nan
    return df.drop(columns="_cpi")


def log_growth(panel: pd.DataFrame, col: str, lag: int, out_col: str) -> pd.DataFrame:
    """log(x_q / x_{q-lag}) matched by CALENDAR quarter within each series (never by row)."""
    df = panel.copy()
    df["_q"] = to_period(df["quarter"])
    past = df[["series_id", "_q", col]].rename(columns={col: "_past"})
    past["_q"] = past["_q"] + lag
    df = df.merge(past, on=["series_id", "_q"], how="left")
    df[out_col] = np.log(df[col] / df["_past"])
    return df.drop(columns=["_q", "_past"])


def prepare_hpi_panel(panel: pd.DataFrame, cpi_monthly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Raw minimal panel + raw monthly CPI -> panel with hpi_real and hpi_qoq."""
    cpi_q = monthly_to_quarterly_mean(cpi_monthly, "cpi")
    df = deflate_to_real(panel, cpi_q, cfg)
    df = log_growth(df, "hpi_real", 1, "hpi_qoq")
    return df.sort_values(["series_id", "quarter"]).reset_index(drop=True)
