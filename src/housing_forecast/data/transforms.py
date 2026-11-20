"""Row-wise transforms that only look backwards in time (no fitting, so no leakage).

Anything that must be FITTED (standardization, feature selection, regressions)
lives in backtesting/scaling.py and the models, and is fitted per training window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_forecast.config import hpi_basis
from housing_forecast.quarters import to_period


def monthly_to_quarterly(monthly: pd.DataFrame, value_col: str, how: str = "mean",
                         min_months: int = 3) -> pd.DataFrame:
    """Monthly long table (country, month 'YYYY-MM', value) -> quarterly.

    how="mean": quarterly average, for rates and flows (interest rate, CPI).
    how="last": value of the quarter's last month, for stocks (loan balances).
    A quarter is kept only if it has at least `min_months` months; with
    how="last" the last month itself must be present.
    """
    df = monthly.dropna(subset=[value_col]).copy()
    m = pd.PeriodIndex(pd.to_datetime(df["month"]), freq="M")
    df["quarter"] = m.asfreq("Q").astype(str)
    df["_is_last"] = (m.month % 3 == 0)
    g = df.groupby(["country", "quarter"])
    if how == "mean":
        out = g[value_col].mean().to_frame(value_col)
    elif how == "last":
        last = df[df["_is_last"]].set_index(["country", "quarter"])[value_col]
        out = last.reindex(g.size().index).to_frame(value_col)
    else:
        raise ValueError("how must be 'mean' or 'last'")
    out["n_months"] = g[value_col].count()
    out = out[(out["n_months"] >= min_months) & out[value_col].notna()].drop(columns="n_months")
    return out.reset_index()


def monthly_to_quarterly_mean(monthly: pd.DataFrame, value_col: str, min_months: int = 3) -> pd.DataFrame:
    return monthly_to_quarterly(monthly, value_col, how="mean", min_months=min_months)


def credit_to_gdp(df: pd.DataFrame, credit_col: str, gdp_col: str, out_col: str,
                  key: str = "country") -> pd.DataFrame:
    """credit (end of quarter) / sum of the last 4 quarters of nominal GDP * 100.

    Same definition as BIS "credit, % of GDP". The 4-quarter sum is taken on a
    calendar-complete index, so a missing quarter gives NaN instead of summing
    the wrong quarters.
    """
    pieces = []
    for k, g in df.groupby(key):
        g = g.assign(_q=to_period(g["quarter"])).set_index("_q").sort_index()
        full = pd.period_range(g.index.min(), g.index.max(), freq="Q")
        gdp4 = g[gdp_col].reindex(full).rolling(4, min_periods=4).sum()
        g[out_col] = g[credit_col] / gdp4.reindex(g.index) * 100.0
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def change_over(panel: pd.DataFrame, col: str, lag: int, out_col: str) -> pd.DataFrame:
    """x_q - x_{q-lag}, matched by CALENDAR quarter within each series."""
    df = panel.copy()
    df["_q"] = to_period(df["quarter"])
    past = df[["series_id", "_q", col]].rename(columns={col: "_past"})
    past["_q"] = past["_q"] + lag
    df = df.merge(past, on=["series_id", "_q"], how="left")
    df[out_col] = df[col] - df["_past"]
    return df.drop(columns=["_q", "_past"])


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


def prepare_panel(panel: pd.DataFrame, cpi_monthly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Deflate nominal HPI and derive growth and macro features.
    Keep rate variables in their stated units. Derive Taiwan inflation from CPI
    levels and quarterly credit changes from the credit/GDP ratio."""
    cpi_q = monthly_to_quarterly_mean(cpi_monthly, "cpi")
    df = deflate_to_real(panel, cpi_q, cfg)
    df = log_growth(df, "hpi_real", 1, "hpi_qoq")

    if "cpi_yoy" in df.columns:
        idx = cpi_q.rename(columns={"cpi": "_cpi"})
        idx["_q"] = to_period(idx["quarter"])
        past = idx[["country", "_q", "_cpi"]].rename(columns={"_cpi": "_cpi_past"})
        past["_q"] = past["_q"] + 4
        idx = idx.merge(past, on=["country", "_q"], how="left")
        idx["_yoy"] = (idx["_cpi"] / idx["_cpi_past"] - 1.0) * 100.0
        df = df.merge(idx[["country", "quarter", "_yoy"]], on=["country", "quarter"], how="left")
        df["cpi_yoy"] = df["cpi_yoy"].fillna(df["_yoy"])
        df = df.drop(columns="_yoy")

    if "household_credit_gdp" in df.columns:
        df = change_over(df, "household_credit_gdp", 4, "household_credit_gdp_d4")
    return df.sort_values(["series_id", "quarter"]).reset_index(drop=True)


prepare_hpi_panel = prepare_panel   # old name, kept for existing callers
