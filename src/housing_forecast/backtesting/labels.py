"""Cumulative real HPI log growth: y(t, h) = log(HPI[t+h] / HPI[t]).

Join by calendar quarter so missing observations cannot shift the targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_forecast.quarters import to_period

HORIZONS = (1, 2, 3, 4)
LABEL_COLUMNS = ["series_id", "origin_quarter", "target_quarter", "horizon", "y_true"]


def make_labels(panel: pd.DataFrame, value_col: str = "hpi_real", horizons=HORIZONS) -> pd.DataFrame:
    base = panel[["series_id", "quarter", value_col]].dropna(subset=[value_col]).copy()
    base["q"] = to_period(base["quarter"])
    base = base.rename(columns={value_col: "hpi_t"})

    pieces = []
    for h in horizons:
        # A row of `future` describes quarter t+h; move its key back by h so it
        # joins the origin t it belongs to.
        future = base[["series_id", "q", "hpi_t"]].rename(columns={"hpi_t": "hpi_future"})
        future["q"] = future["q"] - h

        merged = base.merge(future, on=["series_id", "q"], how="inner")
        merged["y_true"] = np.log(merged["hpi_future"] / merged["hpi_t"])
        merged["target_quarter"] = (merged["q"] + h).astype(str)
        merged["origin_quarter"] = merged["q"].astype(str)
        merged["horizon"] = h
        pieces.append(merged[LABEL_COLUMNS])

    labels = pd.concat(pieces, ignore_index=True)
    return labels.sort_values(["series_id", "origin_quarter", "horizon"]).reset_index(drop=True)
