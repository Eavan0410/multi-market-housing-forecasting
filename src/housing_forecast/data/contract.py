"""Shared CSV contracts (统一 CSV 接口).

Two tables cross team boundaries:

1. Panel (B -> A): raw observations, long format, no scaling / filling / interpolation.
       series_id, country, geography_level, quarter, <var1>, <var2>, ...
   quarter is "2010Q1"; geography_level is "national" or "city";
   missing values stay empty.

2. Predictions (every model -> evaluator):
       model, series_id, origin_quarter, as_of_date, target_quarter, horizon,
       y_true, y_pred, target_type, seed, split_id
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PANEL_KEYS = ["series_id", "country", "geography_level", "quarter"]
GEO_LEVELS = {"national", "city"}

PRED_COLUMNS = [
    "model", "series_id", "origin_quarter", "as_of_date", "target_quarter",
    "horizon", "y_true", "y_pred", "target_type", "seed", "split_id",
]
PRED_KEY = ["model", "series_id", "origin_quarter", "horizon", "seed", "split_id"]


class ContractError(ValueError):
    pass


def _fail(msg: str) -> None:
    raise ContractError(msg)


def validate_panel(df: pd.DataFrame, value_cols: list[str] | None = None) -> None:
    missing = [c for c in PANEL_KEYS if c not in df.columns]
    if missing:
        _fail(f"panel is missing columns {missing}")
    bad_geo = set(df["geography_level"].unique()) - GEO_LEVELS
    if bad_geo:
        _fail(f"unknown geography_level values {bad_geo}")
    try:
        pd.PeriodIndex(df["quarter"], freq="Q")
    except Exception as e:  # noqa: BLE001
        _fail(f"quarter must look like '2010Q1': {e}")
    dup = df.duplicated(["series_id", "quarter"])
    if dup.any():
        _fail(f"duplicate (series_id, quarter) rows, e.g. {df.loc[dup, ['series_id', 'quarter']].head(3).to_dict('records')}")
    if df.groupby("series_id")["country"].nunique().gt(1).any():
        _fail("a series_id maps to more than one country")
    for c in value_cols or []:
        if c not in df.columns:
            _fail(f"panel is missing value column {c}")


def validate_predictions(df: pd.DataFrame) -> None:
    missing = [c for c in PRED_COLUMNS if c not in df.columns]
    if missing:
        _fail(f"predictions are missing columns {missing}")
    if df[PRED_COLUMNS].isna().any().any():
        _fail("prediction fields must not be empty")
    if df[["model", "series_id", "origin_quarter", "horizon", "seed"]].duplicated().any():
        _fail("duplicate prediction keys (model, series_id, origin_quarter, horizon, seed)")
    if not df["horizon"].between(1, 4).all():
        _fail("horizon must be in 1..4")
    origin = pd.PeriodIndex(df["origin_quarter"], freq="Q")
    target = pd.PeriodIndex(df["target_quarter"], freq="Q")
    if not ((target - origin).map(lambda x: x.n) == df["horizon"].to_numpy()).all():
        _fail("target_quarter must equal origin_quarter + horizon")
    as_of = pd.to_datetime(df["as_of_date"])
    if (as_of <= origin.end_time.normalize()).any():
        _fail("as_of_date must be after the end of the origin quarter")
    if not np.isfinite(df[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        _fail("y_true / y_pred must be finite")


def check_alignment(preds: pd.DataFrame, expected_seeds: dict | None = None) -> None:
    """Require complete per-seed coverage and identical forecast definitions.

    expected_seeds maps each model to its required seeds; deterministic models
    may have one seed while stochastic models have the full configured list.
    """
    validate_predictions(preds)
    if preds.empty:
        _fail("predictions must not be empty")
    if expected_seeds is not None and set(expected_seeds) != set(preds["model"]):
        _fail("models do not match expected_seeds")
    keys = ["series_id", "origin_quarter", "horizon"]
    fields = ["target_quarter", "as_of_date", "target_type", "split_id"]
    ref = None
    ref_name = None
    for model, model_rows in preds.groupby("model"):
        if expected_seeds is not None:
            if set(model_rows["seed"]) != set(expected_seeds[model]):
                _fail(f"model {model}: seeds do not match expected seeds {expected_seeds[model]}")
        for seed, g in model_rows.groupby("seed"):
            cur = g.set_index(keys).sort_index()
            name = f"{model}/seed={seed}"
            if ref is None:
                ref, ref_name = cur, name
                continue
            if not cur.index.equals(ref.index):
                _fail(f"{name} is not aligned with {ref_name}: missing or extra forecast rows")
            for field in fields:
                if not cur[field].equals(ref[field]):
                    _fail(f"{name}: {field} differs from {ref_name}")
            if not np.allclose(cur["y_true"], ref["y_true"], rtol=1e-10, atol=1e-12):
                _fail(f"{name}: y_true differs from {ref_name}")
