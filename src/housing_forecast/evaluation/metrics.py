"""Point metrics: RMSE, MAE, direction accuracy (方向准确率)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_forecast.data.contract import check_alignment

def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of forecasts with the right sign of growth. Rows with y_true == 0 are skipped.
    A prediction of exactly 0 ("no change") counts as wrong: it calls no direction."""
    m = y_true != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.sign(y_pred[m]) == np.sign(y_true[m])))


def score(df: pd.DataFrame) -> pd.Series:
    e = df["y_pred"].to_numpy() - df["y_true"].to_numpy()
    return pd.Series({
        "n": len(df),
        "rmse": rmse(e),
        "mae": mae(e),
        "dir_acc": direction_accuracy(df["y_true"].to_numpy(), df["y_pred"].to_numpy()),
    })


def metrics_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Per model x horizon: metrics per seed, then mean and std across seeds."""
    check_alignment(preds)
    recs = []
    for (m, h, s), g in preds.groupby(["model", "horizon", "seed"]):
        recs.append({"model": m, "horizon": h, "seed": s, **score(g).to_dict()})
    per_seed = pd.DataFrame(recs)
    agg = per_seed.groupby(["model", "horizon"]).agg(
        n=("n", "first"), n_seeds=("seed", "nunique"),
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae=("mae", "mean"), mae_sd=("mae", "std"),
        dir_acc=("dir_acc", "mean"),
    )
    return agg.reset_index()
