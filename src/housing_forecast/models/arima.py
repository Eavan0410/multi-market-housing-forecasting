"""One ARIMA per market (traditional "one task - one model" baseline). TEMPLATE for B.

What is already done here (A):
  - rebuilding each market's quarterly growth history from the backtest windows,
  - extending it with the quarters that are known at each forecast date,
  - turning a path of quarterly growth forecasts into cumulative growth for h = 1..20,
  - fallbacks and a record of every market that could not be fitted.

What B fills in (search for "TODO(B)"):
  - fit_one():        choose the ARIMA order and fit it on ONE market's training history,
  - forecast_path():  forecast the next H quarterly growth rates from a fitted model.

Run it like any other model:
    python scripts/run_backtest.py --config configs/experiment_long.toml \
        --models arima --pool national --experiment long_hpi

Rules (from the project plan):
  - choose the order with training data only (e.g. AIC on the training history); never
    look at validation or test errors when choosing it;
  - the series modelled is g_t = log(HPI_real[t] / HPI_real[t-1]) (feature `hpi_qoq`),
    and the target is the cumulative growth  y(t, h) = g_{t+1} + ... + g_{t+h};
  - every market that cannot be fitted or forecast falls back to its own average growth
    and is recorded; run_backtest.py saves the record per fold as diag_arima__<pool>.csv
    (split_id, series_id, n_train_obs, fit_status, reason, n_forecasts, n_fallback_forecasts),
    so fallback forecasts are never reported as pure ARIMA results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_forecast.backtesting.infoset import SampleSet
from housing_forecast.models.base import Forecaster


# ---------------------------------------------------------------------------
# Helpers (done)
# ---------------------------------------------------------------------------

def _quarter_values(s: SampleSet, rows: np.ndarray, j: int) -> dict[int, float]:
    """Map quarter ordinal -> growth value, read from the input windows of `rows`.

    The window of a sample with origin t holds g_{t-L+1}, ..., g_t (L = lookback), so
    overlapping windows give back the whole history, including the first L-1 quarters.
    """
    L = s.X.shape[1]
    origins = pd.PeriodIndex(s.meta["origin"].iloc[rows], freq="Q").asi8
    out: dict[int, float] = {}
    for r, o in zip(rows, origins):
        for i in range(L):
            out[int(o - L + 1 + i)] = float(s.X[r, i, j])
    return out


def _last_contiguous(values: dict[int, float]) -> pd.Series:
    """Values as a series indexed by quarter ordinal, keeping only the last unbroken stretch."""
    if not values:
        return pd.Series(dtype=float)
    q = np.array(sorted(values))
    breaks = np.flatnonzero(np.diff(q) != 1)
    start = q[breaks[-1] + 1] if len(breaks) else q[0]
    q = q[q >= start]
    return pd.Series([values[k] for k in q], index=q, dtype=float)


def series_histories(s: SampleSet, feature: str = "hpi_qoq") -> dict[str, pd.Series]:
    """Each market's growth history available in a training set."""
    j = s.feature_names.index(feature)
    sids = s.meta["series_id"].to_numpy()
    return {sid: _last_contiguous(_quarter_values(s, np.flatnonzero(sids == sid), j))
            for sid in np.unique(sids)}


def extend_history(train_hist: pd.Series | None, s: SampleSet, row: int, feature: str = "hpi_qoq") -> pd.Series:
    """Training history plus the quarters in this forecast's input window (known at the forecast date)."""
    j = s.feature_names.index(feature)
    values = {} if train_hist is None else dict(zip(train_hist.index.astype(int), train_hist.to_numpy()))
    values.update(_quarter_values(s, np.array([row]), j))
    return _last_contiguous(values)


# ---------------------------------------------------------------------------
# B fills in these two functions
# ---------------------------------------------------------------------------

def fit_one(history: np.ndarray, max_p: int = 4, max_q: int = 2):
    """Choose an ARIMA(p, 0, q) order and fit it on one market's training growth history.

    TODO(B):
      - loop over p in 0..max_p and q in 0..max_q, fit each candidate, keep the lowest AIC;
      - statsmodels example:
            from statsmodels.tsa.arima.model import ARIMA
            res = ARIMA(history, order=(p, 0, q), trend="c").fit()
            res.aic
      - return the chosen fitted result (anything forecast_path() can use);
      - raise an exception if no candidate can be fitted (the caller records the fallback).
    """
    import warnings

    from statsmodels.tsa.arima.model import ARIMA   # ImportError here must stop the run, not fall back
    best_model = None
    best_aic = float("inf")

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")      # convergence warnings are checked below
                    model = ARIMA(history, order=(p, 0, q), trend="c").fit()
            except Exception:
                continue
            converged = getattr(model, "mle_retvals", None) or {}
            if not converged.get("converged", True):
                continue                                 # skip candidates whose optimizer did not converge
            if np.isfinite(model.aic) and model.aic < best_aic:
                best_aic = model.aic
                best_model = model

    if best_model is None:
        raise RuntimeError("No ARIMA model could be fitted")

    return best_model



def forecast_path(fitted, history: np.ndarray, steps: int) -> np.ndarray:
    """Forecast the next `steps` quarterly growth rates after the END of `history`.

    `history` can be longer than the training history (it includes quarters known at the
    forecast date). Keep the fitted parameters and only re-run the filter on it.

    TODO(B):
      - statsmodels example:
            res_new = fitted.apply(history)      # same parameters, longer history
            return np.asarray(res_new.forecast(steps))
      - return an array of length `steps`.
    """
    updated = fitted.apply(history, refit=False)
    return np.asarray(updated.forecast(steps))


# ---------------------------------------------------------------------------
# Forecaster (done)
# ---------------------------------------------------------------------------

class ARIMAPerMarket(Forecaster):
    """One ARIMA per market, fitted on that market's own training history only."""

    name = "arima"
    stochastic = False
    needs_scaling = False       # ARIMA works on the raw growth rates

    def __init__(self, feature: str = "hpi_qoq", min_obs: int = 24, max_p: int = 4, max_q: int = 2):
        self.feature, self.min_obs, self.max_p, self.max_q = feature, min_obs, max_p, max_q
        self.models_: dict = {}
        self.hist_: dict[str, pd.Series] = {}
        self.fallbacks_: dict[str, str] = {}
        self.diag_: dict[str, dict] = {}

    def fit(self, train: SampleSet, seed: int) -> None:
        self.hist_ = series_histories(train, self.feature)
        self.models_, self.fallbacks_, self.diag_ = {}, {}, {}
        for sid, h in self.hist_.items():
            self.diag_[sid] = {"n_train_obs": len(h), "fit_status": "fitted", "reason": "",
                               "n_forecasts": 0, "n_fallback_forecasts": 0}
            if len(h) < self.min_obs:
                self._fallback(sid, "too_short", f"{len(h)} < {self.min_obs} quarters")
                continue
            try:
                self.models_[sid] = fit_one(h.to_numpy(), self.max_p, self.max_q)
            except (NotImplementedError, ImportError):
                raise                                    # a missing package is not a market-level failure
            except Exception as e:  # noqa: BLE001  record and fall back, never hide
                self._fallback(sid, "fit_failed", f"{type(e).__name__}: {e}")

    def _fallback(self, sid: str, status: str, reason: str) -> None:
        d = self.diag_.setdefault(sid, {"n_train_obs": 0, "fit_status": status, "reason": "",
                                        "n_forecasts": 0, "n_fallback_forecasts": 0})
        if d["fit_status"] == "fitted" and status.startswith("forecast"):
            d["fit_status"] = "fitted, some forecasts failed"
        elif not status.startswith("forecast"):
            d["fit_status"] = status
        if reason[:200] not in d["reason"]:
            d["reason"] = (d["reason"] + "; " if d["reason"] else "") + reason[:200]
        self.fallbacks_[sid] = d["reason"]

    def diagnostics(self) -> pd.DataFrame:
        """One row per market seen in this fold (training or test); collected by run_backtest."""
        return pd.DataFrame([{"series_id": sid, **d} for sid, d in self.diag_.items()],
                            columns=["series_id", "n_train_obs", "fit_status", "reason",
                                     "n_forecasts", "n_fallback_forecasts"])

    def predict(self, test: SampleSet) -> np.ndarray:
        steps = max(test.horizons)
        idx = np.asarray(test.horizons) - 1
        out = np.empty((len(test), len(test.horizons)))
        for r, sid in enumerate(test.meta["series_id"]):
            hist = extend_history(self.hist_.get(sid), test, r, self.feature)
            if sid not in self.diag_:
                self._fallback(sid, "no_training_history", "market has no training rows in this fold")
            g = None
            if sid in self.models_:
                try:
                    g = np.asarray(forecast_path(self.models_[sid], hist.to_numpy(), steps), dtype=float)
                    if g.shape != (steps,) or not np.isfinite(g).all():
                        raise ValueError(f"forecast_path returned shape {g.shape} or non-finite values")
                except NotImplementedError:
                    raise
                except Exception as e:  # noqa: BLE001
                    self._fallback(sid, "forecast_failed", f"{type(e).__name__}: {e}")
                    g = None
            d = self.diag_[sid]
            d["n_forecasts"] += 1
            if g is None:
                # Fallback: the market's own average growth (drift); no history at all -> no change.
                d["n_fallback_forecasts"] += 1
                g = np.full(steps, hist.mean() if len(hist) else 0.0)
            out[r] = np.cumsum(g)[idx]
        return out
