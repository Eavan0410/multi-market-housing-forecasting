"""No-change and autoregressive baselines.

B adds historical mean / Ridge by following the same Forecaster interface.
"""

from __future__ import annotations

import numpy as np

from housing_forecast.backtesting.infoset import SampleSet
from housing_forecast.models.base import Forecaster


class NoChange(Forecaster):
    """Keep HPI at its current level: zero log growth at every horizon."""

    name = "no_change"

    def fit(self, train: SampleSet, seed: int) -> None:
        pass

    def predict(self, test: SampleSet) -> np.ndarray:
        return np.zeros((len(test), len(test.horizons)))


class ARDirect(Forecaster):
    """Direct AR(p): for each horizon h, OLS of y(t,h) on [1, g_t, g_{t-1}, ..., g_{t-p+1}],
    where g is the series' own quarterly real HPI log growth (feature `hpi_qoq`).

    per_series=True fits one model per series using only that series' training
    rows (the Taiwan-only AR). Where a series has too few rows for a horizon, that
    horizon falls back to the model pooled over all training rows; the other
    horizons keep the series' own fit.
    """

    name = "ar"

    def __init__(self, p: int = 4, feature: str = "hpi_qoq", per_series: bool = True, min_rows: int = 12):
        self.p, self.feature, self.per_series, self.min_rows = p, feature, per_series, min_rows
        self.coefs_: dict = {}

    def _design(self, s: SampleSet) -> np.ndarray:
        j = s.feature_names.index(self.feature)
        lags = s.X[:, -self.p:, j][:, ::-1]                    # newest first: g_t, g_{t-1}, ...
        return np.column_stack([np.ones(len(s)), lags])

    def _ols(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """Coefficients per horizon; a column stays NaN when that horizon has too few rows."""
        B = np.full((Z.shape[1], y.shape[1]), np.nan)
        for k in range(y.shape[1]):
            ok = ~np.isnan(y[:, k])
            if ok.sum() >= max(self.min_rows, Z.shape[1] + 1):
                B[:, k] = np.linalg.lstsq(Z[ok], y[ok, k], rcond=None)[0]
        return None if np.isnan(B).all() else B

    def fit(self, train: SampleSet, seed: int) -> None:
        if self.p > train.X.shape[1]:
            raise ValueError("AR order p is larger than the lookback window")
        Z = self._design(train)
        self.coefs_ = {"__pooled__": self._ols(Z, train.y)}
        if self.per_series:
            sids = train.meta["series_id"].to_numpy()
            for sid in np.unique(sids):
                m = sids == sid
                B = self._ols(Z[m], train.y[m])
                if B is not None:
                    self.coefs_[sid] = B

    def predict(self, test: SampleSet) -> np.ndarray:
        Z = self._design(test)
        pooled = self.coefs_["__pooled__"]
        n_h = len(test.horizons)
        pooled = np.full((Z.shape[1], n_h), np.nan) if pooled is None else pooled
        out = np.empty((len(test), n_h))
        for i, sid in enumerate(test.meta["series_id"]):
            own = self.coefs_.get(sid) if self.per_series else None
            B = pooled if own is None else np.where(np.isnan(own), pooled, own)   # per-horizon fallback
            if np.isnan(B).any():
                bad = [h for h, col in zip(test.horizons, np.isnan(B).any(axis=0)) if col]
                raise RuntimeError(f"AR could not be fitted for horizons {bad}: not enough training rows")
            out[i] = Z[i] @ B
        return out


class ARPooled(ARDirect):
    """AR fitted on all training series together (no per-series models).

    With --pool overseas_plus_target this is one AR shared by the overseas
    markets and Taiwan, so the overseas data actually affects Taiwan's forecast.
    """

    name = "ar_pooled"

    def __init__(self, p: int = 4, feature: str = "hpi_qoq", min_rows: int = 12):
        super().__init__(p=p, feature=feature, per_series=False, min_rows=min_rows)


REGISTRY = {
    "no_change": NoChange,
    "ar": ARDirect,
    "ar_pooled": ARPooled,
}
