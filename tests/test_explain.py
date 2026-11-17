"""Lag weights and attributions: AR contributions add up, and the Transformer gradients and
Integrated Gradients match a network whose answer is known."""

import numpy as np
import pandas as pd
import pytest

from housing_forecast.backtesting.infoset import SampleSet
from housing_forecast.backtesting.scaling import WindowStandardizer
from housing_forecast.evaluation.explain import ar_attributions, ar_lag_weights, transformer_attributions
from housing_forecast.models.baselines import ARPooled, ARPooled8


def _samples(n=80, L=8, H=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.01, 0.02, size=(n, L, 1))
    y = X[:, -1, :1] * np.array([1.0, 1.5, 2.0]) + rng.normal(0, 0.001, size=(n, H))
    meta = pd.DataFrame({"series_id": "A", "country": "A", "geography_level": "national",
                         "origin": pd.period_range("1980Q1", periods=n, freq="Q"),
                         "as_of_date": pd.Timestamp("2000-01-01")})
    return SampleSet(X=X, y=y, y_avail=np.full((n, H), np.datetime64("NaT")), meta=meta,
                     feature_names=["hpi_qoq"], horizons=(1, 2, 3))


@pytest.mark.parametrize("factory", [ARPooled, ARPooled8])
def test_ar_contributions_add_up_to_forecast_minus_baseline(factory):
    s = _samples()
    ar = factory()
    ar.fit(s, 0)
    base = float(s.X[..., 0].mean())
    att = ar_attributions(ar, s.X, 0, base)
    flat = s.replace(X=np.full_like(s.X, base))
    assert np.allclose(att.sum(axis=2), ar.predict(s) - ar.predict(flat))
    W = ar_lag_weights(ar, 8)
    assert W.shape == (3, 8) and (W[:, ar.p:] == 0).all()
    assert np.allclose(W[:, 0], ar.coefs_["__pooled__"][1])     # lag 1 = most recent quarter


def test_transformer_gradients_and_ig_match_a_known_network():
    torch = pytest.importorskip("torch")
    s = _samples()
    scaler = WindowStandardizer().fit(s.X)

    class Net(torch.nn.Module):            # out_h = sum_i a_i * x_i^2 * (h+1): nonlinear, known gradient
        def forward(self, x):
            a = torch.arange(1, 9, dtype=x.dtype)
            q = (a * x[:, :, 0] ** 2).sum(dim=1, keepdim=True)
            return q * torch.tensor([1.0, 2.0, 3.0])

    class Model:
        net = Net()
        y_mean_ = np.array([0.1, 0.2, 0.3])
        y_std_ = np.array([2.0, 2.0, 2.0])

    att = transformer_attributions(Model(), scaler, s.X, steps=64)
    xs = scaler.transform(s.X)[..., 0]                            # (n, L), window order
    a = np.arange(1, 9)
    for h, k in enumerate((1.0, 2.0, 3.0)):
        grad_scaled = 2 * a * xs * k                              # d out / d x_scaled
        expected = (grad_scaled * 2.0 / scaler.std_[0])[:, ::-1]  # raw units, lag order
        assert np.allclose(att["grad"][:, h, :], expected, rtol=1e-4, atol=1e-6)
        ig_expected = (a * xs ** 2 * k * 2.0)[:, ::-1]           # exact IG for a quadratic from 0
        assert np.allclose(att["ig"][:, h, :], ig_expected, rtol=1e-3, atol=1e-6)
    assert att["completeness_error"] < 1e-3
    assert np.allclose(att["pred_baseline"], Model.y_mean_)
