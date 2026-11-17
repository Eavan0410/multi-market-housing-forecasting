"""Input-noise robustness: the no-noise run matches the normal backtest, overlapping
windows, folds and models share the same noise (one fixed scale), and band RMSE follows the main experiment."""

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_cpi, synthetic_panel
from housing_forecast.backtesting.infoset import build_samples
from housing_forecast.backtesting.robustness import band_metrics, noise_backtest, noisy_inputs
from housing_forecast.backtesting.runner import pool_overseas, run_backtest, training_samples
from housing_forecast.backtesting.splits import last_label_origin, make_splits
from housing_forecast.data.transforms import prepare_hpi_panel
from housing_forecast.models.base import Forecaster
from housing_forecast.models.baselines import ARDirect, NoChange


class LastValue(Forecaster):
    """Scaled, 'stochastic' toy model: forecast = last scaled input, for every horizon."""
    name, stochastic, needs_scaling = "last_value", True, True

    def fit(self, train, seed):
        pass

    def predict(self, test):
        return np.repeat(test.X[:, -1, :1], len(test.horizons), axis=1)


@pytest.fixture
def setup(cfg):
    samples = build_samples(prepare_hpi_panel(synthetic_panel(), synthetic_cpi(), cfg), cfg)
    splits = make_splits(cfg, "validation", last_label_origin(samples, cfg["target"]["series_id"]))
    return samples, splits


def _run(setup, cfg, factory, **kw):
    samples, splits = setup
    return noise_backtest(samples, splits, factory, cfg, pool_overseas(), levels=[0.5, 1.0], draws=2, **kw)


def test_no_noise_reproduces_backtest(setup, cfg):
    samples, splits = setup
    for factory in (ARDirect, LastValue):
        p = run_backtest(samples, splits, factory, cfg, pool_overseas())
        p = p.assign(se=(p["y_pred"] - p["y_true"]) ** 2).groupby(["seed", "series_id", "horizon"])["se"].sum()
        e = _run(setup, cfg, factory)
        zero = e[e["noise_level"].eq(0)].set_index(["seed", "series_id", "horizon"])["sse"]
        assert np.allclose(zero.sort_index().to_numpy(), p.sort_index().to_numpy())
        assert e.groupby("noise_level")["draw"].nunique().to_dict() == {0.0: 1, 0.5: 2, 1.0: 2}
        assert (e.loc[e["noise_level"].eq(0), "sum_abs_change"] == 0).all()


def test_overlapping_windows_share_the_same_noise(setup):
    samples, _ = setup
    test = samples.subset((samples.meta["series_id"] == "Japan").to_numpy())
    noise = noisy_inputs(test, 0, 0.02, 1.0, 0, 2026)[:, :, 0] - test.X[:, :, 0]
    origins = pd.PeriodIndex(test.meta["origin"], freq="Q")
    r = int(np.where((origins[1:] - origins[:-1]).map(lambda d: d.n) == 1)[0][0])
    assert np.allclose(noise[r, 1:], noise[r + 1, :-1])          # same quarters, same error
    assert np.allclose(noisy_inputs(test, 0, 0.02, 0.5, 0, 2026)[:, :, 0] - test.X[:, :, 0], noise / 2)
    assert not np.allclose(noisy_inputs(test, 0, 0.02, 1.0, 1, 2026)[:, :, 0] - test.X[:, :, 0], noise)


def test_one_noise_scale_for_all_folds(setup, cfg):
    samples, splits = setup
    first = float(np.nanstd(training_samples(samples, splits[0], pool_overseas()).X[:, :, 0]))
    assert np.isclose(_run(setup, cfg, ARDirect).attrs["noise_sd"], first)


def test_no_change_ignores_noise_and_draws_repeat(setup, cfg):
    nc = _run(setup, cfg, NoChange)
    totals = nc.groupby(["noise_level", "draw"])["sse"].sum()
    assert np.allclose(totals.to_numpy(), totals.iloc[0])
    first, again = _run(setup, cfg, ARDirect), _run(setup, cfg, ARDirect)
    assert np.array_equal(first["sse"].to_numpy(), again["sse"].to_numpy())
    noisy = first["noise_level"].gt(0)
    assert (first.loc[noisy, "sum_abs_change"] > 0).any()


def test_band_rmse_pools_horizons_then_averages_seeds():
    e = pd.DataFrame({"model": "m", "noise_level": 0.0, "draw": 0, "series_id": "A",
                      "seed": [0, 0, 1, 1], "horizon": [3, 4, 3, 4], "n": [10, 10, 10, 10],
                      "sse": [1.0, 3.0, 4.0, 4.0], "sum_abs_change": 0.0})
    m = band_metrics(e, [3, 4], min_origins=8)
    expected = (np.sqrt(4.0 / 20) + np.sqrt(8.0 / 20)) / 2
    assert np.isclose(m["rmse"].iloc[0], expected)
    assert band_metrics(e, [3, 4], min_origins=11).empty
