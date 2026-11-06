"""One model per market (transformer_per_market) and the paired ratio bootstrap."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_cpi, synthetic_panel
from housing_forecast.backtesting.infoset import build_samples
from housing_forecast.backtesting.runner import pool_national, run_backtest
from housing_forecast.backtesting.splits import end_series, last_label_origin, make_splits
from housing_forecast.config import load_config
from housing_forecast.data.transforms import prepare_hpi_panel
from housing_forecast.evaluation.markets import (
    metrics_by_market, paired_ratio_bootstrap, relative_to_reference, summary_by_horizon,
)
from housing_forecast.models.baselines import ARDirect, NoChange

LONG = Path(__file__).resolve().parents[1] / "configs" / "experiment_long.toml"


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config(LONG))
    c["periods"]["validation_start"] = "2006Q1"
    c["bootstrap"]["n_boot"] = 200
    return c


@pytest.fixture(scope="module")
def samples(cfg):
    return build_samples(prepare_hpi_panel(synthetic_panel(), synthetic_cpi(), cfg), cfg)


def _run(cfg, samples, factory):
    splits = make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))
    return run_backtest(samples, splits, factory, cfg, pool_national())


@pytest.fixture(scope="module")
def preds(cfg, samples):
    return pd.concat([_run(cfg, samples, NoChange), _run(cfg, samples, ARDirect)], ignore_index=True)


def test_point_estimate_matches_summary_by_horizon(cfg, preds):
    ci = paired_ratio_bootstrap(preds, "ar", "no_change", cfg, min_origins=4, block_length=4)
    rel = relative_to_reference(metrics_by_market(preds), "no_change", min_origins=4)
    curve = summary_by_horizon(rel).query("model == 'ar'").set_index("horizon")["median_rel_rmse"]
    per_h = ci[~ci["horizon"].str.startswith("h")].assign(h=lambda d: d["horizon"].astype(int)).set_index("h")
    np.testing.assert_allclose(per_h["median_ratio"], curve.reindex(per_h.index), rtol=1e-10)
    assert {"h3-6", "h16-20"} <= set(ci["horizon"])
    assert (ci["ci_low"] <= ci["median_ratio"] + 1e-12).all() and (ci["median_ratio"] <= ci["ci_high"] + 1e-12).all()


def test_same_model_gives_ratio_one(cfg, preds):
    twin = preds[preds["model"] == "ar"].assign(model="ar_copy")
    ci = paired_ratio_bootstrap(pd.concat([preds, twin]), "ar", "ar_copy", cfg, min_origins=4, block_length=4)
    np.testing.assert_allclose(ci[["median_ratio", "ci_low", "ci_high"]].to_numpy(), 1.0)


def test_half_the_error_gives_ratio_half_in_every_draw(cfg, preds):
    nc = preds[preds["model"] == "no_change"]
    half = nc.assign(model="half", y_pred=nc["y_true"] + 0.5 * (nc["y_pred"] - nc["y_true"]))
    ci = paired_ratio_bootstrap(pd.concat([preds, half]), "half", "no_change", cfg, min_origins=4, block_length=4)
    np.testing.assert_allclose(ci[["median_ratio", "ci_low", "ci_high"]].to_numpy(), 0.5)
    assert (ci["share_draws_below_1"] == 1.0).all()


def test_unknown_model_is_an_error(cfg, preds):
    with pytest.raises(ValueError):
        paired_ratio_bootstrap(preds, "ar", "missing", cfg)


FAST = dict(d_model=16, n_layers=1, max_epochs=10, patience=3, default_epochs=3)


def _toy(n_long=40, n_short=10):
    from housing_forecast.backtesting.infoset import SampleSet
    rng = np.random.default_rng(0)
    metas, Xs, ys = [], [], []
    for sid, n in [("A", n_long), ("B", n_long), ("Short", n_short)]:
        X = rng.normal(size=(n, 8, 1))
        ys.append(X[:, -1, :1] * 0.02 + rng.normal(0, 0.005, size=(n, 4)))
        Xs.append(X)
        o = pd.period_range("2000Q1", periods=n, freq="Q")
        metas.append(pd.DataFrame({"series_id": sid, "country": sid, "geography_level": "national",
                                   "origin": o, "as_of_date": o.end_time.normalize()}))
    y = np.vstack(ys)
    return SampleSet(X=np.vstack(Xs), y=y, y_avail=np.full(len(y), np.datetime64("NaT")),
                     meta=pd.concat(metas, ignore_index=True), feature_names=["hpi_qoq"], horizons=(1, 2, 3, 4))


def test_per_market_fallback_diagnostics_and_seed():
    pytest.importorskip("torch")
    from housing_forecast.models.transformer import TransformerPerMarket
    s = _toy()
    runs = []
    for _ in range(2):
        m = TransformerPerMarket(min_market_rows=24, **FAST)
        m.fit(s, seed=0)
        runs.append(m.predict(s))
    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-5, atol=1e-7)
    short = (s.meta["series_id"] == "Short").to_numpy()
    assert (runs[0][short] == 0).all() and np.isfinite(runs[0]).all()
    d = m.diagnostics().set_index("series_id")
    assert d.loc["Short", "fit_status"] == "too_short"
    assert d.loc["Short", "n_fallback_forecasts"] == short.sum()
    assert d.loc["A", "fit_status"] == "fitted" and d.loc["A", "n_fallback_forecasts"] == 0


def test_per_market_model_never_sees_other_markets():
    pytest.importorskip("torch")
    from housing_forecast.models.transformer import TransformerPerMarket
    s = _toy()
    other = s.replace(X=s.X.copy(), y=s.y.copy())
    b = (s.meta["series_id"] == "B").to_numpy()
    other.X[b] = other.X[b] * 10 + 3
    other.y[b] = -other.y[b]
    p = []
    for data in (s, other):
        m = TransformerPerMarket(**FAST)
        m.fit(data, seed=0)
        p.append(m.predict(data))
    a = (s.meta["series_id"] == "A").to_numpy()
    np.testing.assert_allclose(p[0][a], p[1][a], rtol=1e-5, atol=1e-7)


def test_unseen_market_falls_back():
    pytest.importorskip("torch")
    from housing_forecast.models.transformer import TransformerPerMarket
    s = _toy()
    m = TransformerPerMarket(**FAST)
    m.fit(s.subset((s.meta["series_id"] != "B").to_numpy()), seed=0)
    out = m.predict(s)
    assert (out[(s.meta["series_id"] == "B").to_numpy()] == 0).all()
    assert m.diagnostics().set_index("series_id").loc["B", "fit_status"] == "no_training_history"


def test_market_without_long_horizon_labels_falls_back_only_at_that_horizon():
    pytest.importorskip("torch")
    from housing_forecast.models.transformer import TransformerPerMarket

    s = _toy()
    market = s.meta["series_id"].eq("A").to_numpy()
    y = s.y.copy()
    y[market, -1] = np.nan
    model = TransformerPerMarket(**FAST)
    model.fit(s.replace(y=y), seed=0)
    predictions = model.predict(s)
    assert np.isfinite(predictions).all()
    assert (predictions[market, -1] == 0).all()
    assert np.any(predictions[market, 0] != 0)
    diag = model.diagnostics().set_index("series_id")
    assert diag.loc["A", "fallback_horizons"] == "4"
    assert diag.loc["A", "n_fallback_forecasts"] == market.sum()
