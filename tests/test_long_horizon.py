"""Multi-market, 1–20 quarter setup (configs/experiment_long.toml) on synthetic data."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_cpi, synthetic_panel
from housing_forecast.backtesting.infoset import build_samples
from housing_forecast.backtesting.runner import pool_national, run_backtest
from housing_forecast.backtesting.splits import last_label_origin, make_splits
from housing_forecast.config import check_config, load_config
from housing_forecast.data.contract import check_alignment
from housing_forecast.data.transforms import prepare_hpi_panel
from housing_forecast.evaluation.markets import (
    history_groups, metrics_by_market, relative_to_reference, summary_by_history, summary_by_horizon,
)
from housing_forecast.models.baselines import ARDirect, ARPooled, NoChange

LONG = Path(__file__).resolve().parents[1] / "configs" / "experiment_long.toml"


@pytest.fixture
def long_cfg():
    cfg = copy.deepcopy(load_config(LONG))
    # Synthetic data start in 1995, so the first fold needs a later start to have any h=20
    # training labels (real data go back to 1970).
    cfg["periods"]["validation_start"] = "2006Q1"
    return cfg


@pytest.fixture
def long_samples(long_cfg):
    return build_samples(prepare_hpi_panel(synthetic_panel(), synthetic_cpi(), long_cfg), long_cfg)


def test_long_config_is_valid_and_keeps_validation_before_test(long_cfg):
    p = long_cfg["periods"]
    assert max(long_cfg["target"]["horizons"]) == 20
    assert pd.Period(p["validation_end"], "Q") + 20 <= pd.Period(p["test_start"], "Q")


def test_short_gap_needs_explicit_flag(long_cfg):
    long_cfg["periods"]["allow_short_gap"] = False
    with pytest.raises(ValueError):
        check_config(long_cfg)


def _run(cfg, samples, factory):
    splits = make_splits(cfg, "validation", last_label_origin(samples, cfg["target"]["series_id"]))
    return run_backtest(samples, splits, factory, cfg, pool_national())   # check_no_leakage runs on every fold


def test_all_national_markets_scored_up_to_h20_without_leakage(long_cfg, long_samples):
    preds = _run(long_cfg, long_samples, NoChange)
    assert set(preds["series_id"]) == {"Taiwan_national", "Japan", "United States"}   # no city series
    assert preds["horizon"].max() == 20
    origin = pd.PeriodIndex(preds["origin_quarter"], freq="Q")
    assert origin.max() <= pd.Period(long_cfg["periods"]["validation_end"], "Q")


def test_per_market_and_pooled_ar_align_and_summarise(long_cfg, long_samples, tmp_path):
    preds = pd.concat([
        _run(long_cfg, long_samples, NoChange),
        _run(long_cfg, long_samples, ARDirect).assign(model="ar__national"),
        _run(long_cfg, long_samples, ARPooled).assign(model="ar_pooled__national"),
    ], ignore_index=True)
    check_alignment(preds)
    by_market = metrics_by_market(preds)
    rel = relative_to_reference(by_market, "no_change", min_origins=4)
    assert np.allclose(rel.loc[rel["model"] == "no_change", "rel_rmse"], 1.0)
    curve = summary_by_horizon(rel)
    assert set(curve["horizon"]) == set(range(1, 21))
    hist = summary_by_history(rel, history_groups(synthetic_panel()))
    assert {"h1-2", "h3-6", "h7-15", "h16-20"} <= set(hist["horizon_band"])


def test_relative_metric_drops_thin_cells():
    bm = pd.DataFrame({"model": ["no_change", "m"] * 2, "series_id": ["A", "A", "B", "B"],
                       "horizon": [1, 1, 1, 1], "n_origins": [10, 10, 3, 3],
                       "rmse": [2.0, 1.0, 2.0, 1.0], "mae": [1.0, 1.0, 1.0, 1.0], "dir_acc": [0.5] * 4})
    rel = relative_to_reference(bm, "no_change", min_origins=8)
    assert set(rel["series_id"]) == {"A"}
    assert rel.loc[rel["model"] == "m", "rel_rmse"].item() == 0.5


def test_transformer_inner_gap_covers_longest_horizon(long_samples):
    pytest.importorskip("torch")
    from housing_forecast.models.transformer import TransformerForecaster
    m = TransformerForecaster(val_quarters=8, gap_quarters=4)
    split = m._inner_split(long_samples)
    tr, va = split
    o_tr = pd.PeriodIndex(tr.meta["origin"], freq="Q").max()
    o_va = pd.PeriodIndex(va.meta["origin"], freq="Q").min()
    assert (o_va - o_tr).n > 20            # train labels (up to t+20) end before validation origins


def test_per_market_ar_falls_back_per_horizon_only():
    from housing_forecast.backtesting.infoset import SampleSet
    rng = np.random.default_rng(0)
    n_long, n_short = 40, 14
    X = rng.normal(size=(n_long + n_short, 8, 1))
    y = X[:, -1, :1] * np.array([[0.5, 1.0]]) + rng.normal(0, 0.01, (n_long + n_short, 2))
    y[n_long:, 1] = np.nan                         # short market: enough rows for h=1 only
    meta = pd.DataFrame({"series_id": ["L"] * n_long + ["S"] * n_short, "country": "x",
                         "geography_level": "national",
                         "origin": list(pd.period_range("1990Q1", periods=n_long, freq="Q"))
                         + list(pd.period_range("2000Q1", periods=n_short, freq="Q"))})
    s = SampleSet(X=X, y=y, y_avail=np.full(y.shape, np.datetime64("NaT")), meta=meta,
                  feature_names=["hpi_qoq"], horizons=(1, 2))
    m = ARDirect(p=1)
    m.fit(s, seed=0)
    assert not np.isnan(m.coefs_["S"][:, 0]).any()          # own fit kept for h=1
    assert np.isnan(m.coefs_["S"][:, 1]).all()              # h=2 not fitted for S ...
    pred = m.predict(s.subset((meta["series_id"] == "S").to_numpy()))
    Z = np.column_stack([np.ones(n_short), X[n_long:, -1, 0]])
    assert np.allclose(pred[:, 1], Z @ m.coefs_["__pooled__"][:, 1])   # ... so it uses the pooled fit
    assert np.allclose(pred[:, 0], Z @ m.coefs_["S"][:, 0])


def test_common_forecast_date_for_every_market(long_samples):
    end = pd.PeriodIndex(long_samples.meta["origin"], freq="Q").end_time.normalize()
    lag = (long_samples.meta["as_of_date"] - end).dt.days
    assert (lag == 122).all()


def test_labels_keep_each_market_publication_date(long_samples):
    s = long_samples
    end1 = (pd.PeriodIndex(s.meta["origin"], freq="Q") + 1).end_time.normalize()
    pub = pd.Series(s.y_avail[:, 0]) - pd.Series(end1)
    tw = (s.meta["country"] == "Taiwan").to_numpy() & ~np.isnan(s.y[:, 0])
    jp = (s.meta["country"] == "Japan").to_numpy() & ~np.isnan(s.y[:, 0])
    assert (pub[tw].dt.days == 60).all() and (pub[jp].dt.days == 100).all()


def test_fold_cutoff_does_not_depend_on_target_country(long_cfg):
    a = make_splits(long_cfg, "validation", pd.Period("2025Q1", "Q"))
    other = copy.deepcopy(long_cfg)
    other["target"]["country"] = "Japan"
    b = make_splits(other, "validation", pd.Period("2025Q1", "Q"))
    assert [s.cutoff_date for s in a] == [s.cutoff_date for s in b]
    first = pd.Period(long_cfg["periods"]["validation_start"], "Q")
    assert a[0].cutoff_date == first.end_time.normalize() + pd.Timedelta(days=122)


def test_forecast_lag_must_cover_slowest_hpi_release(long_cfg):
    long_cfg["periods"]["forecast_lag_days"] = 60
    with pytest.raises(ValueError):
        check_config(long_cfg)


def test_original_rule_unchanged_without_forecast_lag(cfg):
    from housing_forecast.config import forecast_lag_days
    assert forecast_lag_days(cfg) is None
