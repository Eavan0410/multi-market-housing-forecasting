"""The five fair-test rules, as executable checks.  Run:  pytest -v"""

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_cpi, synthetic_panel
from housing_forecast.backtesting.infoset import build_samples, delay_quarters
from housing_forecast.backtesting.runner import pool_overseas, pool_target_only, run_backtest
from housing_forecast.backtesting.scaling import WindowStandardizer
from housing_forecast.backtesting.splits import (
    PeriodLockedError, check_no_leakage, last_label_origin, make_splits,
    mask_unpublished_labels, train_mask,
)
from housing_forecast.data.contract import ContractError, check_alignment, validate_predictions
from housing_forecast.data.transforms import prepare_hpi_panel
from housing_forecast.models.baselines import ARDirect, NoChange


@pytest.fixture
def samples(cfg):
    prepared = prepare_hpi_panel(synthetic_panel(), synthetic_cpi(), cfg)
    return build_samples(prepared, cfg)


def _splits(cfg, samples):
    return make_splits(cfg, "validation", last_label_origin(samples, cfg["target"]["series_id"]))


# Rule 1 + 2: expanding window with a 4-quarter gap ------------------------------------

def test_expanding_window_and_gap(cfg, samples):
    splits = _splits(cfg, samples)
    ends = [s.train_origin_end for s in splits]
    assert ends == sorted(ends), "training window must only grow"
    for s in splits:
        assert (s.first_test_origin - s.train_origin_end).n == cfg["periods"]["gap_quarters"]
        train = mask_unpublished_labels(samples.subset(train_mask(samples, s)), s)
        tr_origin = pd.PeriodIndex(train.meta["origin"], freq="Q")
        for j, h in enumerate(train.horizons):
            vis = ~np.isnan(train.y[:, j])
            assert (tr_origin[vis] + h).max() <= s.first_test_origin   # no label overlaps the test period


def test_leakage_check_catches_overlap(cfg, samples):
    s = _splits(cfg, samples)[0]
    leaky = samples.subset((samples.meta["origin"] <= s.first_test_origin).to_numpy())
    test = samples.subset((samples.meta["origin"] == s.first_test_origin).to_numpy())
    with pytest.raises(AssertionError):
        check_no_leakage(leaky, test, s, cfg["periods"]["gap_quarters"])


# Rule 3: one calendar cutoff for every country -----------------------------------------

def test_same_calendar_cutoff_for_all_countries(cfg, samples):
    s = _splits(cfg, samples)[0]
    train = samples.subset(train_mask(samples, s))
    by_country = train.meta.groupby("country")["origin"].max()
    assert (by_country <= s.train_origin_end).all()


def test_unpublished_labels_are_hidden(cfg, samples):
    # Overseas HPI is published later (100 days) than Taiwan's (60), so some overseas
    # labels that end exactly at the cutoff quarter are not yet public and must be hidden.
    cfg["periods"]["gap_quarters"] = 4
    s = _splits(cfg, samples)[0]
    train = mask_unpublished_labels(samples.subset(train_mask(samples, s)), s)
    visible = ~np.isnan(train.y)
    assert (train.y_avail[visible] <= np.datetime64(s.cutoff_date)).all()


def test_release_lag_delays_features():
    assert delay_quarters(30, 60) == 0          # published before HPI -> same quarter usable
    assert delay_quarters(100, 60) == 1         # published after HPI  -> one quarter older
    assert delay_quarters(200, 60) == 2


def test_as_of_date_after_origin(samples):
    end = pd.PeriodIndex(samples.meta["origin"], freq="Q").end_time.normalize()
    assert (samples.meta["as_of_date"] > end).all()


def test_test_period_is_locked(cfg, samples):
    with pytest.raises(PeriodLockedError):
        make_splits(cfg, "test", pd.Period("2025Q2", "Q"))


def test_validation_labels_stay_before_test(cfg):
    from housing_forecast.config import check_config
    cfg["periods"]["validation_end"] = "2018Q3"     # 2018Q3 + 4 = 2019Q3 > test_start 2019Q1
    with pytest.raises(ValueError):
        check_config(cfg)


# Rule 4: fitting only inside the training window ---------------------------------------

def test_scaler_uses_training_window_only():
    X_train = np.ones((10, 8, 1))
    X_test = np.full((5, 8, 1), 100.0)
    sc = WindowStandardizer().fit(X_train)
    assert np.allclose(sc.mean_, 1.0)                # test values never touched the fit
    assert np.isfinite(sc.transform(X_test)).all()


# Rule 5: same folds / same rows / shared format for every model --------------------------

def test_models_row_aligned_and_valid(cfg, samples):
    splits = _splits(cfg, samples)
    target = cfg["target"]["series_id"]
    a = run_backtest(samples, splits, NoChange, cfg, pool_target_only(target), "no_change")
    b = run_backtest(samples, splits, ARDirect, cfg, pool_target_only(target), "ar")
    c = run_backtest(samples, splits, lambda: ARDirect(per_series=False), cfg, pool_overseas(), "ar_overseas")
    both = pd.concat([a, b, c])
    validate_predictions(both)
    check_alignment(both)
    assert set(both["series_id"]) == {target}


def test_pretraining_pool_excludes_taiwan(samples):
    m = pool_overseas("Taiwan")(samples.meta)
    assert not samples.meta.loc[m, "country"].eq("Taiwan").any()


def test_alignment_check_fails_on_missing_rows(cfg, samples):
    splits = _splits(cfg, samples)
    target = cfg["target"]["series_id"]
    a = run_backtest(samples, splits, NoChange, cfg, pool_target_only(target), "no_change")
    b = a.assign(model="other").iloc[1:]
    with pytest.raises(ContractError):
        check_alignment(pd.concat([a, b]))


def test_taiwan_own_history_is_a_feature(samples):
    assert "hpi_qoq" in samples.feature_names
