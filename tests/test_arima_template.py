"""Checks for the parts of the ARIMA template that A wrote (history building, cumulative
forecasts, fallbacks). B's fit_one / forecast_path are replaced by a simple stand-in here."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_cpi, synthetic_panel
from housing_forecast.backtesting.infoset import build_samples
from housing_forecast.backtesting.runner import pool_national, run_backtest
from housing_forecast.backtesting.splits import last_label_origin, make_splits
from housing_forecast.data.transforms import prepare_panel
from housing_forecast.models import arima
from housing_forecast.models.arima import ARIMAPerMarket, extend_history, series_histories


@pytest.fixture
def prepared(cfg):
    return prepare_panel(synthetic_panel(), synthetic_cpi(), cfg)


@pytest.fixture
def samples(cfg, prepared):
    return build_samples(prepared, cfg)


def test_history_matches_real_log_growth(prepared, samples):
    h = series_histories(samples)["Japan"]
    g = prepared[prepared["series_id"] == "Japan"].assign(q=lambda d: pd.PeriodIndex(d["quarter"], freq="Q"))
    g = np.log(g.set_index("q")["hpi_real"]).diff()
    q = pd.PeriodIndex.from_ordinals(h.index.to_numpy(), freq="Q")
    assert np.allclose(h.to_numpy(), g.reindex(q).to_numpy())
    assert (np.diff(h.index.to_numpy()) == 1).all()          # one unbroken quarterly history


def test_extend_history_adds_quarters_known_at_forecast_date(samples):
    tr = samples.subset(np.asarray(pd.PeriodIndex(samples.meta["origin"], freq="Q") <= pd.Period("2010Q4", "Q")))
    hist = series_histories(tr)["Japan"]
    row = np.flatnonzero(((samples.meta["series_id"] == "Japan")
                          & (pd.PeriodIndex(samples.meta["origin"], freq="Q") == pd.Period("2012Q2", "Q"))).to_numpy())[0]
    ext = extend_history(hist, samples, row)
    assert ext.index.max() == pd.Period("2012Q2", "Q").ordinal
    assert ext.index.min() == hist.index.min() and len(ext) == len(hist) + 6


def test_missing_statsmodels_stops_the_run(monkeypatch, samples):
    def missing(*args, **kwargs):
        raise ImportError("No module named 'statsmodels'")

    monkeypatch.setattr(arima, "fit_one", missing)
    with pytest.raises(ImportError):
        ARIMAPerMarket().fit(samples, seed=0)


def test_arima_fits_and_forecasts_real_histories(samples):
    pytest.importorskip("statsmodels")
    model = ARIMAPerMarket(max_p=2, max_q=1)
    model.fit(samples, seed=0)
    assert model.models_                                       # at least one market fitted
    pred = model.predict(samples)
    assert pred.shape == samples.y.shape and np.isfinite(pred).all()
    d = model.diagnostics().set_index("series_id")
    fitted = d["fit_status"].eq("fitted")
    assert (d.loc[fitted, "n_fallback_forecasts"] == 0).all()


def test_cumulative_forecasts_with_stand_in(monkeypatch, cfg, samples):
    # Stand-in for B's code: forecast every future quarter at the training mean.
    monkeypatch.setattr(arima, "fit_one", lambda h, *a, **k: float(np.mean(h)))
    monkeypatch.setattr(arima, "forecast_path", lambda m, h, steps: np.full(steps, m))
    cfg["target"]["eval_series"] = "all_national"
    splits = make_splits(cfg, "validation", last_label_origin(samples, cfg["target"]["series_id"]))
    preds = run_backtest(samples, splits, ARIMAPerMarket, cfg, pool_national())
    diag = preds.attrs["diagnostics"]
    assert {"split_id", "series_id", "n_train_obs", "fit_status", "reason",
            "n_forecasts", "n_fallback_forecasts"} <= set(diag.columns)
    assert diag["split_id"].nunique() == len(splits)
    one = preds[(preds["series_id"] == "Japan") & (preds["origin_quarter"] == preds["origin_quarter"].min())]
    by_h = one.set_index("horizon")["y_pred"]
    assert np.allclose(by_h.to_numpy(), by_h.loc[1] * by_h.index.to_numpy())   # cumulative = h x quarterly mean


def test_short_market_falls_back_and_is_recorded(monkeypatch, samples):
    monkeypatch.setattr(arima, "fit_one", lambda h, *a, **k: 0.0)
    monkeypatch.setattr(arima, "forecast_path", lambda m, h, steps: np.zeros(steps))
    m = ARIMAPerMarket(min_obs=10_000)
    m.fit(samples, seed=0)
    assert set(m.fallbacks_) == set(samples.meta["series_id"])
    d = m.diagnostics().set_index("series_id")
    assert (d["fit_status"] == "too_short").all()
    assert np.isfinite(m.predict(samples)).all()


@pytest.mark.parametrize("bad", [lambda steps: np.full(steps, np.nan), lambda steps: np.zeros(steps + 1)])
def test_bad_forecast_falls_back_and_is_recorded(monkeypatch, samples, bad):
    monkeypatch.setattr(arima, "fit_one", lambda h, *a, **k: 0.0)
    monkeypatch.setattr(arima, "forecast_path", lambda m, h, steps: bad(steps))
    m = ARIMAPerMarket()
    m.fit(samples, seed=0)
    pred = m.predict(samples)
    assert np.isfinite(pred).all()
    d = m.diagnostics().set_index("series_id")
    assert (d["fit_status"] == "fitted, some forecasts failed").all()
    assert (d["n_fallback_forecasts"] == d["n_forecasts"]).all() and (d["n_forecasts"] > 0).all()


def test_unfinished_forecast_code_raises_instead_of_falling_back(monkeypatch, samples):
    def unfinished(*args, **kwargs):
        raise NotImplementedError("forecast_path")

    monkeypatch.setattr(arima, "fit_one", lambda h, *a, **k: 0.0)
    monkeypatch.setattr(arima, "forecast_path", unfinished)
    m = ARIMAPerMarket()
    m.fit(samples, seed=0)
    with pytest.raises(NotImplementedError):
        m.predict(samples)


@pytest.mark.parametrize("models", [("no_change", "arima"), ("arima", "no_change")])
def test_cli_mixed_models_saves_diagnostics(monkeypatch, tmp_path, cfg, capsys, models):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_backtest.py"
    spec = importlib.util.spec_from_file_location("run_backtest_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cfg["features"]["use"] = ["hpi_qoq"]
    cfg["target"]["eval_series"] = "all_national"
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(arima, "fit_one", lambda h, *a, **k: float(np.mean(h)))
    monkeypatch.setattr(arima, "forecast_path", lambda m, h, steps: np.full(steps, np.nan))
    panel, cpi = tmp_path / "panel.csv", tmp_path / "cpi.csv"
    synthetic_panel().to_csv(panel, index=False)
    synthetic_cpi().to_csv(cpi, index=False)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--models", *models, "--pool", "national",
        "--panel", str(panel), "--cpi", str(cpi),
        "--out-dir", str(tmp_path), "--experiment", "mixed",
    ])

    cli.main()

    out = tmp_path / "mixed" / "validation"
    for model in models:
        assert (out / f"preds_{model}__national.meta.json").exists()
        preds = pd.read_csv(out / f"preds_{model}__national.csv")
        assert np.isfinite(preds["y_pred"]).all()
    diag = pd.read_csv(out / "diag_arima__national.csv")
    assert diag["split_id"].nunique() > 1
    assert (diag["n_fallback_forecasts"] == diag["n_forecasts"]).all()
    assert diag["n_fallback_forecasts"].sum() > 0
    assert not (out / "diag_no_change__national.csv").exists()
    assert "all models row-aligned" in capsys.readouterr().out
