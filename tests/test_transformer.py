"""Quick checks for the Transformer forecasters (skipped when torch is not installed)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from housing_forecast.backtesting.infoset import SampleSet  # noqa: E402
from housing_forecast.models.transformer import (  # noqa: E402
    TransformerFineTune, TransformerForecaster, TransformerFTHead, TransformerFTLowLR, TransformerFTShort,
    TransformerMarket, SharedTransformerFineTune, SharedTransformerFineTuneSelect,
)

FAST = dict(d_model=16, n_layers=1, max_epochs=15, patience=5, default_epochs=5)


def toy_samples(n_per_series: int = 60, lookback: int = 8, n_features: int = 2, seed: int = 0) -> SampleSet:
    rng = np.random.default_rng(seed)
    metas, Xs, ys = [], [], []
    for sid, country in [("Taiwan_national", "Taiwan"), ("Japan", "Japan"), ("United States", "United States")]:
        X = rng.normal(size=(n_per_series, lookback, n_features))
        y = X[:, -1, :1] * 0.02 + rng.normal(0, 0.005, size=(n_per_series, 4))
        origins = pd.period_range("2000Q1", periods=n_per_series, freq="Q")
        metas.append(pd.DataFrame({"series_id": sid, "country": country, "geography_level": "national",
                                   "origin": origins, "as_of_date": origins.end_time.normalize()}))
        Xs.append(X)
        ys.append(y)
    y = np.vstack(ys)
    y[-3:, 2:] = np.nan                                     # some hidden labels
    return SampleSet(X=np.vstack(Xs), y=y, y_avail=np.full(len(y), np.datetime64("NaT")),
                     meta=pd.concat(metas, ignore_index=True),
                     feature_names=[f"f{i}" for i in range(n_features)], horizons=(1, 2, 3, 4))


@pytest.mark.parametrize("cls", [TransformerForecaster, TransformerFineTune,
                                 TransformerFTHead, TransformerFTShort, TransformerFTLowLR,
                                 TransformerMarket])
def test_shape_and_same_seed_same_result(cls):
    s = toy_samples()
    preds = []
    for _ in range(2):
        m = cls(**FAST)
        m.fit(s, seed=0)
        preds.append(m.predict(s))
    assert preds[0].shape == s.y.shape
    assert np.isfinite(preds[0]).all()
    np.testing.assert_allclose(preds[0], preds[1], rtol=1e-5, atol=1e-7)


def test_finetune_runs_two_stages():
    m = TransformerFineTune(**FAST)
    m.fit(toy_samples(), seed=1)
    assert len(m.epochs_used_) == 2                          # pretrain + fine-tune


def test_finetune_needs_overseas_rows():
    s = toy_samples()
    taiwan = s.subset((s.meta["country"] == "Taiwan").to_numpy())
    with pytest.raises(ValueError):
        TransformerFineTune(**FAST).fit(taiwan, seed=0)


def test_head_variant_only_trains_the_head():
    m = TransformerFTHead(**FAST)
    m.fit(toy_samples(), seed=0)
    trainable = {n for n, p in m.net.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("norm.", "head.")) for n in trainable)


def test_variant_settings():
    assert TransformerFTShort().ft_max_epochs == 20
    assert TransformerFTLowLR().ft_lr_scale == 0.01
    assert TransformerFTHead().ft_trainable == "head"


def test_market_embedding_and_unknown_market():
    s = toy_samples()
    m = TransformerMarket(**FAST)
    m.fit(s, seed=0)
    assert m.net.market is not None and m.net.market.num_embeddings == 4    # 3 markets + unknown
    assert set(m.market_index_.values()) == {1, 2, 3}
    unseen = s.replace(meta=s.meta.assign(series_id="Mars"))                 # falls back to slot 0
    assert np.isfinite(m.predict(unseen)).all()


def test_plain_transformer_has_no_market_embedding():
    m = TransformerForecaster(**FAST)
    m.fit(toy_samples(), seed=0)
    assert m.net.market is None


def test_local_adaptation_keeps_the_shared_checkpoint_and_other_markets_independent():
    samples = toy_samples()
    runs = []
    for names in [["Japan"], ["Japan", "Taiwan_national"]]:
        model = SharedTransformerFineTune(adapt_markets=names, ft_epochs=2, **FAST)
        model.fit(samples, seed=0)
        runs.append((model.predict(samples), model.predict_shared(samples)))
    japan = samples.meta["series_id"].eq("Japan").to_numpy()
    np.testing.assert_allclose(runs[0][0][japan], runs[1][0][japan], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(runs[0][1], runs[1][1], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(runs[0][0][~japan], runs[0][1][~japan])
    original = TransformerForecaster(**FAST)
    original.fit(samples, seed=0)
    np.testing.assert_allclose(runs[0][1], original.predict(samples), rtol=1e-5, atol=1e-7)


def test_local_adaptation_falls_back_to_shared_for_missing_labels():
    samples = toy_samples()
    japan = samples.meta["series_id"].eq("Japan").to_numpy()
    labels = samples.y.copy()
    labels[japan, -1] = np.nan
    model = SharedTransformerFineTune(adapt_markets=["Japan"], ft_epochs=2, **FAST)
    model.fit(samples.replace(y=labels), seed=0)
    pred = model.predict(samples)
    shared = model.predict_shared(samples)
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred[japan, -1], shared[japan, -1])
    diag = model.diagnostics().set_index("series_id")
    assert diag.loc["Japan", "fallback_horizons"] == "4"
    assert diag.loc["Japan", "n_train_rows"] == japan.sum()
    short = SharedTransformerFineTune(min_market_rows=1000, **FAST)
    short.fit(samples, seed=0)
    np.testing.assert_allclose(short.predict(samples), short.predict_shared(samples))


def test_runner_exports_the_matching_shared_predictions(cfg):
    from conftest import synthetic_cpi, synthetic_panel
    from housing_forecast.backtesting.infoset import build_samples
    from housing_forecast.backtesting.runner import pool_national, run_backtest
    from housing_forecast.backtesting.splits import last_label_origin, make_splits
    from housing_forecast.data.contract import check_alignment
    from housing_forecast.data.transforms import prepare_hpi_panel

    cfg["features"]["use"] = ["hpi_qoq"]
    cfg["run"]["seeds"] = [0]
    samples = build_samples(prepare_hpi_panel(synthetic_panel(), synthetic_cpi(), cfg), cfg)
    splits = make_splits(cfg, "validation", last_label_origin(samples, cfg["target"]["series_id"]))[:1]
    pred = run_backtest(samples, splits,
                        lambda: SharedTransformerFineTune(ft_epochs=2, **FAST), cfg, pool_national(),
                        reference_model_name="shared")
    reference = pred.attrs.pop("reference_predictions")
    diag = pred.attrs.pop("diagnostics")
    check_alignment(pd.concat([pred, reference], ignore_index=True),
                    expected_seeds={"transformer_shared_ft": [0], "shared": [0]})
    assert len(diag[diag["stage"].eq("shared")]) == 1


def test_epoch_selection_uses_only_inner_training_rows(monkeypatch):
    samples = toy_samples()
    seen = []
    original_fit = TransformerForecaster.fit

    def recording_fit(self, train, seed):
        seen.append(pd.PeriodIndex(train.meta["origin"], freq="Q").max())
        return original_fit(self, train, seed)

    monkeypatch.setattr(TransformerForecaster, "fit", recording_fit)
    model = SharedTransformerFineTuneSelect(ft_max_epochs=3, **FAST)
    model.fit(samples, seed=0)
    last = pd.PeriodIndex(samples.meta["origin"], freq="Q").max()
    # First fit = inner pretraining: it ends val_quarters (8) + gap (4) before the last origin.
    assert seen[0] <= last - 12
    assert seen[1] == last                                   # then the full-fold pretraining
    diag = model.diagnostics().set_index("series_id")
    chosen = diag.loc[diag["stage"].eq("adaptation")]
    assert chosen["select_reason"].eq("selected").all()
    assert chosen["selected_epochs"].between(0, 3).all()
    fine_tuned = chosen["fit_status"].eq("fine_tuned")
    assert (chosen.loc[fine_tuned, "epochs"] == chosen.loc[fine_tuned, "selected_epochs"]).all()
    assert (chosen.loc[~fine_tuned, "selected_epochs"] == 0).all()


def test_epoch_selection_is_repeatable_and_zero_keeps_shared():
    samples = toy_samples()
    runs = []
    for _ in range(2):
        model = SharedTransformerFineTuneSelect(ft_max_epochs=3, **FAST)
        model.fit(samples, seed=0)
        runs.append((model.predict(samples), model.diagnostics()))
    np.testing.assert_allclose(runs[0][0], runs[1][0], rtol=1e-5, atol=1e-7)
    pd.testing.assert_series_equal(runs[0][1]["selected_epochs"], runs[1][1]["selected_epochs"])
    strict = SharedTransformerFineTuneSelect(ft_max_epochs=3, min_val_windows=1000, **FAST)
    strict.fit(samples, seed=0)
    diag = strict.diagnostics()
    adaptation = diag[diag["stage"].eq("adaptation")]
    assert adaptation["select_reason"].eq("inner_validation_too_short").all()
    assert adaptation["fit_status"].eq("kept_shared").all()
    np.testing.assert_allclose(strict.predict(samples), strict.predict_shared(samples))
