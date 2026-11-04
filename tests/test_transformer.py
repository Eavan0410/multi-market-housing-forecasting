"""Quick checks for the Transformer forecasters (skipped when torch is not installed)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from housing_forecast.backtesting.infoset import SampleSet  # noqa: E402
from housing_forecast.models.transformer import (  # noqa: E402
    TransformerFineTune, TransformerForecaster, TransformerFTHead, TransformerFTLowLR, TransformerFTShort,
    TransformerMarket,
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
