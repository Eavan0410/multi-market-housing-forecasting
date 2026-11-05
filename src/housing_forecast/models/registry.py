"""Every model name the runner accepts.

Torch models are registered only when torch is installed, so the baselines
still run in an environment without it.
"""

from housing_forecast.models.baselines import REGISTRY as _BASELINES

REGISTRY = dict(_BASELINES)

from housing_forecast.models.arima import ARIMAPerMarket  # noqa: E402  (statsmodels imported only inside B's code)

REGISTRY["arima"] = ARIMAPerMarket

try:
    from housing_forecast.models.transformer import (
        TransformerFineTune, TransformerForecaster, TransformerFTHead, TransformerFTLowLR, TransformerFTShort,
        TransformerMarket,
    )
except ImportError:          # torch not installed
    pass
else:
    REGISTRY.update({
        "transformer": TransformerForecaster,
        "transformer_ft": TransformerFineTune,
        "transformer_ft_head": TransformerFTHead,
        "transformer_ft_short": TransformerFTShort,
        "transformer_ft_lowlr": TransformerFTLowLR,
        "transformer_market": TransformerMarket,
    })
