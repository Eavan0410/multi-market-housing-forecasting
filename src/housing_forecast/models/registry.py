"""Every model name the runner accepts.

Torch models are registered only when torch is installed, so the baselines
still run in an environment without it.
"""

from housing_forecast.models.baselines import REGISTRY as _BASELINES

REGISTRY = dict(_BASELINES)

try:
    from housing_forecast.models.transformer import TransformerFineTune, TransformerForecaster
except ImportError:          # torch not installed
    pass
else:
    REGISTRY.update({
        "transformer": TransformerForecaster,
        "transformer_ft": TransformerFineTune,
    })
