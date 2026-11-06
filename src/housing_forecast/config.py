"""Load and check the experiment contract (configs/experiment.toml)."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import pandas as pd

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "experiment.toml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    check_config(cfg)
    return cfg


def check_config(cfg: dict) -> None:
    p = cfg["periods"]
    max_h = max(cfg["target"]["horizons"])
    v0, v1 = pd.Period(p["validation_start"], "Q"), pd.Period(p["validation_end"], "Q")
    t0 = pd.Period(p["test_start"], "Q")
    if not v0 <= v1 < t0:
        raise ValueError("periods must satisfy validation_start <= validation_end < test_start")
    # Validation labels must not reach into the test period.
    if v1 + max_h > t0:
        raise ValueError(
            f"validation_end + {max_h} quarters ({v1 + max_h}) passes test_start ({t0}); "
            "validation labels would use test-period prices"
        )
    # A shorter gap keeps more short-horizon samples; unavailable longer-horizon labels
    # are masked separately and checked for overlap in splits.py.
    if p["gap_quarters"] < max_h and not p.get("allow_short_gap", False):
        raise ValueError("gap_quarters must be >= the largest horizon (or set periods.allow_short_gap = true)")
    common = forecast_lag_days(cfg)
    if common is not None:
        slowest = max(int(x) for x in cfg["release_lag_days"]["hpi"].values())
        if common < slowest:
            raise ValueError(f"forecast_lag_days ({common}) must be >= the slowest HPI release lag ({slowest}), "
                             "otherwise some markets' HPI for quarter t is not public at the forecast date")
    if cfg["bootstrap"]["block_length"] < 1:
        raise ValueError("block_length must be >= 1")


def forecast_lag_days(cfg: dict) -> int | None:
    """Common forecast lag, or None to use the existing market-specific dates."""
    v = cfg["periods"].get("forecast_lag_days")
    if v is None:
        return None
    if v == "max_hpi_lag":
        return max(int(x) for x in cfg["release_lag_days"]["hpi"].values())
    if type(v) is not int:
        raise ValueError("periods.forecast_lag_days must be an integer or 'max_hpi_lag'")
    return v


def lag_days(cfg: dict, variable: str, country: str) -> int:
    """Release lag of `variable` for `country` (falls back to the variable default)."""
    table = cfg["release_lag_days"][variable]
    return int(table.get(country, table["default"]))


def hpi_basis(cfg: dict, country: str) -> str:
    table = cfg["hpi_basis"]
    return table.get(country, table["default"])
