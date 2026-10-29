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
    if p["gap_quarters"] < max_h:
        raise ValueError("gap_quarters must be >= the largest horizon")
    if cfg["bootstrap"]["block_length"] < 1:
        raise ValueError("block_length must be >= 1")


def lag_days(cfg: dict, variable: str, country: str) -> int:
    """Release lag of `variable` for `country` (falls back to the variable default)."""
    table = cfg["release_lag_days"][variable]
    return int(table.get(country, table["default"]))


def hpi_basis(cfg: dict, country: str) -> str:
    table = cfg["hpi_basis"]
    return table.get(country, table["default"])
