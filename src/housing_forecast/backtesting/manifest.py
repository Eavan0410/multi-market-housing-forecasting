"""Save run settings beside predictions and reject incompatible comparisons.

Features, cutoffs and data fingerprints must match; models and training pools
may differ."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from housing_forecast.data.contract import ContractError

# Settings that must be identical for two prediction files to be compared.
COMPARABLE_KEYS = [
    "phase", "features", "target", "periods", "lookback", "ffill_limit",
    "drop_nan_windows", "hpi_basis", "release_lag_days", "panel_sha256", "cpi_sha256",
]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(preds_path: str | Path) -> Path:
    p = Path(preds_path)
    return p.with_name(p.stem + ".meta.json")


def make_manifest(cfg: dict, *, experiment: str, phase: str, model: str, base_model: str,
                  pool: str, stochastic: bool, seeds: list, panel_path: str, cpi_path: str) -> dict:
    return {
        "experiment": experiment,
        "phase": phase,
        "model": model,
        "base_model": base_model,
        "pool": pool,
        "stochastic": stochastic,
        "seeds": list(seeds),
        "features": list(cfg["features"]["use"]),
        "target": cfg["target"],
        "periods": cfg["periods"],
        "lookback": cfg["features"]["lookback"],
        "ffill_limit": cfg["features"]["ffill_limit"],
        "drop_nan_windows": cfg["features"]["drop_nan_windows"],
        "hpi_basis": cfg["hpi_basis"],
        "release_lag_days": cfg["release_lag_days"],
        "panel_file": str(panel_path),
        "panel_sha256": file_sha256(panel_path),
        "cpi_file": str(cpi_path),
        "cpi_sha256": file_sha256(cpi_path),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(preds_path: str | Path, manifest: dict) -> None:
    with open(manifest_path(preds_path), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def read_manifest(preds_path: str | Path) -> dict | None:
    p = manifest_path(preds_path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def check_comparable(manifests: dict[str, dict]) -> None:
    """manifests: {file name: manifest}. Raise if any COMPARABLE_KEYS differ."""
    items = list(manifests.items())
    if len(items) < 2:
        return
    ref_name, ref = items[0]
    for name, m in items[1:]:
        diff = [k for k in COMPARABLE_KEYS if m.get(k) != ref.get(k)]
        if diff:
            raise ContractError(
                f"{name} was produced with different settings than {ref_name}: {diff}. "
                "Score only files from the same experiment run."
            )
