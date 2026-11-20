"""Aggregate policy extractions into quarterly flow and stance indices.

Weight direction by confidence, relevance and housing specificity. Group duplicate
announcements by effective date and policy type. Decay the stance between quarters."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TYPE_WEIGHTS = {
    "policy_rate": 1.0, "selective_credit": 1.0, "bank_credit": 1.0,
    "housing_tax": 1.0, "housing_market": 1.0, "other": 0.0,
}


def quarterly_index(events: pd.DataFrame, start: str, end: str, half_life: float = 4.0,
                    type_weights: dict | None = None, min_confidence: float = 0.0,
                    require_evidence: bool = False, country: str = "Taiwan") -> pd.DataFrame:
    """events needs: date (publication), policy_type, direction, confidence;
    optional: relevant, evidence_found. Returns one row per quarter from start to end."""
    w = {**DEFAULT_TYPE_WEIGHTS, **(type_weights or {})}
    e = events.copy()
    e["date"] = pd.to_datetime(e["date"])
    keep = e["confidence"] >= min_confidence
    if "relevant" in e:
        keep &= e["relevant"].astype(bool)
    if require_evidence and "evidence_found" in e:
        keep &= e["evidence_found"].astype(bool)
    e = e[keep]
    e = e.assign(score=e["direction"] * e["confidence"] * e["policy_type"].map(w).fillna(0.0),
                 quarter=e["date"].dt.to_period("Q"))

    quarters = pd.period_range(start, end, freq="Q")
    flow = e.groupby("quarter")["score"].sum().reindex(quarters, fill_value=0.0)
    n = e.groupby("quarter").size().reindex(quarters, fill_value=0)
    decay = 0.5 ** (1.0 / half_life)
    stance = np.zeros(len(flow))
    for i, v in enumerate(flow.to_numpy()):
        stance[i] = v + (decay * stance[i - 1] if i else 0.0)
    return pd.DataFrame({"country": country, "quarter": quarters.astype(str),
                         "policy_flow": flow.to_numpy(), "policy_stance": stance,
                         "n_docs": n.to_numpy()})


def extractions_to_events(ex: pd.DataFrame) -> pd.DataFrame:
    ok = ex[ex["parse_ok"]].copy()
    return ok.rename(columns={"published_date": "date"})[
        ["doc_id", "date", "policy_type", "direction", "confidence", "relevant", "evidence_found",
         "housing_specific"]]


def timeline_to_events(tl: pd.DataFrame) -> pd.DataFrame:
    return tl.assign(confidence=1.0, relevant=True, evidence_found=True)[
        ["event_id", "date", "policy_type", "direction", "confidence", "relevant", "evidence_found",
         "housing_specific"]]
