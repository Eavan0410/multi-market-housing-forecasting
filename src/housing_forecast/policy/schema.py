"""Data records for the LLM policy index.

Three tables move through the pipeline:

    announcements  (C collects)      doc_id, source, published_date, title, url, text
    extractions    (LLM output)      one structured record per announcement
    manual timeline (C, gold standard)  event_id, date, policy_type, direction, housing_specific, description, source_url

Direction convention, used everywhere: + = tightening, - = easing.
    +2 strong tightening, +1 tightening, 0 neutral / unclear, -1 easing, -2 strong easing
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

POLICY_TYPES = (
    "policy_rate",        # Policy rates, including the discount rate
    "selective_credit",   # Loan-to-value limits, grace periods and second-home mortgages
    "bank_credit",        # Bank lending limits, window guidance and reserve requirements
    "housing_tax",        # Property transaction and holding taxes
    "housing_market",     # Land rules, presales, speculation controls and housing supply
    "other",
)

ANNOUNCEMENT_COLUMNS = ["doc_id", "source", "published_date", "title", "url", "text"]
TIMELINE_COLUMNS = ["event_id", "date", "policy_type", "direction", "housing_specific",
                    "description", "source_url"]

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "policy_type": {"type": "string", "enum": list(POLICY_TYPES)},
        "direction": {"type": "integer", "minimum": -2, "maximum": 2},
        "housing_specific": {"type": "boolean"},
        "effective_date": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": ["relevant", "policy_type", "direction", "housing_specific",
                 "effective_date", "confidence", "evidence"],
    "additionalProperties": False,
}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Announcement:
    doc_id: str
    source: str
    published_date: str          # YYYY-MM-DD, the day the text became public
    title: str
    url: str
    text: str

    @property
    def text_sha(self) -> str:
        return text_hash(self.title + "\n" + self.text)


@dataclass
class Extraction:
    doc_id: str
    published_date: str
    text_sha: str
    model: str
    prompt_version: str
    parse_ok: bool
    relevant: bool = False
    policy_type: str = "other"
    direction: int = 0
    housing_specific: bool = False
    effective_date: str | None = None
    confidence: float = 0.0
    evidence: str = ""
    evidence_found: bool = False   # evidence is a real substring of the text (hallucination check)
    raw_output: str = ""
    error: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ExtractionRun:
    """Timing and throughput for one extraction run."""
    model: str
    backend: str
    n_docs: int = 0
    n_parse_ok: int = 0
    n_retried: int = 0
    wall_seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        s = max(self.wall_seconds, 1e-9)
        return {**asdict(self),
                "docs_per_second": self.n_docs / s,
                "output_tokens_per_second": self.output_tokens / s,
                "parse_rate": self.n_parse_ok / max(self.n_docs, 1)}


def load_announcements(path: str | Path) -> list[Announcement]:
    """Read announcements from .jsonl (preferred: texts keep their line breaks) or .csv."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ANNOUNCEMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    df = df[ANNOUNCEMENT_COLUMNS].fillna("").astype(str)
    if df["doc_id"].duplicated().any():
        raise ValueError(f"duplicate doc_id in {path}: {df.loc[df['doc_id'].duplicated(), 'doc_id'].tolist()[:5]}")
    pd.to_datetime(df["published_date"], format="%Y-%m-%d")
    return [Announcement(**r) for r in df.to_dict("records")]


def load_timeline(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in TIMELINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    bad = set(df["policy_type"]) - set(POLICY_TYPES)
    if bad:
        raise ValueError(f"unknown policy_type in timeline: {sorted(bad)}")
    if not df["direction"].between(-2, 2).all():
        raise ValueError("timeline direction must be in -2..2")
    df["housing_specific"] = df["housing_specific"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def read_extractions(path: str | Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])
