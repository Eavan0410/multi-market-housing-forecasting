"""Run the model over announcements, validate its JSON, and cache the results.

Each announcement gets exactly one Extraction line in a JSONL cache file. The
cache key is (doc_id, text hash, model, prompt version), so re-running only
processes new or changed documents, and changing the model or prompt starts a
fresh file instead of mixing outputs.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from housing_forecast.policy.backends import Backend
from housing_forecast.policy.prompts import PROMPT_VERSION, build_messages
from housing_forecast.policy.schema import (
    EXTRACTION_SCHEMA, POLICY_TYPES, Announcement, Extraction, ExtractionRun,
)

_PUNCT = re.compile(r"[\s，。、；：！？「」『』（）()\"'“”‘’,.;:!?]")


def cache_path(out_dir: str | Path, model: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    return Path(out_dir) / f"extractions__{slug}__{PROMPT_VERSION}.jsonl"


def _first_json_object(text: str) -> dict:
    """Parse the model output; tolerate code fences or text around the JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def validate(rec: dict) -> dict:
    """Check the record against EXTRACTION_SCHEMA by hand (no extra dependency)."""
    missing = [k for k in EXTRACTION_SCHEMA["required"] if k not in rec]
    if missing:
        raise ValueError(f"missing fields {missing}")
    if not isinstance(rec["relevant"], bool) or not isinstance(rec["housing_specific"], bool):
        raise ValueError("relevant / housing_specific must be true or false")
    if rec["policy_type"] not in POLICY_TYPES:
        raise ValueError(f"policy_type {rec['policy_type']!r} not allowed")
    d = rec["direction"]
    if isinstance(d, bool) or not isinstance(d, int) or not -2 <= d <= 2:
        raise ValueError(f"direction {d!r} must be an integer in -2..2")
    c = rec["confidence"]
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 <= c <= 1:
        raise ValueError(f"confidence {c!r} must be in [0, 1]")
    if rec["effective_date"] is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(rec["effective_date"])):
        rec["effective_date"] = None                       # keep the record, drop a malformed date
    if not rec["relevant"]:
        rec["direction"] = 0                               # irrelevant documents never move the index
    return rec


def evidence_in_text(evidence: str, doc: Announcement) -> bool:
    if not evidence:
        return False
    norm = lambda s: _PUNCT.sub("", s)
    return norm(evidence) in norm(doc.title + doc.text)


def _to_extraction(doc: Announcement, model: str, gen) -> Extraction:
    base = dict(doc_id=doc.doc_id, published_date=doc.published_date, text_sha=doc.text_sha,
                model=model, prompt_version=PROMPT_VERSION, raw_output=gen.text,
                prompt_tokens=gen.prompt_tokens, output_tokens=gen.output_tokens)
    try:
        rec = validate(_first_json_object(gen.text))
    except (ValueError, json.JSONDecodeError) as e:
        return Extraction(parse_ok=False, error=str(e)[:200], **base)
    return Extraction(parse_ok=True, relevant=rec["relevant"], policy_type=rec["policy_type"],
                      direction=int(rec["direction"]), housing_specific=rec["housing_specific"],
                      effective_date=rec["effective_date"], confidence=float(rec["confidence"]),
                      evidence=str(rec["evidence"])[:200],
                      evidence_found=evidence_in_text(str(rec["evidence"]), doc), **base)


def _load_cache(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["doc_id"], r["text_sha"]): r for r in rows}


def run_extraction(docs: list[Announcement], backend: Backend, out_dir: str | Path,
                   batch_size: int = 32, max_chars: int = 4000, retry_failed: bool = True,
                   log=print) -> tuple[Path, ExtractionRun]:
    """Extract every document not yet in the cache; append results to the cache file."""
    path = cache_path(out_dir, backend.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_cache(path)
    todo = [d for d in docs if (d.doc_id, d.text_sha) not in cached]
    run = ExtractionRun(model=backend.model, backend=backend.name)
    log(f"{len(docs)} documents, {len(docs) - len(todo)} cached, {len(todo)} to run -> {path}")

    t0 = time.perf_counter()
    with open(path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            gens = backend.generate([build_messages(d, max_chars) for d in chunk])
            results = [_to_extraction(d, backend.model, g) for d, g in zip(chunk, gens)]
            if retry_failed:
                bad = [j for j, r in enumerate(results) if not r.parse_ok]
                if bad:
                    regen = backend.generate([build_messages(chunk[j], max_chars) for j in bad])
                    for j, g in zip(bad, regen):
                        results[j] = _to_extraction(chunk[j], backend.model, g)
                    run.n_retried += len(bad)
            for r in results:
                f.write(r.to_json() + "\n")
                run.n_docs += 1
                run.n_parse_ok += r.parse_ok
                run.prompt_tokens += r.prompt_tokens
                run.output_tokens += r.output_tokens
            log(f"  {min(i + batch_size, len(todo))}/{len(todo)}")
    run.wall_seconds = time.perf_counter() - t0
    return path, run
