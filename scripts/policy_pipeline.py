"""LLM policy index pipeline (nice to have, exploratory).

    python scripts/policy_pipeline.py extract  [--backend mock|vllm|http] [--limit N]
    python scripts/policy_pipeline.py index
    python scripts/policy_pipeline.py eval
    python scripts/policy_pipeline.py spotcheck-export
    python scripts/policy_pipeline.py spotcheck-score results/policy/spotcheck_filled.csv

Settings live in configs/policy.toml. Outputs go to results/policy/ and the
quarterly index to data/interim/taiwan_policy_index.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from housing_forecast.policy.backends import make_backend
from housing_forecast.policy.evaluate import event_match, quarter_agreement, score_spotcheck, spotcheck_sample
from housing_forecast.policy.extract import cache_path, run_extraction
from housing_forecast.policy.index import extractions_to_events, quarterly_index, timeline_to_events
from housing_forecast.policy.schema import load_announcements, load_timeline, read_extractions


def load_cfg(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def model_name(cfg: dict, backend: str) -> str:
    return "mock-keywords" if backend == "mock" else cfg["model"]["name"]


def cmd_extract(cfg, args):
    docs = load_announcements(args.input or cfg["paths"]["announcements"])
    if args.limit:
        docs = docs[:args.limit]
    m = cfg["model"]
    kind = args.backend or m["backend"]
    kw = {}
    if kind == "vllm":
        kw = dict(max_tokens=m["max_tokens"], max_model_len=m["max_model_len"],
                  gpu_memory_utilization=m["gpu_memory_utilization"], quantization=m["quantization"] or None)
    elif kind == "http":
        kw = dict(base_url=m["http_base_url"], max_tokens=m["max_tokens"])
    t0 = time.perf_counter()
    backend = make_backend(kind, model_name(cfg, kind), **kw)
    load_s = time.perf_counter() - t0
    path, run = run_extraction(docs, backend, cfg["paths"]["out_dir"],
                               batch_size=cfg["extraction"]["batch_size"], max_chars=cfg["extraction"]["max_chars"])
    summary = {**run.summary(), "model_load_seconds": load_s}
    if kind == "vllm":
        summary["guided_decoding"] = backend.guided
    out = Path(cfg["paths"]["out_dir"]) / f"throughput__{path.stem}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _extractions(cfg, args) -> pd.DataFrame:
    kind = args.backend or cfg["model"]["backend"]
    return read_extractions(cache_path(cfg["paths"]["out_dir"], model_name(cfg, kind)))


def _index(cfg, events):
    i = cfg["index"]
    return quarterly_index(events, i["start"], i["end"], half_life=i["half_life"],
                           min_confidence=i["min_confidence"], require_evidence=i["require_evidence"])


def cmd_index(cfg, args):
    ex = _extractions(cfg, args)
    idx = _index(cfg, extractions_to_events(ex))
    out = Path(cfg["paths"]["index_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(out, index=False)
    print(f"{len(ex)} extractions ({int(ex['parse_ok'].sum())} parsed, {int(ex['relevant'].sum())} relevant) "
          f"-> {len(idx)} quarters -> {out}")


def cmd_eval(cfg, args):
    ex = _extractions(cfg, args)
    llm_events = extractions_to_events(ex)
    man_events = timeline_to_events(load_timeline(args.timeline or cfg["paths"]["timeline"]))
    llm_idx, man_idx = _index(cfg, llm_events), _index(cfg, man_events)
    report = {
        "parse_rate": float(ex["parse_ok"].mean()),
        "evidence_found_rate": float(ex.loc[ex["relevant"], "evidence_found"].mean()) if ex["relevant"].any() else None,
        "quarter_flow": quarter_agreement(llm_idx, man_idx, "policy_flow"),
        "quarter_stance": quarter_agreement(llm_idx, man_idx, "policy_stance"),
        "events": event_match(llm_events, man_events, cfg["eval"]["window_days"]),
    }
    out = Path(cfg["paths"]["out_dir"]) / "eval_vs_manual_timeline.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_spot_export(cfg, args):
    ex = _extractions(cfg, args)
    docs = pd.DataFrame([d.__dict__ for d in load_announcements(cfg["paths"]["announcements"])])
    sheet = spotcheck_sample(ex, docs, n=cfg["eval"]["spotcheck_n"])
    out = Path(cfg["paths"]["out_dir"]) / "spotcheck_to_label.csv"
    sheet.to_csv(out, index=False, encoding="utf-8-sig")       # utf-8-sig so Excel shows Chinese correctly
    print(f"{len(sheet)} rows -> {out}; fill the human_* columns, save as a new file, then run spotcheck-score")


def cmd_spot_score(cfg, args):
    report = score_spotcheck(pd.read_csv(args.sheet))
    out = Path(cfg["paths"]["out_dir"]) / "spotcheck_scores.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/policy.toml")
    ap.add_argument("--backend", choices=["vllm", "http", "mock"], default=None, help="override [model].backend")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--input", default=None)
    e.add_argument("--limit", type=int, default=None, help="only the first N documents (smoke test)")
    sub.add_parser("index")
    ev = sub.add_parser("eval")
    ev.add_argument("--timeline", default=None)
    sub.add_parser("spotcheck-export")
    sc = sub.add_parser("spotcheck-score")
    sc.add_argument("sheet")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    {"extract": cmd_extract, "index": cmd_index, "eval": cmd_eval,
     "spotcheck-export": cmd_spot_export, "spotcheck-score": cmd_spot_score}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
