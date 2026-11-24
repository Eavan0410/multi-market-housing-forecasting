"""Throughput benchmark for the policy extraction pipeline (vLLM offline).

One engine setting per process; batch size and guided decoding are swept inside it.
The prefix cache is reset before every timed run.

    python scripts/bench/run_bench.py --model <path> --tag base
    python scripts/bench/run_bench.py --summarize
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("results/bench")


def make_params(guided: bool, max_tokens: int):
    from vllm import SamplingParams

    from housing_forecast.policy.schema import EXTRACTION_SCHEMA

    kw = dict(temperature=0.0, max_tokens=max_tokens)
    if guided:  # API name differs across vLLM versions
        try:
            from vllm.sampling_params import GuidedDecodingParams
            kw["guided_decoding"] = GuidedDecodingParams(json=EXTRACTION_SCHEMA)
        except ImportError:
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs"] = StructuredOutputsParams(json=EXTRACTION_SCHEMA)
    return SamplingParams(**kw)


def parse_ok(text: str) -> bool:
    from housing_forecast.policy.extract import _first_json_object, validate
    try:
        validate(_first_json_object(text))
        return True
    except Exception:
        return False


def reset_prefix_cache(llm) -> bool:
    try:
        llm.reset_prefix_cache()
        return True
    except Exception:
        return False


def kv_info(llm) -> dict:
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        out = {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size}
        if cc.num_gpu_blocks:
            out["kv_cache_tokens"] = cc.num_gpu_blocks * cc.block_size
        return out
    except Exception:
        return {}


def env_info() -> dict:
    info = {}
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    return info


def run_once(llm, prompts, params, batch_size: int) -> dict:
    t0 = time.perf_counter()
    outs = []
    for i in range(0, len(prompts), batch_size):
        outs += llm.chat(prompts[i:i + batch_size], params, use_tqdm=False)
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "n": len(outs),
        "prompt_tokens": sum(len(o.prompt_token_ids or []) for o in outs),
        "output_tokens": sum(len(o.outputs[0].token_ids) for o in outs),
        "parse_ok": sum(parse_ok(o.outputs[0].text) for o in outs),
    }


def bench(args):
    from vllm import LLM

    from housing_forecast.policy.prompts import build_messages
    from housing_forecast.policy.schema import load_announcements

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_announcements(args.announcements)
    prompts = [build_messages(d, max_chars=args.max_chars) for d in docs]

    t0 = time.perf_counter()
    llm = LLM(model=args.model, max_model_len=args.max_model_len, seed=0,
              gpu_memory_utilization=args.gpu_mem,
              enable_prefix_caching=(args.prefix_caching == "on"))
    load_s = time.perf_counter() - t0

    meta = {"tag": args.tag, "model": args.model, "prefix_caching": args.prefix_caching,
            "max_model_len": args.max_model_len, "gpu_memory_utilization": args.gpu_mem,
            "max_chars": args.max_chars, "n_docs": len(docs), "model_load_s": round(load_s, 1),
            **env_info(), **kv_info(llm)}
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    (OUT_DIR / f"{args.tag}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if args.kv_only:
        return

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    guided_opts = [g.strip() == "on" for g in args.guided.split(",")]
    out_path = OUT_DIR / f"{args.tag}.jsonl"
    out_path.write_text("")

    for guided in guided_opts:
        params = make_params(guided, args.max_tokens)
        reset_prefix_cache(llm)
        run_once(llm, prompts, params, max(batch_sizes))  # warm-up, not recorded
        for bs in batch_sizes:
            for r in range(args.repeats):
                cache_reset = reset_prefix_cache(llm)
                res = run_once(llm, prompts, params, bs)
                row = {**{k: meta[k] for k in ("tag", "prefix_caching", "max_model_len")},
                       "guided": guided, "batch_size": bs, "repeat": r,
                       "cache_reset": cache_reset, **res}
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"guided={guided!s:5} bs={bs:<3} rep={r}  {res['wall_s']:.2f}s  "
                      f"parse {res['parse_ok']}/{res['n']}")


def summarize():
    rows = []
    for p in sorted(OUT_DIR.glob("*.jsonl")):
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        print("no results in results/bench/")
        return

    groups = defaultdict(list)
    for r in rows:
        groups[(r["tag"], r["prefix_caching"], r["guided"], r["batch_size"])].append(r)

    table = {}
    lines = ["| tag | prefix cache | guided | batch | median wall (s) | docs/s | output tok/s | "
             "total tok/s | parse rate | repeats |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(groups, key=lambda k: (k[0], k[2] is False, k[3])):
        g = groups[key]
        wall = statistics.median(x["wall_s"] for x in g)
        n, p, o = g[0]["n"], g[0]["prompt_tokens"], g[0]["output_tokens"]
        parse = sum(x["parse_ok"] for x in g) / sum(x["n"] for x in g)
        table[key] = {"wall": wall, "docs_s": n / wall}
        lines.append(f"| {key[0]} | {key[1]} | {'on' if key[2] else 'off'} | {key[3]} | {wall:.2f} | "
                     f"{n / wall:.2f} | {o / wall:.0f} | {(p + o) / wall:.0f} | {parse:.1%} | {len(g)} |")

    def speed(tag, guided, bs):
        for (t, _, gd, b), v in table.items():
            if t == tag and gd == guided and b == bs:
                return v["docs_s"]
        return None

    notes = []
    base_bs = sorted({k[3] for k in table if k[0] == "base"})
    if base_bs:
        lo, hi = base_bs[0], base_bs[-1]
        a, b = speed("base", True, lo), speed("base", True, hi)
        if a and b:
            notes.append(f"- Batching: batch {hi} vs batch {lo} (guided on) -> {b / a:.2f}x throughput")
        on, off = speed("base", True, hi), speed("base", False, hi)
        if on and off:
            notes.append(f"- Guided decoding cost at batch {hi}: throughput {on / off - 1:+.1%} vs guided off")
        np_ = speed("no_prefix", True, hi)
        if on and np_:
            notes.append(f"- Prefix caching: on vs off at batch {hi} -> {on / np_:.2f}x throughput")
    for m in sorted(OUT_DIR.glob("*.meta.json")):
        meta = json.loads(m.read_text())
        if "kv_cache_tokens" in meta:
            notes.append(f"- KV cache [{meta['tag']}, max_model_len={meta['max_model_len']}]: "
                         f"{meta['kv_cache_tokens']:,} tokens")

    text = "\n".join(lines + [""] + notes)
    print(text)
    (OUT_DIR / "summary.md").write_text(text + "\n")
    print(f"\nsaved {OUT_DIR / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--prefix-caching", choices=["on", "off"], default="on")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-chars", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-sizes", default="1,4,8,32")
    ap.add_argument("--guided", default="on,off")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--announcements", default="data/raw/policy/announcements.jsonl")
    ap.add_argument("--kv-only", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        if not args.model:
            ap.error("--model is required")
        bench(args)


if __name__ == "__main__":
    main()
