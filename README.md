# LLM Inference for Policy Signals: vLLM Batch Extraction on a Single L4

This branch holds the LLM inference side of a house price forecasting project. A local 14B model reads Chinese policy announcements and turns them into a structured, point-in-time policy signal. The work here is about running that model efficiently on one 24 GB GPU and checking that its output can be trusted.

The forecasting study itself (one shared Transformer for 48 housing markets) lives on [`main`](https://github.com/Eavan0410/multi-market-housing-forecasting/tree/main).

## Key results

All numbers come from one machine: **1× NVIDIA L4 (24 GB), vLLM 0.34.0, Qwen2.5-14B-Instruct GPTQ-Int4**, 30 announcements per pass, median of 3 timed runs.

| Change | Result |
|---|---|
| Batching + prefix caching + larger KV cache budget, vs one request at a time | **2.85× throughput** (124.5 s → 43.7 s per pass) |
| Batch 32 vs batch 1 | 2.69× throughput |
| Prefix caching on vs off (batch 32) | 1.17× throughput |
| JSON Schema guided decoding on vs off | parse rate **93.3% → 100%**, no measurable throughput cost |
| `gpu_memory_utilization` 0.85 → 0.95 | KV cache +36% (36,864 → 50,176 tokens), throughput only +5.6% |

**Main finding:** the workload is prefill-bound. Each pass sends about 65k prompt tokens and gets back about 2.5k output tokens, a ratio of roughly 26:1. A single 2k-token prompt already keeps the GPU busy during prefill, so batching mostly speeds up the short decode phase, and its gain flattens after batch 8. The KV cache experiment confirms this: 36% more KV cache gave only 6% more throughput, so memory is not the main limit. Skipping repeated prefill (prefix caching) helps more than adding memory.

Full table: [`results/bench/summary.md`](results/bench/summary.md). Raw per-run timings: `results/bench/*.jsonl`. Engine settings and KV cache size per run: `results/bench/*.meta.json`.

## Pipeline

```
announcements.jsonl ──► prompt (system + 1 example + document)
                          │
                          ▼
              vLLM offline engine (batched)  ◄── JSON Schema guided decoding (xgrammar)
                          │
                          ▼
     parse + validate ──► evidence check ──► point-in-time quarterly index
```

- **Inference backends** (`src/housing_forecast/policy/backends.py`): a vLLM offline engine and an OpenAI-compatible HTTP client share one interface, so the same pipeline runs batched offline or against `vllm serve`. A keyword mock backend runs the pipeline in tests without a GPU.
- **Structured output**: every response is constrained to a JSON Schema (policy type, direction from −2 to +2, effective date, confidence, evidence). A hand-written validator rejects anything outside the schema.
- **Evidence check**: the model must quote one sentence from the announcement. The quote is matched against the original text after removing punctuation, and records whose evidence is not found can be dropped.
- **Point-in-time index**: records are dated by publication date, not by effective date, so the quarterly signal only uses information that was public at the time. This matters when the signal feeds a backtest.
- **Caching**: extractions are cached by document hash, model and prompt version, so a rerun only processes new or changed documents.

## Signal quality

Checked against a hand-labelled timeline of Taiwan housing and credit policy:

| Check | Value |
|---|---|
| Relevance accuracy (30 labelled documents) | 93% |
| Policy type accuracy | 93% |
| Direction sign accuracy | 68% |
| Quarterly policy stance vs manual series (Pearson) | 0.98 |
| Event recall / precision (14-day window) | 79% / 77% |

Direction is the weakest part. The model often agrees on the type of measure but not on how strong it is.

## Benchmark method

`scripts/bench/run_bench.py` and `scripts/bench/run_all.sh`.

- One process per engine setting, because vLLM does not reliably free GPU memory when a second engine is created in the same process.
- One warm-up pass per setting (CUDA graphs, grammar compilation), not timed.
- **The prefix cache is reset before every timed run**, so a run never reuses KV blocks from the previous run of the same documents. Without this, repeated runs would look much faster than real use.
- Model load time is recorded separately and not counted in throughput.
- Guided-off runs do not retry failed parses, so the parse rate shows the model's raw behaviour.

Reproduce (about 30 minutes on one L4):

```bash
pip install vllm && pip install -e .
hf download Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 --local-dir <model_dir>
bash scripts/bench/run_all.sh <model_dir>
python scripts/bench/run_bench.py --model <model_dir> --tag gpu095 --gpu-mem 0.95 --batch-sizes 32 --guided on
python scripts/bench/run_bench.py --summarize
```

## Memory on a 24 GB card

From the vLLM startup log at `gpu_memory_utilization=0.85`: weights and non-torch memory take about 9.6 GiB, peak activations about 0.9 GiB and CUDA graphs about 0.7 GiB, which leaves about 36.9k tokens of KV cache. Changing `max_model_len` from 16,384 to 4,096 did not change the KV cache size; it only changed vLLM's estimate of maximum concurrency (2.4× → 9×). The KV cache size is set by the memory budget, not by the context limit.

## Limitations

- The workload is small: 30 announcements per pass, about 10 to 120 seconds per run. Results show relative gains, not production throughput.
- One GPU and one model. Absolute numbers would be very different on a higher-bandwidth card such as an RTX 4090 or H100.
- The labelled set has only 30 documents, so accuracy numbers have wide uncertainty.
- The policy index has not yet been tested as a forecasting feature.

## Next steps

- A larger workload (several hundred mainland and Taiwan policy documents) for more stable throughput numbers.
- Knowledge distillation: use the 14B model to label the larger set, fine-tune a 3B model with LoRA, and compare throughput and accuracy on the hand-labelled set.
- Compare quantization formats (GPTQ, AWQ, FP8) on throughput, memory and extraction accuracy.
