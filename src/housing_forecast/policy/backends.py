"""Ways to run the model. All run open-source models locally; no external API.

    VLLMBackend    vLLM inside this process (GPU machine, e.g. the 3060 or the project GPU)
    HTTPBackend    a local `vllm serve` server through its OpenAI-compatible endpoint
                   (e.g. server on the GPU machine, client on the Mac)
    MockBackend    keyword rules, no model; for tests and dry runs of the pipeline

Every backend takes a list of chat-message lists and returns one Generation per input.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

from housing_forecast.policy.schema import EXTRACTION_SCHEMA


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0


class Backend:
    name = "base"

    def __init__(self, model: str):
        self.model = model

    def generate(self, batch: list[list[dict]]) -> list[Generation]:
        raise NotImplementedError


class VLLMBackend(Backend):
    """Batched offline inference. vLLM is imported only here, so the rest runs without it."""

    name = "vllm"

    def __init__(self, model: str, max_tokens: int = 256, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85, quantization: str | None = None,
                 guided: bool = True, seed: int = 0):
        super().__init__(model)
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, max_model_len=max_model_len, seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization, quantization=quantization)
        kw = dict(temperature=0.0, max_tokens=max_tokens)
        self.guided = False
        if guided:   # constrained JSON decoding; the API name differs across vLLM versions
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kw["guided_decoding"] = GuidedDecodingParams(json=EXTRACTION_SCHEMA)
                self.guided = True
            except ImportError:
                try:
                    from vllm.sampling_params import StructuredOutputsParams
                    kw["structured_outputs"] = StructuredOutputsParams(json=EXTRACTION_SCHEMA)
                    self.guided = True
                except ImportError:
                    pass       # fall back to plain decoding + the parser in extract.py
        self.params = SamplingParams(**kw)

    def generate(self, batch):
        outs = self.llm.chat(batch, self.params, use_tqdm=False)
        return [Generation(o.outputs[0].text, len(o.prompt_token_ids or []), len(o.outputs[0].token_ids))
                for o in outs]


class HTTPBackend(Backend):
    """Client for a local OpenAI-compatible server, e.g. `vllm serve <model> --port 8000`."""

    name = "http"

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", max_tokens: int = 256,
                 timeout: float = 120.0, guided: bool = True):
        super().__init__(model)
        self.base_url, self.max_tokens, self.timeout, self.guided = base_url.rstrip("/"), max_tokens, timeout, guided

    def _one(self, messages: list[dict]) -> Generation:
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": self.max_tokens}
        if self.guided:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "policy_extraction", "schema": EXTRACTION_SCHEMA}}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        usage = out.get("usage", {})
        return Generation(out["choices"][0]["message"]["content"],
                          usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

    def generate(self, batch):
        # Sequential requests are fine for hundreds of documents; the server batches internally
        # when several clients send at once. Swap in a thread pool if this becomes the bottleneck.
        return [self._one(m) for m in batch]


class MockBackend(Backend):
    """Keyword rules standing in for a model. Only for testing the pipeline, never for results."""

    name = "mock"
    TIGHTEN = ("升息", "調升", "调升", "調降貸款成數", "调降贷款成数", "收紧", "收緊", "加強管制", "加强管制", "囤房稅", "囤房税")
    EASE = ("降息", "調降利率", "调降利率", "放寬", "放宽", "解除")

    def __init__(self, model: str = "mock-keywords", delay: float = 0.0):
        super().__init__(model)
        self.delay = delay

    def generate(self, batch):
        out = []
        for messages in batch:
            text = messages[-1]["content"]
            up = next((k for k in self.TIGHTEN if k in text), None)
            down = next((k for k in self.EASE if k in text), None)
            hit = up or down
            if not hit:
                ptype = "other"
            elif re.search("息|利率", hit):
                ptype = "policy_rate"
            elif re.search("稅|税", hit):
                ptype = "housing_tax"
            else:
                ptype = "selective_credit"
            rec = {"relevant": bool(hit), "policy_type": ptype,
                   "direction": 1 if up else (-1 if down else 0),
                   "housing_specific": ptype != "policy_rate" and bool(hit),
                   "effective_date": None, "confidence": 0.6 if hit else 0.0,
                   "evidence": hit or ""}
            time.sleep(self.delay)
            out.append(Generation(json.dumps(rec, ensure_ascii=False), len(text), 40))
        return out


def make_backend(kind: str, model: str, **kw) -> Backend:
    if kind == "vllm":
        return VLLMBackend(model, **kw)
    if kind == "http":
        return HTTPBackend(model, **kw)
    if kind == "mock":
        return MockBackend(model)
    raise ValueError(f"unknown backend {kind!r}; use vllm, http or mock")
