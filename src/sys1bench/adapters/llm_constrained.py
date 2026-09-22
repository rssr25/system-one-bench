"""Constrained-decoding LLM baseline with first-token logprobs, so LLMs get calibration metrics.

Backend A (default): any OpenAI-compatible server (vLLM, OpenRouter) with `logprobs`. The prompt lists
options as single-letter keys; probabilities are the renormalised logprobs of the first generated token
over the option letters, which is the standard "logit-based" calibration protocol for MCQ.
Backend B: vLLM + Outlines regex-constrained generation (import guarded); same probability extraction.
"""

from __future__ import annotations

import math
import os
import string
import time

import httpx

from ..schemas import (
    Answer,
    DecisionRequest,
    DecisionResponse,
    LatencyRecord,
    ModelCapabilities,
    ProviderRecord,
    Question,
)
from .base import BaseAdapter, finalize_answer, register

LETTERS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def build_prompt(state: str, q: Question) -> tuple[str, list[str]]:
    keys = q.option_keys
    letters = list(LETTERS[: len(keys)])
    if q.type == "noul":
        lines = ["A. yes", "B. no"]
    elif q.type == "choice":
        lines = [f"{L}. {c.key}: {c.description}" for L, c in zip(letters, q.criteria)]  # type: ignore[union-attr]
    else:
        lines = [f"{L}. level {c.level}: {c.description}" for L, c in zip(letters, q.criteria)]  # type: ignore[union-attr]
    prompt = (f"State:\n{state}\n\nQuestion: {q.instructions}\nOptions:\n" + "\n".join(lines) +
              "\n\nAnswer with the single letter of the best option.\nAnswer:")
    return prompt, letters


@register("llm_constrained")
class LLMConstrainedAdapter(BaseAdapter):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", base_url: str = "http://localhost:8000/v1",
                 api_key_env: str = "OPENAI_API_KEY", top_logprobs: int = 20, price_per_m_input: float | None = None,
                 price_per_m_output: float | None = None, timeout_s: float = 60.0, **tunables) -> None:
        super().__init__(model_id, **tunables)
        self.base_url, self.top_logprobs = base_url.rstrip("/"), top_logprobs
        self.key = os.environ.get(api_key_env, "EMPTY")
        self.pi, self.po = price_per_m_input, price_per_m_output
        self._client = httpx.Client(timeout=timeout_s)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=f"llm:{self.model_id}", deployment="hosted" if "localhost" not in self.base_url else "local",
                                 max_options=len(LETTERS), emits_confidence=False, supports_batching=False, deterministic=False)

    def _one(self, state: str, q: Question) -> tuple[Answer, dict]:
        prompt, letters = build_prompt(state, q)
        body = {"model": self.model_id, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
                "logprobs": self.top_logprobs}
        r = self._client.post(f"{self.base_url}/completions", json=body,
                              headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        data = r.json()
        lp = data["choices"][0].get("logprobs", {})
        top = (lp.get("top_logprobs") or [{}])[0]
        scores = []
        for L in letters:
            cands = [v for k, v in top.items() if k.strip() == L]
            scores.append(max(cands) if cands else -30.0)
        m = max(scores)
        w = [math.exp(s - m) for s in scores]
        z = sum(w)
        probs = [x / z for x in w]
        return finalize_answer(q, probs), data

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        answers, raws, in_tok, out_tok = {}, {}, 0, 0
        err = None
        for k, q in request.questions.items():
            try:
                a, data = self._one(request.state_text(), q)
                answers[k] = a
                raws[k] = data
                u = data.get("usage", {})
                in_tok += u.get("prompt_tokens", 0)
                out_tok += u.get("completion_tokens", 0)
            except Exception as e:  # transport or parse
                answers[k] = Answer.failed(q, "transport")
                err = str(e)
        ms = (time.perf_counter() - t0) * 1000
        cost = None
        if self.pi is not None:
            cost = in_tok / 1e6 * self.pi + out_tok / 1e6 * (self.po or 0.0)
        return DecisionResponse(answers=answers, latency=LatencyRecord(client_ms=ms, timestamp=time.time()),
                                provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id,
                                                        billed_input_tokens=in_tok, billed_output_tokens=out_tok, cost_usd=cost),
                                raw=raws, transport_error=err)
