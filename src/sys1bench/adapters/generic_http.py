"""Config-driven adapter for future hosted System One models.

A new vendor is added with a YAML file, not code:

    adapter: generic_http
    model_id: vendor/model-1.0
    url: https://api.vendor.com/v1/decide
    auth_env: VENDOR_API_KEY
    type_map: {choice: choice, score: score, noul: boolean}
    body_template: {"model": "{model_id}", "input": "{state}", "questions": "{questions}"}
    answers_path: ["result", "answers"]
    fields: {probs: ["probabilities", "distribution"], p_true: ["probability"], confidence: ["confidence"],
             model_returned: ["model"], generation_id: ["id"], input_tokens: ["usage", "input_tokens"]}
    price_per_m_input: 0.05

Anything the template cannot express is a real adapter; this covers the common JSON shape.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

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


def _dig(d: Any, path: list[str] | str | None, default=None):
    if path is None:
        return default
    if isinstance(path, str):
        path = [path]
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _first_path(d: Any, paths: list[Any] | None, default=None):
    for p in paths or []:
        v = _dig(d, p)
        if v is not None:
            return v
    return default


@register("generic_http")
class GenericHTTPAdapter(BaseAdapter):
    def __init__(self, model_id: str, url: str, auth_env: str | None = None, headers: dict | None = None,
                 type_map: dict | None = None, body_template: dict | None = None, answers_path: list | None = None,
                 fields: dict | None = None, price_per_m_input: float | None = None, deployment: str = "hosted",
                 max_options: int | None = None, max_state_tokens: int | None = None, timeout_s: float = 30.0,
                 max_retries: int = 5, **tunables) -> None:
        super().__init__(model_id, **tunables)
        self.url, self.timeout_s, self.max_retries = url, timeout_s, max_retries
        self.headers = dict(headers or {})
        if auth_env and os.environ.get(auth_env):
            self.headers.setdefault("Authorization", f"Bearer {os.environ[auth_env]}")
        self.type_map = type_map or {"choice": "choice", "score": "score", "noul": "boolean"}
        self.body_template = body_template or {"model": "{model_id}", "state": "{state}", "questions": "{questions}"}
        self.answers_path = answers_path or ["answers"]
        self.fields = fields or {}
        self.price = price_per_m_input
        self._caps = ModelCapabilities(name=model_id, deployment=deployment, max_options=max_options,  # type: ignore[arg-type]
                                       max_state_tokens=max_state_tokens, supports_batching=True)
        self._client = httpx.Client(timeout=timeout_s)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def _vendor_q(self, q: Question) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type_map[q.type], "instructions": q.instructions}
        if q.type == "choice":
            out["criteria"] = [{"key": c.key, "description": c.description} for c in q.criteria]  # type: ignore[union-attr]
        elif q.type == "score":
            out["criteria"] = [{"level": c.level, "description": c.description} for c in q.criteria]  # type: ignore[union-attr]
        return out

    def _render(self, request: DecisionRequest) -> dict[str, Any]:
        subs = {"{model_id}": self.model_id, "{state}": request.state,
                "{questions}": {k: self._vendor_q(q) for k, q in request.questions.items()}}

        def walk(x):
            if isinstance(x, str) and x in subs:
                return subs[x]
            if isinstance(x, str):
                return x.replace("{model_id}", self.model_id)
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            if isinstance(x, list):
                return [walk(v) for v in x]
            return x

        return walk(json.loads(json.dumps(self.body_template)))

    def _parse(self, q: Question, payload: dict[str, Any]) -> Answer:
        keys = q.option_keys
        conf = _first_path(payload, self.fields.get("confidence", [["confidence"]]))
        if q.type == "noul":
            p = _first_path(payload, self.fields.get("p_true", [["probability"], ["p_true"]]))
            return Answer.failed(q, "no probability") if p is None else finalize_answer(q, [float(p), 1 - float(p)], confidence=conf)
        dist = _first_path(payload, self.fields.get("probs", [["probabilities"], ["distribution"]]))
        if isinstance(dist, dict):
            probs = [float(dist.get(k, 0.0)) for k in keys]
        elif isinstance(dist, list):
            probs = [float(x) for x in dist[: len(keys)]]
        else:
            return Answer.failed(q, "no distribution")
        return finalize_answer(q, probs, confidence=conf)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        body = self._render(request)
        delay, last = 0.5, None
        for _ in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = self._client.post(self.url, json=body, headers=self.headers)
                ms = (time.perf_counter() - t0) * 1000
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 16)
        else:
            return DecisionResponse(answers={k: Answer.failed(q, "transport") for k, q in request.questions.items()},
                                    latency=LatencyRecord(client_ms=float("nan"), timestamp=time.time()),
                                    provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id),
                                    transport_error=str(last))
        raw_answers = _dig(data, self.answers_path, {}) or {}
        answers = {k: (self._parse(q, raw_answers[k]) if isinstance(raw_answers.get(k), dict) else Answer.failed(q, "missing answer"))
                   for k, q in request.questions.items()}
        in_tok = _first_path(data, self.fields.get("input_tokens", [["usage", "prompt_tokens"], ["usage", "input_tokens"]]))
        cost = (in_tok / 1e6 * self.price) if (in_tok and self.price) else None
        return DecisionResponse(
            answers=answers, latency=LatencyRecord(client_ms=ms, timestamp=time.time()),
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id,
                                    model_id_returned=_first_path(data, self.fields.get("model_returned", [["model"]])),
                                    generation_id=_first_path(data, self.fields.get("generation_id", [["id"]])),
                                    billed_input_tokens=in_tok, cost_usd=cost),
            raw=data,
        )
