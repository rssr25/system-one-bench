"""TypeSafe Jev via the OpenRouter alpha Decisions API.

Endpoint: POST https://openrouter.ai/api/alpha/decisions
Body:     {"model": "typesafe/jev-1.13", "state": <str|obj|list>, "questions": {key: {type, instructions, criteria}}}
Types:    choice (<=255 options), score (2..10 levels), boolean (our "noul").
Limits:   32k state tokens, 64k per request, 1200 rpm. Input billed at $0.042/M tokens, output free.

Response parsing is defensive: the alpha API's field names may move. Everything received is stored
verbatim in `raw`, so re-parsing later never needs another API call. The model id string returned by
the provider is recorded because `jev-latest` can move silently; pin `jev-1.13`.
"""

from __future__ import annotations

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

DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
PRICE_PER_M_INPUT = 0.042


def _to_vendor_question(q: Question) -> dict[str, Any]:
    if q.type == "noul":
        return {"type": "boolean", "instructions": q.instructions}
    if q.type == "choice":
        opts = [{"key": c.key, "description": c.description} for c in q.criteria]  # type: ignore[union-attr]
        if q.allow_abstain:
            opts.append({"key": q.abstain_key, "description": "None of the listed options applies."})
        return {"type": "choice", "instructions": q.instructions, "criteria": opts}
    levels = [{"level": c.level, "description": c.description} for c in q.criteria]  # type: ignore[union-attr]
    return {"type": "score", "instructions": q.instructions, "criteria": levels}


def _first(d: dict, *names: str, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def parse_answer(q: Question, payload: dict[str, Any]) -> Answer:
    """Map a vendor answer object to the contract. Handles the shapes seen in public harnesses:
    choice: {"choice": key, "probabilities": {key: p} | [p...], "confidence": c}
    score:  {"score"|"expected": x, "distribution"|"probabilities": {...}|[...], "confidence": c}
    boolean:{"probability"|"p_true"|"value": p, "confidence": c}
    """
    keys = q.option_keys
    conf = _first(payload, "confidence")
    if q.type == "noul":
        p = _first(payload, "probability", "p_true", "prob", "value")
        if isinstance(p, bool):
            p = 1.0 if p else 0.0
        if p is None:
            return Answer.failed(q, "no probability field")
        return finalize_answer(q, [float(p), 1.0 - float(p)], confidence=conf)
    dist = _first(payload, "probabilities", "distribution", "probs")
    if dist is None:
        return Answer.failed(q, "no distribution field")
    if isinstance(dist, dict):
        probs = [float(dist.get(k, dist.get(int(k) if k.isdigit() else k, 0.0))) for k in keys]
        extra = set(map(str, dist)) - set(keys)
        abst = q.allow_abstain and q.abstain_key in extra
    elif isinstance(dist, list):
        if all(isinstance(x, dict) for x in dist):
            m = {str(_first(x, "key", "level", "option")): float(_first(x, "probability", "p", "prob")) for x in dist}
            probs = [m.get(k, 0.0) for k in keys]
            abst = q.allow_abstain and q.abstain_key in m
        else:
            probs = [float(x) for x in dist[: len(keys)]]
            abst = False
    else:
        return Answer.failed(q, f"unparseable distribution {type(dist).__name__}")
    if abst:
        # abstain mass is recorded, then folded out so the vector stays over the manifest's options
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
    ans = finalize_answer(q, probs, confidence=conf, abstained=bool(abst and max(probs) < 0.5))
    return ans


@register("jev_openrouter")
class JevOpenRouterAdapter(BaseAdapter):
    def __init__(self, model_id: str = "typesafe/jev-1.13", url: str = DEFAULT_URL, api_key: str | None = None,
                 timeout_s: float = 30.0, max_retries: int = 5, route: str = "openrouter", **tunables) -> None:
        super().__init__(model_id, **tunables)
        if model_id.endswith("latest"):
            raise ValueError("pin an explicit Jev version (e.g. typesafe/jev-1.13); 'latest' moves silently")
        self.url, self.timeout_s, self.max_retries, self.route = url, timeout_s, max_retries, route
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = httpx.Client(timeout=timeout_s)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=self.model_id, deployment="hosted", max_options=255, max_score_levels=10,
                                 max_state_tokens=32_000, emits_confidence=True, supports_native_abstain=False,
                                 supports_batching=True, languages=None)

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, str]]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = self._client.post(self.url, json=body, headers=headers)
                ms = (time.perf_counter() - t0) * 1000
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r.json(), ms, dict(r.headers)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last_exc = e
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 16.0)
        raise RuntimeError(f"jev request failed after {self.max_retries} attempts: {last_exc}")

    def parse_raw_answer(self, q: Question, payload: dict) -> Answer:
        return parse_answer(q, payload)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        body = {"model": self.model_id, "state": request.state,
                "questions": {k: _to_vendor_question(q) for k, q in request.questions.items()}}
        try:
            data, ms, headers = self._post(body)
        except Exception as e:
            return DecisionResponse(
                answers={k: Answer.failed(q, "transport") for k, q in request.questions.items()},
                latency=LatencyRecord(client_ms=float("nan"), route=self.route, timestamp=time.time()),
                provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, route=self.route),
                transport_error=str(e),
            )
        raw_answers = _first(data, "answers", "decisions", "results", default={})
        answers = {}
        for k, q in request.questions.items():
            payload = raw_answers.get(k) if isinstance(raw_answers, dict) else None
            answers[k] = parse_answer(q, payload) if isinstance(payload, dict) else Answer.failed(q, "missing answer")
        usage = data.get("usage", {}) or {}
        in_tok = _first(usage, "prompt_tokens", "input_tokens")
        cost = (in_tok / 1e6) * PRICE_PER_M_INPUT if in_tok else _first(usage, "cost")
        server_ms = None
        for h in ("x-processing-ms", "openrouter-processing-ms", "x-response-time"):
            if h in headers:
                try:
                    server_ms = float(str(headers[h]).rstrip("ms"))
                except ValueError:
                    pass
        return DecisionResponse(
            answers=answers,
            latency=LatencyRecord(client_ms=ms, server_ms=server_ms, route=self.route, timestamp=time.time()),
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id,
                                    model_id_returned=_first(data, "model"), generation_id=_first(data, "id", "generation_id"),
                                    route=self.route, billed_input_tokens=in_tok, billed_output_tokens=0, cost_usd=cost),
            raw=data,
        )
