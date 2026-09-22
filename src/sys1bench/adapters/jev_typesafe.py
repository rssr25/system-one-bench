"""TypeSafe Jev via the first-party API (preferred over the OpenRouter alpha route).

Endpoint: POST https://api.typesafe.ai/v1/systemone   Authorization: Bearer <TypeSafe_API_KEY>
Body:     {"state": <str|obj|list>, "model": "jev-1.13.0", "questions": {id: Question}}
Question: choice  -> {"type": "choice", "instructions": str|obj, "criteria": {option: description|null}}   (<=255 options)
          score   -> {"type": "score",  "instructions": str|obj, "criteria": [level_description, ...]}    (2..10 levels)
          noul    -> {"type": "noul",   "instructions": str|obj, "criteria": {"true": ..., "false": ...}?}
Answer:   choice  -> {"type","choice","probabilities":{option: p},"confidence"}
          score   -> {"type","score" (expected level index),"legend":{"0": desc,...},"probabilities":{"0": p,...},"confidence"}
          noul    -> {"type","noul": P(yes)}
Response also carries "model" (versioned id that answered) and "usage": {input_tokens, output_tokens}.
Limits (models page, 2026-09-22): 64k tokens per request, 32k for state plus longest question, 1200 rpm, 250k tok/s.
Price: $0.042 per million input tokens; output free.

Score probabilities are keyed by *level index*, so they are mapped back onto the manifest's level values by
position. Option order in `criteria` is preserved as JSON object order, which is what the permutation suite varies.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

from ..schemas import Answer, DecisionRequest, DecisionResponse, LatencyRecord, ModelCapabilities, ProviderRecord, Question
from .base import BaseAdapter, finalize_answer, register

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_M_INPUT = 0.042
ENV_KEYS = ("TypeSafe_API_KEY", "TYPESAFE_API_KEY")


def _load_dotenv_key() -> str | None:
    for k in ENV_KEYS:
        if os.environ.get(k):
            return os.environ[k]
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    name, _, val = line.partition("=")
                    if name.strip() in ENV_KEYS and val.strip():
                        return val.strip().strip('"').strip("'")
    return None


def to_vendor_question(q: Question) -> dict[str, Any]:
    if q.type == "noul":
        return {"type": "noul", "instructions": q.instructions}
    if q.type == "choice":
        crit: dict[str, Any] = {c.key: (c.description or None) for c in q.criteria}  # type: ignore[union-attr]
        if q.allow_abstain:
            crit[q.abstain_key] = "None of the listed options applies."
        return {"type": "choice", "instructions": q.instructions, "criteria": crit}
    levels = [(c.description or f"level {c.level}") for c in q.criteria]  # type: ignore[union-attr]
    return {"type": "score", "instructions": q.instructions, "criteria": levels}


def parse_answer(q: Question, payload: dict[str, Any]) -> Answer:
    keys = q.option_keys
    if q.type == "noul":
        p = payload.get("noul")
        if p is None:
            return Answer.failed(q, "no noul field")
        return finalize_answer(q, [float(p), 1.0 - float(p)])
    dist = payload.get("probabilities")
    if not isinstance(dist, dict):
        return Answer.failed(q, "no probabilities map")
    conf = payload.get("confidence")
    if q.type == "score":
        # keyed by level index as string, in the order we sent the levels
        probs = [float(dist.get(str(i), 0.0)) for i in range(len(keys))]
        return finalize_answer(q, probs, confidence=conf)
    probs = [float(dist.get(k, 0.0)) for k in keys]
    abstained = False
    if q.allow_abstain and q.abstain_key in dist:
        p_abst = float(dist[q.abstain_key])
        abstained = p_abst >= max(probs)
        s = sum(probs)
        probs = [p / s for p in probs] if s > 0 else [1.0 / len(keys)] * len(keys)
    return finalize_answer(q, probs, confidence=conf, abstained=abstained)


@register("jev_typesafe")
class JevTypeSafeAdapter(BaseAdapter):
    def __init__(self, model_id: str = "jev-1.13.0", url: str = DEFAULT_URL, api_key: str | None = None,
                 timeout_s: float = 30.0, max_retries: int = 6, allow_alias: bool = False, **tunables) -> None:
        super().__init__(model_id, **tunables)
        if (model_id.endswith("latest") or model_id.endswith("preview")) and not allow_alias:
            raise ValueError("pin a versioned Jev id (e.g. jev-1.13.0); aliases move silently. Pass allow_alias=True to override.")
        self.url, self.timeout_s, self.max_retries = url, timeout_s, max_retries
        self.api_key = api_key or _load_dotenv_key()
        self._client = httpx.Client(timeout=timeout_s, http2=False)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=self.model_id, deployment="hosted", max_options=255, max_score_levels=10,
                                 max_state_tokens=32_000, emits_confidence=True, supports_native_abstain=False,
                                 supports_batching=True, languages={"en"})

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, str]]:
        if not self.api_key:
            raise RuntimeError("TypeSafe_API_KEY not set (env or .env)")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay, last = 0.5, None
        for _ in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = self._client.post(self.url, json=body, headers=headers)
                ms = (time.perf_counter() - t0) * 1000
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    ra = r.headers.get("retry-after")
                    if ra:
                        try:
                            delay = max(delay, float(ra))
                        except ValueError:
                            pass
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    # request rejected (400 over the 32k state limit, 413, 422 malformed): never retry, classify as vendor rejection
                    raise ValueError(f"{r.status_code} rejected: {r.text[:300]}")
                r.raise_for_status()
                return r.json(), ms, dict(r.headers)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 20.0)
        raise RuntimeError(f"typesafe request failed after {self.max_retries} attempts: {last}")

    def parse_raw_answer(self, q: Question, payload: dict) -> Answer:
        return parse_answer(q, payload)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        body = {"state": request.state, "model": self.model_id,
                "questions": {k: to_vendor_question(q) for k, q in request.questions.items()}}
        try:
            data, ms, headers = self._post(body)
        except Exception as e:
            kind = "rejected_by_vendor" if isinstance(e, ValueError) else "transport"
            return DecisionResponse(
                answers={k: Answer.failed(q, kind) for k, q in request.questions.items()},
                latency=LatencyRecord(client_ms=float("nan"), route="typesafe", timestamp=time.time()),
                provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, route="typesafe"),
                transport_error=str(e)[:500],
            )
        raw_answers = data.get("answers", {}) or {}
        answers = {k: (parse_answer(q, raw_answers[k]) if isinstance(raw_answers.get(k), dict) else Answer.failed(q, "missing answer"))
                   for k, q in request.questions.items()}
        usage = data.get("usage", {}) or {}
        in_tok = usage.get("input_tokens")
        return DecisionResponse(
            answers=answers,
            latency=LatencyRecord(client_ms=ms, route="typesafe", timestamp=time.time()),
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, model_id_returned=data.get("model"),
                                    generation_id=headers.get("x-request-id") or headers.get("request-id"), route="typesafe",
                                    billed_input_tokens=in_tok, billed_output_tokens=usage.get("output_tokens"),
                                    cost_usd=(in_tok / 1e6 * PRICE_PER_M_INPUT) if in_tok is not None else None),
            raw=data,
        )
