"""Deterministic synthetic model for offline tests and CI.

Behaviour is controlled so metric kernels can be validated end to end:
`skill` sets P(argmax == truth); `temperature` scales logits (>1 under-confident,
<1 over-confident); `position_bias` adds mass to index 0; `quantise` rounds probs.
It reads `ground_truth` only through a hash of the state so it behaves like a real
model that is right some fraction of the time, not like an oracle.
"""

from __future__ import annotations

import hashlib
import random
import time

import numpy as np

from ..schemas import DecisionRequest, DecisionResponse, LatencyRecord, ModelCapabilities, ProviderRecord
from .base import BaseAdapter, finalize_answer, register


@register("mock")
class MockAdapter(BaseAdapter):
    def __init__(self, model_id: str = "mock-v1", skill: float = 0.8, temperature: float = 1.0,
                 position_bias: float = 0.0, quantise: float | None = None, framing_sensitivity: float = 0.0,
                 latency_ms: float = 5.0, **tunables) -> None:
        super().__init__(model_id, **tunables)
        self.skill, self.temperature, self.position_bias = skill, temperature, position_bias
        self.quantise, self.framing_sensitivity, self.latency_ms = quantise, framing_sensitivity, latency_ms

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=self.model_id, deployment="local", max_options=255, emits_confidence=True,
                                 supports_native_abstain=True, supports_batching=True)

    def _rng(self, request: DecisionRequest, key: str) -> random.Random:
        seed_src = f"{request.state_text()}|{key}|{self.model_id}"
        if self.framing_sensitivity > 0:
            seed_src += "|" + request.questions[key].instructions
        return random.Random(int(hashlib.sha256(seed_src.encode()).hexdigest()[:16], 16))

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = self._now_ms()
        answers = {}
        for key, q in request.questions.items():
            rng = self._rng(request, key)
            n = q.cardinality
            keys = q.option_keys
            truth = q.ground_truth
            truth_key = None
            if truth is not None:
                truth_key = str(truth).lower() if q.type == "noul" else str(truth)
            skill = self.skill
            if self.framing_sensitivity > 0:
                skill = max(0.0, min(1.0, skill + self.framing_sensitivity * (rng.random() - 0.5)))
            if truth_key in keys and rng.random() < skill:
                target = keys.index(truth_key)
            else:
                candidates = [i for i in range(n) if keys[i] != truth_key] or list(range(n))
                target = rng.choice(candidates)
            logits = np.array([rng.gauss(0, 1) for _ in range(n)])
            logits[target] += 3.0
            logits[0] += self.position_bias
            logits = logits / self.temperature
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            if self.quantise:
                probs = np.round(probs / self.quantise) * self.quantise
                probs[int(probs.argmax())] += 1.0 - probs.sum()  # keep sum exactly 1 like a vendor would
            answers[key] = finalize_answer(q, list(probs), confidence=float(probs.max()))
        return DecisionResponse(
            answers=answers,
            latency=LatencyRecord(client_ms=self.latency_ms + (self._now_ms() - t0), compute_ms=self.latency_ms,
                                  timestamp=time.time()),
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id,
                                    model_id_returned=self.model_id, version_hash="mock", cost_usd=0.0),
        )
