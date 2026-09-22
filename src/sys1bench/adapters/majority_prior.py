"""Class-prior baseline. Priors are fitted from a manifest (or uniform if none given)."""

from __future__ import annotations

import time
from collections import Counter

from ..schemas import (
    DecisionRequest,
    DecisionResponse,
    LatencyRecord,
    ModelCapabilities,
    ProviderRecord,
    TaskItem,
)
from .base import BaseAdapter, finalize_answer, register


@register("majority_prior")
class MajorityPriorAdapter(BaseAdapter):
    def __init__(self, model_id: str = "majority_prior", **tunables) -> None:
        super().__init__(model_id, **tunables)
        self.priors: dict[str, Counter] = {}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=self.model_id, deployment="local", supports_batching=True)

    def fit(self, items: list[TaskItem]) -> None:
        for it in items:
            for k, q in it.questions.items():
                if q.ground_truth is None:
                    continue
                gt = str(q.ground_truth).lower() if q.type == "noul" else str(q.ground_truth)
                self.priors.setdefault(q.framing_group or k, Counter())[gt] += 1

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = self._now_ms()
        answers = {}
        for k, q in request.questions.items():
            keys = q.option_keys
            counts = self.priors.get(q.framing_group or k)
            if counts:
                total = sum(counts.get(x, 0) for x in keys) + len(keys) * 1e-3
                probs = [(counts.get(x, 0) + 1e-3) / total for x in keys]
            else:
                probs = [1.0 / len(keys)] * len(keys)
            answers[k] = finalize_answer(q, probs)
        return DecisionResponse(
            answers=answers,
            latency=LatencyRecord(client_ms=self._now_ms() - t0, timestamp=time.time()),
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, cost_usd=0.0),
        )
