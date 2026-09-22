"""Hybrid baseline: a System One model decides; below a confidence threshold, escalate to a second
model (typically an LLM). Reports both the combined answer and which path was taken, so accuracy,
latency and cost can be plotted against escalation rate. This is the realistic production deployment
and the comparison vendors' marketing avoids."""

from __future__ import annotations

import time

from ..schemas import DecisionRequest, DecisionResponse, LatencyRecord, ModelCapabilities, ProviderRecord
from .base import BaseAdapter, get_adapter, register


@register("hybrid_router")
class HybridRouterAdapter(BaseAdapter):
    def __init__(self, model_id: str = "hybrid", primary: dict | None = None, fallback: dict | None = None,
                 threshold: float = 0.7, **tunables) -> None:
        super().__init__(model_id, **tunables)
        primary = dict(primary or {"adapter": "mock"})
        fallback = dict(fallback or {"adapter": "mock", "skill": 0.95, "latency_ms": 1500.0})
        self.primary = get_adapter(primary.pop("adapter"), **primary)
        self.fallback = get_adapter(fallback.pop("adapter"), **fallback)
        self.threshold = threshold
        self.tunables.update({"threshold": threshold})

    @property
    def capabilities(self) -> ModelCapabilities:
        p = self.primary.capabilities
        return p.model_copy(update={"name": f"hybrid({p.name}->{self.fallback.capabilities.name})@{self.threshold}",
                                    "deployment": "hosted" if "hosted" in (p.deployment, self.fallback.capabilities.deployment) else "local"})

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        first = self.primary.decide(request)
        low = [k for k, a in first.answers.items() if (not a.ok) or max(a.probs) < self.threshold]
        if not low:
            first.raw = {"path": "primary", "escalated": []}
            return first
        sub = DecisionRequest(state=request.state, questions={k: request.questions[k] for k in low}, meta=request.meta)
        second = self.fallback.decide(sub)
        answers = dict(first.answers)
        answers.update(second.answers)
        lat = LatencyRecord(client_ms=first.latency.client_ms + second.latency.client_ms, timestamp=time.time())
        cost = (first.provider.cost_usd or 0.0) + (second.provider.cost_usd or 0.0)
        return DecisionResponse(
            answers=answers, latency=lat,
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.capabilities.name, cost_usd=cost,
                                    version_hash=f"{first.provider.version_hash}+{second.provider.version_hash}"),
            raw={"path": "escalated", "escalated": low, "primary": first.raw, "fallback": second.raw},
        )
