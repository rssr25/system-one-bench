"""Hand-written rule baseline. Rules come from a YAML file per domain:

    queue:
      billing: ["refund", "charged", "invoice"]
      shipping: ["deliver", "tracking", "package"]
    is_angry: ["unacceptable", "furious", "ridiculous"]      # noul: any match -> true
    priority: {"3": ["urgent", "asap"], "4": ["outage", "down"]}  # score: highest matching level wins

Confidence is a fixed soft score (match -> 0.9 on the winner) so the baseline gets calibration
metrics too, which shows what a rule system's probabilities are worth.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import yaml

from ..schemas import DecisionRequest, DecisionResponse, LatencyRecord, ModelCapabilities, ProviderRecord
from .base import BaseAdapter, finalize_answer, register


@register("regex_keyword")
class RegexKeywordAdapter(BaseAdapter):
    def __init__(self, model_id: str = "regex_keyword", rules_path: str | None = None, rules: dict | None = None,
                 hit_confidence: float = 0.9, **tunables) -> None:
        super().__init__(model_id, **tunables)
        if rules is None and rules_path:
            rules = yaml.safe_load(Path(rules_path).read_text())
        self.rules = rules or {}
        self.hit_confidence = hit_confidence
        self._compiled: dict[str, dict[str, list[re.Pattern]]] = {}
        for qkey, spec in self.rules.items():
            if isinstance(spec, list):
                self._compiled[qkey] = {"true": [re.compile(p, re.IGNORECASE) for p in spec]}
            else:
                self._compiled[qkey] = {str(k): [re.compile(p, re.IGNORECASE) for p in v] for k, v in spec.items()}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=self.model_id, deployment="local", emits_confidence=True, supports_batching=True)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = self._now_ms()
        text = request.state_text()
        answers = {}
        for k, q in request.questions.items():
            keys = q.option_keys
            rules = self._compiled.get(q.framing_group or k) or self._compiled.get(k) or {}
            hits = {opt: sum(1 for p in pats if p.search(text)) for opt, pats in rules.items()}
            hits = {o: h for o, h in hits.items() if h > 0 and (o in keys or q.type == "noul")}
            n = len(keys)
            if not hits:
                probs = [1.0 / n] * n
            elif q.type == "noul":
                probs = [self.hit_confidence, 1.0 - self.hit_confidence]
            elif q.type == "score":
                winner = str(max(int(o) for o in hits))
                probs = [(self.hit_confidence if o == winner else (1 - self.hit_confidence) / (n - 1)) for o in keys]
            else:
                winner = max(hits, key=hits.get)
                probs = [(self.hit_confidence if o == winner else (1 - self.hit_confidence) / (n - 1)) for o in keys]
            answers[k] = finalize_answer(q, probs, confidence=max(probs))
        return DecisionResponse(
            answers=answers,
            latency=LatencyRecord(client_ms=self._now_ms() - t0, timestamp=time.time()),
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, cost_usd=0.0),
        )
