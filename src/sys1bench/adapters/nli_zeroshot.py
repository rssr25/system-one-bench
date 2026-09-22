"""Zero-shot NLI baseline (BART-large-MNLI / DeBERTa-v3-large-MNLI). Hypothesis = label description."""

from __future__ import annotations

import time

import numpy as np

from ..schemas import DecisionRequest, DecisionResponse, LatencyRecord, ModelCapabilities, ProviderRecord
from .base import BaseAdapter, finalize_answer, register


@register("nli_zeroshot")
class NLIZeroShotAdapter(BaseAdapter):
    def __init__(self, model_id: str = "facebook/bart-large-mnli", device: int | str = -1,
                 hypothesis_template: str = "This text is about {}.", **tunables) -> None:
        super().__init__(model_id, **tunables)
        self.device, self.template = device, hypothesis_template
        self._pipe = None

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=f"nli:{self.model_id}", deployment="local", supports_batching=True)

    def _p(self):
        if self._pipe is None:
            from transformers import pipeline  # type: ignore

            self._pipe = pipeline("zero-shot-classification", model=self.model_id, device=self.device)
        return self._pipe

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        text = request.state_text()
        answers = {}
        for k, q in request.questions.items():
            if q.type == "noul":
                labels = [f"{q.instructions} yes", f"{q.instructions} no"]
            elif q.type == "choice":
                labels = [c.description or c.key for c in q.criteria]  # type: ignore[union-attr]
            else:
                labels = [c.description or f"level {c.level}" for c in q.criteria]  # type: ignore[union-attr]
            out = self._p()(text, candidate_labels=labels, hypothesis_template=self.template, multi_label=False)
            score = dict(zip(out["labels"], out["scores"]))
            p = np.array([score[lab] for lab in labels])
            p /= p.sum()
            answers[k] = finalize_answer(q, [float(x) for x in p], confidence=float(p.max()))
        ms = (time.perf_counter() - t0) * 1000
        return DecisionResponse(answers=answers, latency=LatencyRecord(client_ms=ms, compute_ms=ms, timestamp=time.time()),
                                provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, cost_usd=0.0))
