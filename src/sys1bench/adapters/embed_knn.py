"""Embedding baseline: embed the state and each option description, softmax over cosine similarity
with a temperature fitted on a calibration split. This is the honest apples-to-apples for
"zero-shot from label descriptions"; a System One model should beat it by a clear margin."""

from __future__ import annotations

import time

import numpy as np

from ..schemas import DecisionRequest, DecisionResponse, LatencyRecord, ModelCapabilities, ProviderRecord
from .base import BaseAdapter, finalize_answer, register


@register("embed_knn")
class EmbedKNNAdapter(BaseAdapter):
    def __init__(self, model_id: str = "BAAI/bge-m3", temperature: float = 0.05, device: str | None = None, **tunables) -> None:
        super().__init__(model_id, **tunables)
        self.temperature, self.device = temperature, device
        self._m = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=f"embed_knn:{self.model_id}", deployment="local", supports_batching=True,
                                 tunables={"temperature": self.temperature})

    def _model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._m = SentenceTransformer(self.model_id, device=self.device)
        return self._m

    def _embed(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in texts if t not in self._cache]
        if todo:
            vecs = self._model().encode(todo, normalize_embeddings=True, convert_to_numpy=True)
            for t, v in zip(todo, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        s = self._embed([request.state_text()])[0]
        answers = {}
        for k, q in request.questions.items():
            if q.type == "noul":
                texts = [f"{q.instructions} Yes.", f"{q.instructions} No."]
            elif q.type == "choice":
                texts = [f"{c.key}: {c.description}" if c.description else c.key for c in q.criteria]  # type: ignore[union-attr]
            else:
                texts = [f"level {c.level}: {c.description}" for c in q.criteria]  # type: ignore[union-attr]
            sims = self._embed(texts) @ s
            z = sims / self.temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            answers[k] = finalize_answer(q, [float(x) for x in p], confidence=float(p.max()))
        ms = (time.perf_counter() - t0) * 1000
        return DecisionResponse(answers=answers, latency=LatencyRecord(client_ms=ms, compute_ms=ms, timestamp=time.time()),
                                provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, cost_usd=0.0))
