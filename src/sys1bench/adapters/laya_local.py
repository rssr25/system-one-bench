"""Convai Laya, run locally (Apache 2.0). Requires `pip install laya` (or the package from
github.com/NandhaKishorM/laya) plus torch. Exposes the option token budget (`head_max_len`) and
context (`max_len`) as tunables so Suite C can ablate them; the 77-label collapse at the default
budget is a configuration effect that must be separated from architecture.

Checkpoints: laya (ModernBERT-large, 421M, ctx 512), laya-multilingual (mmBERT-base, 322M, ctx 1024),
laya-typed-decisions (fine-tuned, ctx 1024). `checkpoint="router"` uses Laya's language router.
"""

from __future__ import annotations

import time
from typing import Any

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

CONTEXT = {"laya": 512, "laya-multilingual": 1024, "laya-typed-decisions": 1024, "router": 1024}
DEFAULT_HEAD = {"laya": 192, "laya-multilingual": 256, "laya-typed-decisions": 192, "router": 256}


def _to_vendor_question(q: Question) -> dict[str, Any]:
    if q.type == "noul":
        return {"type": "noul", "instructions": q.instructions}
    if q.type == "choice":
        crit = [{"key": c.key, "description": c.description} for c in q.criteria]  # type: ignore[union-attr]
        if q.allow_abstain:
            crit.append({"key": q.abstain_key, "description": "None of the listed options applies."})
        return {"type": "choice", "instructions": q.instructions, "criteria": crit}
    return {"type": "score", "instructions": q.instructions,
            "criteria": [{"level": c.level, "description": c.description} for c in q.criteria]}  # type: ignore[union-attr]


def parse_answer(q: Question, payload: dict[str, Any]) -> Answer:
    keys = q.option_keys
    conf = payload.get("confidence")
    if q.type == "noul":
        p = payload.get("probability", payload.get("p_true"))
        if p is None:
            return Answer.failed(q, "no probability")
        return finalize_answer(q, [float(p), 1 - float(p)], confidence=conf)
    dist = payload.get("probabilities", payload.get("distribution"))
    if isinstance(dist, dict):
        probs = [float(dist.get(k, dist.get(int(k) if k.isdigit() else k, 0.0))) for k in keys]
    elif isinstance(dist, list):
        probs = [float(x) for x in dist[: len(keys)]]
    else:
        return Answer.failed(q, "no distribution")
    escalate = bool(payload.get("escalate", False))
    if q.allow_abstain and isinstance(dist, dict) and q.abstain_key in dist:
        s = sum(probs)
        probs = [p / s for p in probs] if s > 0 else probs
    return finalize_answer(q, probs, confidence=conf, abstained=escalate)


@register("laya_local")
class LayaLocalAdapter(BaseAdapter):
    def __init__(self, model_id: str = "convaiinnovations/laya", checkpoint: str = "laya", device: str = "cuda",
                 head_max_len: int | None = None, max_len: int | None = None, batch_size: int = 1,
                 hardware: str | None = None, **tunables) -> None:
        super().__init__(model_id, **tunables)
        self.checkpoint, self.device, self.batch_size, self.hardware = checkpoint, device, batch_size, hardware
        self.head_max_len = head_max_len or DEFAULT_HEAD[checkpoint]
        self.max_len = max_len or CONTEXT[checkpoint]
        self.tunables.update({"head_max_len": self.head_max_len, "max_len": self.max_len, "checkpoint": checkpoint})
        self._model = None

    @property
    def capabilities(self) -> ModelCapabilities:
        per_opt = max(1, self.head_max_len // 4)
        return ModelCapabilities(name=f"{self.model_id}:{self.checkpoint}", deployment="local", max_options=255,
                                 max_state_tokens=self.max_len - self.head_max_len, emits_confidence=True,
                                 supports_native_abstain=True, supports_batching=True,
                                 languages={"en"} if self.checkpoint == "laya" else None,
                                 tunables={"head_max_len": self.head_max_len, "max_len": self.max_len,
                                           "approx_tokens_per_option_at_20": per_opt})

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import laya  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("install Laya: pip install laya (github.com/NandhaKishorM/laya)") from e
        if self.checkpoint == "router":
            self._model = laya.Router(preload=True)
        else:
            self._model = laya.Laya.from_pretrained(self.checkpoint, device=self.device)  # type: ignore[attr-defined]
        return self._model

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        model = self._load()
        vq = {k: _to_vendor_question(q) for k, q in request.questions.items()}
        kwargs = {"head_max_len": self.head_max_len, "max_len": self.max_len}
        t0 = time.perf_counter()
        try:
            out = model.predict(request.state, vq, **kwargs)
        except TypeError:
            out = model.predict(request.state, vq)
        ms = (time.perf_counter() - t0) * 1000
        raw_answers = out.get("answers", out) if isinstance(out, dict) else {}
        truncated = bool(out.get("truncated", False)) if isinstance(out, dict) else False
        answers = {}
        for k, q in request.questions.items():
            p = raw_answers.get(k)
            a = parse_answer(q, p) if isinstance(p, dict) else Answer.failed(q, "missing answer")
            a.truncated = truncated
            answers[k] = a
        return DecisionResponse(
            answers=answers,
            latency=LatencyRecord(client_ms=ms, compute_ms=ms, batch_size=self.batch_size, timestamp=time.time()),
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id,
                                    model_id_returned=str(out.get("model", self.checkpoint)) if isinstance(out, dict) else self.checkpoint,
                                    version_hash=str(out.get("routing", {}).get("checkpoint", self.checkpoint)) if isinstance(out, dict) else None,
                                    hardware=self.hardware, cost_usd=0.0),
            raw=out if isinstance(out, dict) else {"out": str(out)},
        )
