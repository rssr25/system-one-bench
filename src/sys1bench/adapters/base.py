from __future__ import annotations

import importlib.metadata as md
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from ..schemas import Answer, DecisionRequest, DecisionResponse, ModelCapabilities, Question

_REGISTRY: dict[str, type[BaseAdapter]] = {}


def register(adapter_id: str) -> Callable[[type[BaseAdapter]], type[BaseAdapter]]:
    def deco(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        cls.adapter_id = adapter_id
        _REGISTRY[adapter_id] = cls
        return cls

    return deco


def _load_entry_points() -> None:
    try:
        eps = md.entry_points(group="sys1bench.adapters")
    except Exception:  # pragma: no cover
        return
    for ep in eps:
        if ep.name not in _REGISTRY:
            try:
                cls = ep.load()
                cls.adapter_id = ep.name
                _REGISTRY[ep.name] = cls
            except Exception:  # pragma: no cover
                continue


def list_adapters() -> list[str]:
    _load_entry_points()
    return sorted(_REGISTRY)


def get_adapter(adapter_id: str, **config: Any) -> BaseAdapter:
    _load_entry_points()
    if adapter_id not in _REGISTRY:
        raise KeyError(f"unknown adapter {adapter_id!r}; known: {list_adapters()}")
    return _REGISTRY[adapter_id](**config)


def detect_quantisation(probs: list[float]) -> float | None:
    """Smallest decimal step that reproduces every probability (0.01, 0.001, ...), or None if finer than 1e-6."""
    arr = np.asarray(probs, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    for step in (0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5, 1e-6):
        if np.allclose(np.round(arr / step) * step, arr, atol=step * 1e-3):
            return step
    return None


SUM_EXACT_TOL = 1e-4   # sums within this are taken as-is
SUM_SLACK_TOL = 0.02   # sums within this are rescaled and flagged `renormalised` (vendors quantise to 0.01)


def finalize_answer(q: Question, probs: list[float], *, confidence: float | None = None,
                    abstained: bool = False, truncated: bool = False, tol: float = SUM_SLACK_TOL) -> Answer:
    """Validate a probability vector against the contract. Vectors whose sum is off by more than `tol` are recorded as
    schema failures. Vectors off by less (quantisation slack, e.g. 0.99 from 0.01-rounded probabilities) are rescaled
    and flagged `renormalised=True` with the raw sum kept, so the rate is reportable and nothing is hidden."""
    keys = q.option_keys
    if len(probs) != len(keys):
        return Answer.failed(q, f"probs length {len(probs)} != options {len(keys)}")
    arr = np.asarray(probs, dtype=float)
    if not np.all(np.isfinite(arr)):
        return Answer.failed(q, "non-finite probability")
    if np.any(arr < -tol) or np.any(arr > 1 + tol):
        return Answer.failed(q, "probability outside [0,1]")
    s = float(arr.sum())
    if abs(s - 1.0) > tol or s <= 0:
        return Answer.failed(q, f"probabilities sum to {s:.6f}")
    step = detect_quantisation(list(arr))
    renorm = abs(s - 1.0) > SUM_EXACT_TOL
    if renorm:
        arr = arr / s
    arr = np.clip(arr, 0.0, 1.0)
    return Answer(
        type=q.type,
        option_keys=keys,
        probs=[float(x) for x in arr],
        argmax=keys[int(arr.argmax())],
        confidence=confidence,
        abstained=abstained,
        truncated=truncated,
        renormalised=renorm,
        raw_prob_sum=s,
        quantisation_step=step,
    )


class BaseAdapter(ABC):
    adapter_id: str = "base"

    def __init__(self, model_id: str = "", **tunables: Any) -> None:
        self.model_id = model_id
        self.tunables: dict[str, Any] = tunables

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities: ...

    @abstractmethod
    def decide(self, request: DecisionRequest) -> DecisionResponse: ...

    def decide_batch(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        return [self.decide(r) for r in requests]

    def reparse(self, request: DecisionRequest, cached: DecisionResponse) -> DecisionResponse | None:
        """Rebuild answers from `cached.raw` with the current parser (no API call). Adapters that store the vendor
        payload verbatim override `parse_raw_answer`; return None if nothing can be rebuilt."""
        raw_answers = (cached.raw or {}).get("answers") if isinstance(cached.raw, dict) else None
        if not isinstance(raw_answers, dict):
            return None
        answers = {}
        for k, q in request.questions.items():
            payload = raw_answers.get(k)
            answers[k] = self.parse_raw_answer(q, payload) if isinstance(payload, dict) else Answer.failed(q, "missing answer")
        return cached.model_copy(update={"answers": answers})

    def parse_raw_answer(self, q: Question, payload: dict) -> Answer:  # pragma: no cover - overridden
        raise NotImplementedError

    def warmup(self) -> None:  # pragma: no cover
        return None

    def check_request(self, request: DecisionRequest) -> list[str]:
        """Capability violations for this request (reported, not fatal)."""
        caps = self.capabilities
        issues: list[str] = []
        if caps.max_questions_per_request and len(request.questions) > caps.max_questions_per_request:
            issues.append("too_many_questions")
        for k, q in request.questions.items():
            if q.type not in caps.primitives:
                issues.append(f"{k}:unsupported_primitive")
            if q.type == "choice" and caps.max_options and q.cardinality > caps.max_options:
                issues.append(f"{k}:too_many_options")
            if q.type == "score" and caps.max_score_levels and q.cardinality > caps.max_score_levels:
                issues.append(f"{k}:too_many_levels")
        return issues

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000.0
