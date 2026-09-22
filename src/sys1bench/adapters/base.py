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


def finalize_answer(q: Question, probs: list[float], *, confidence: float | None = None,
                    abstained: bool = False, truncated: bool = False, tol: float = 1e-4) -> Answer:
    """Validate a probability vector against the contract. Out-of-tolerance vectors are recorded as failures,
    never silently renormalised."""
    keys = q.option_keys
    if len(probs) != len(keys):
        return Answer.failed(q, f"probs length {len(probs)} != options {len(keys)}")
    arr = np.asarray(probs, dtype=float)
    if not np.all(np.isfinite(arr)):
        return Answer.failed(q, "non-finite probability")
    if np.any(arr < -tol) or np.any(arr > 1 + tol):
        return Answer.failed(q, "probability outside [0,1]")
    s = float(arr.sum())
    if abs(s - 1.0) > tol:
        return Answer.failed(q, f"probabilities sum to {s:.6f}")
    arr = np.clip(arr, 0.0, 1.0)
    return Answer(
        type=q.type,
        option_keys=keys,
        probs=[float(x) for x in arr],
        argmax=keys[int(arr.argmax())],
        confidence=confidence,
        abstained=abstained,
        truncated=truncated,
        raw_prob_sum=s,
        quantisation_step=detect_quantisation(list(arr)),
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
