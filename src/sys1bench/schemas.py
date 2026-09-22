"""Pydantic v2 schemas: manifest v2, the model contract, and capability declarations.

The contract is model-agnostic. A model participates by implementing
`BaseAdapter.decide` against `DecisionRequest` / `DecisionResponse` and declaring
`ModelCapabilities`. Nothing in the harness refers to a specific vendor.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Primitive = Literal["choice", "score", "noul"]
Tier = Literal["P", "G", "H"]


# --------------------------------------------------------------------------- manifest


class Option(BaseModel):
    key: str
    description: str = ""


class Level(BaseModel):
    level: int
    description: str = ""


class Question(BaseModel):
    type: Primitive
    instructions: str
    criteria: list[Option] | list[Level] | None = None
    ground_truth: str | int | bool | None = None
    framing_group: str | None = None
    framing_id: str = "f0"
    decomposition_of: str | None = None
    allow_abstain: bool = False
    abstain_key: str = "none_of_the_above"

    @model_validator(mode="after")
    def _check(self) -> Question:
        if self.type == "choice":
            if not self.criteria or not all(isinstance(c, Option) for c in self.criteria):
                raise ValueError("choice questions need list[Option] criteria")
            keys = [c.key for c in self.criteria]
            if len(set(keys)) != len(keys):
                raise ValueError("duplicate option keys")
            if self.ground_truth is not None and self.ground_truth not in keys + [self.abstain_key]:
                raise ValueError(f"ground_truth {self.ground_truth!r} not among option keys")
        elif self.type == "score":
            if not self.criteria or not all(isinstance(c, Level) for c in self.criteria):
                raise ValueError("score questions need list[Level] criteria")
            levels = [c.level for c in self.criteria]
            if levels != sorted(levels) or len(set(levels)) != len(levels):
                raise ValueError("score levels must be strictly increasing")
            if not 2 <= len(levels) <= 10:
                raise ValueError("score questions need 2..10 levels")
            if self.ground_truth is not None and self.ground_truth not in levels:
                raise ValueError("ground_truth not among levels")
        elif self.type == "noul":
            if self.criteria is not None and len(self.criteria) > 0:
                raise ValueError("noul questions take no criteria")
            if self.ground_truth is not None and not isinstance(self.ground_truth, bool):
                raise ValueError("noul ground_truth must be bool")
        return self

    @property
    def option_keys(self) -> list[str]:
        if self.type == "choice":
            return [c.key for c in self.criteria]  # type: ignore[union-attr]
        if self.type == "score":
            return [str(c.level) for c in self.criteria]  # type: ignore[union-attr]
        return ["true", "false"]

    @property
    def cardinality(self) -> int:
        return len(self.option_keys)


class LabelProvenance(BaseModel):
    source: Literal["generator", "human", "public", "llm_reference"]
    generator: str | None = None
    seed: int | None = None
    noise_injected: bool = False
    annotators: int | None = None
    agreement: float | None = None


class Controls(BaseModel):
    unknowable: bool = False
    none_correct: bool = False
    distractor_density: float = 0.0
    perturbation: str | None = None
    prior_shift: str | None = None


class TaskItem(BaseModel):
    """One manifest row. Framing variants are separate rows sharing task_id."""

    task_id: str
    tier: Tier
    domain: str
    language: str = "en"
    state: str | dict[str, Any] | list[Any]
    state_tokens: int | None = None
    questions: dict[str, Question]
    label_provenance: LabelProvenance
    controls: Controls = Field(default_factory=Controls)
    cost_matrix: dict[str, dict[str, dict[str, float]]] | None = None  # question_key -> {true_label -> {pred -> cost}}
    canary: str | None = None
    permutation_id: str = "p0"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("questions")
    @classmethod
    def _non_empty(cls, v: dict[str, Question]) -> dict[str, Question]:
        if not v:
            raise ValueError("at least one question")
        return v

    def state_text(self) -> str:
        if isinstance(self.state, str):
            return self.state
        return json.dumps(self.state, sort_keys=True, ensure_ascii=False)

    def row_id(self) -> str:
        parts = [self.task_id, self.permutation_id] + [f"{k}:{q.framing_id}" for k, q in sorted(self.questions.items())]
        return "|".join(parts)


# --------------------------------------------------------------------------- contract


class ModelCapabilities(BaseModel):
    """What a model can accept. Checked at manifest-load time; violations become metrics, not crashes."""

    name: str
    deployment: Literal["hosted", "local"]
    primitives: set[Primitive] = {"choice", "score", "noul"}
    max_options: int | None = None
    max_score_levels: int | None = 10
    max_state_tokens: int | None = None
    max_questions_per_request: int | None = None
    emits_probabilities: bool = True
    emits_confidence: bool = False
    supports_native_abstain: bool = False
    supports_batching: bool = False
    deterministic: bool = True
    languages: set[str] | None = None  # None = undocumented
    tunables: dict[str, Any] = Field(default_factory=dict)  # e.g. Laya head_max_len


class RequestMeta(BaseModel):
    task_id: str
    row_id: str
    permutation_id: str = "p0"
    framing_ids: dict[str, str] = Field(default_factory=dict)
    suite: str | None = None
    arm: str | None = None


class DecisionRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question]
    meta: RequestMeta

    def state_text(self) -> str:
        return self.state if isinstance(self.state, str) else json.dumps(self.state, sort_keys=True, ensure_ascii=False)

    def cache_key(self, adapter_id: str, model_id: str, tunables: dict[str, Any] | None = None) -> str:
        payload = {
            "adapter": adapter_id,
            "model": model_id,
            "tunables": tunables or {},
            "state": self.state,
            "questions": {k: q.model_dump(exclude={"ground_truth", "framing_group"}) for k, q in self.questions.items()},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class LatencyRecord(BaseModel):
    client_ms: float
    server_ms: float | None = None
    compute_ms: float | None = None
    batch_size: int = 1
    concurrency: int = 1
    route: str | None = None
    timestamp: float | None = None


class ProviderRecord(BaseModel):
    adapter_id: str
    model_id_requested: str
    model_id_returned: str | None = None
    generation_id: str | None = None
    version_hash: str | None = None
    route: str | None = None
    hardware: str | None = None
    billed_input_tokens: int | None = None
    billed_output_tokens: int | None = None
    cost_usd: float | None = None


class Answer(BaseModel):
    type: Primitive
    option_keys: list[str]
    probs: list[float]
    argmax: str
    confidence: float | None = None
    abstained: bool = False
    truncated: bool = False
    renormalised: bool = False          # raw vector summed to 1 ± tol (quantisation slack) and was rescaled
    raw_prob_sum: float | None = None
    quantisation_step: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def prob_of(self, key: str) -> float:
        return self.probs[self.option_keys.index(key)]

    @staticmethod
    def failed(q: Question, error: str) -> Answer:
        n = q.cardinality
        return Answer(type=q.type, option_keys=q.option_keys, probs=[float("nan")] * n, argmax="", error=error)


class DecisionResponse(BaseModel):
    answers: dict[str, Answer]
    latency: LatencyRecord
    provider: ProviderRecord
    raw: dict[str, Any] = Field(default_factory=dict)
    transport_error: str | None = None


class PredictionRow(BaseModel):
    """Flat per-question record written to predictions.jsonl; the unit of all analysis."""

    row_id: str
    task_id: str
    question_key: str
    primitive: Primitive
    tier: Tier
    domain: str
    language: str
    framing_id: str
    framing_group: str | None
    permutation_id: str
    cardinality: int
    option_keys: list[str]
    probs: list[float]
    argmax: str
    ground_truth: str | None
    correct: bool | None
    confidence: float | None
    abstained: bool
    truncated: bool
    renormalised: bool = False
    error: str | None
    quantisation_step: float | None
    controls: Controls
    latency_ms: float
    cost_usd: float | None
    adapter_id: str
    model_id: str
    version_hash: str | None
    suite: str | None = None
    arm: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
