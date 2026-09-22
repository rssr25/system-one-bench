"""Tier G generators: pure functions of (version, seed, knobs) producing manifest rows whose labels
depend on a stated policy that is not fully derivable from the state text. Each generator also emits
a regex rule file so the trivial-baseline ceiling is always available."""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..schemas import TaskItem

GENERATORS: dict[str, type[BaseGenerator]] = {}


def register_generator(name: str) -> Callable[[type[BaseGenerator]], type[BaseGenerator]]:
    def deco(cls):
        cls.name = name
        GENERATORS[name] = cls
        return cls

    return deco


def get_generator(name: str, **knobs: Any) -> BaseGenerator:
    if name not in GENERATORS:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(GENERATORS)}")
    return GENERATORS[name](**knobs)


@dataclass
class Knobs:
    n: int = 1000
    seed: int = 42
    cardinality: int | None = None          # override number of choice options where supported
    target_tokens: int | None = None        # pad state with realistic filler to about this many tokens
    distractor_density: float = 0.0         # 0..1 fraction of items that get injected distractor content
    label_noise: float = 0.0                # fraction of labels randomly corrupted (control arm)
    unknowable_frac: float = 0.0            # fraction of items whose label depends on removed information
    none_correct_frac: float = 0.0          # fraction of choice items whose true answer is not among options
    language: str = "en"
    extra: dict[str, Any] = field(default_factory=dict)


FILLER = [
    "Previous interaction notes: agent confirmed identity via security question; customer verified email on file.",
    "System: session id {sid}; channel web; client version 4.{minor}.{patch}; locale en-US; ab_test_bucket {bucket}.",
    "Internal: this account was migrated from the legacy platform in Q{q} and has {n} archived conversations.",
    "Customer added: \"Also, unrelated, but your app's dark mode looks great now, whoever did that, thanks.\"",
    "Attachment metadata: {n} file(s), total {kb} KB, types: pdf, png. Virus scan clean.",
    "Survey snippet from last month: rated support {stars}/5, comment: \"quick reply, issue resolved\".",
    "Agent macro history: greeting_v2, ask_order_number, hold_music_long, wrap_up_v1.",
    "Timezone: UTC{tz}; preferred contact window 09:00-17:00; do-not-call flag false.",
]


def pad_to_tokens(rng: random.Random, text: str, target_tokens: int | None) -> str:
    """Approximate tokens as words * 1.3. Pads with realistic filler that never changes any label."""
    if not target_tokens:
        return text
    parts = [text]
    est = lambda s: int(len(s.split()) * 1.3)
    while est(" ".join(parts)) < target_tokens:
        f = rng.choice(FILLER).format(sid=rng.randrange(10**6), minor=rng.randrange(20), patch=rng.randrange(9),
                                      bucket=rng.choice("ABCD"), q=rng.randrange(1, 5), n=rng.randrange(1, 40),
                                      kb=rng.randrange(20, 4000), stars=rng.randrange(3, 6), tz=rng.choice(["-8", "-5", "+1", "+5:30"]))
        parts.append(f)
    return "\n".join(parts)


def canary_for(task_id: str, secret: str = "s1b-canary-v1") -> str:
    return "s1b-" + hashlib.sha256(f"{secret}:{task_id}".encode()).hexdigest()[:12]


class BaseGenerator(ABC):
    name: str = "base"
    version: str = "0.0.0"

    def __init__(self, **knobs: Any) -> None:
        self.knobs = Knobs(**knobs)

    @property
    def generator_id(self) -> str:
        return f"{self.name}@{self.version}"

    @abstractmethod
    def generate(self) -> list[TaskItem]: ...

    @abstractmethod
    def regex_rules(self) -> dict[str, Any]: ...

    @abstractmethod
    def cost_matrices(self) -> dict[str, dict[str, dict[str, float]]]: ...

    def rng(self, salt: str = "") -> random.Random:
        return random.Random(f"{self.generator_id}:{self.knobs.seed}:{salt}")
