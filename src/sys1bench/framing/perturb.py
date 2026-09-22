"""Surface perturbations for Suite D: casing, whitespace, character typos, unicode homoglyphs. Labels unchanged."""

from __future__ import annotations

import random
import re
from typing import Any

from ..schemas import TaskItem

HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}  # Cyrillic look-alikes


def _walk(x: Any, fn):
    if isinstance(x, str):
        return fn(x)
    if isinstance(x, dict):
        return {k: _walk(v, fn) for k, v in x.items()}
    if isinstance(x, list):
        return [_walk(v, fn) for v in x]
    return x


def typos(text: str, rate: float, rng: random.Random) -> str:
    out = list(text)
    for i, ch in enumerate(out):
        if ch.isalpha() and rng.random() < rate:
            op = rng.random()
            if op < 0.4 and i + 1 < len(out):
                out[i], out[i + 1] = out[i + 1], out[i]
            elif op < 0.7:
                out[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            else:
                out[i] = ""
    return "".join(out)


def homoglyphs(text: str, rate: float, rng: random.Random) -> str:
    return "".join(HOMOGLYPHS.get(c, c) if (c in HOMOGLYPHS and rng.random() < rate) else c for c in text)


def whitespace(text: str, rng: random.Random) -> str:
    text = re.sub(r" ", lambda m: " " * rng.choice([1, 1, 2, 3]), text)
    return text.replace(". ", ".\n") if rng.random() < 0.5 else text


PERTURBATIONS = {
    "upper": lambda t, rng: t.upper(),
    "lower": lambda t, rng: t.lower(),
    "whitespace": lambda t, rng: whitespace(t, rng),
    "typos_2pct": lambda t, rng: typos(t, 0.02, rng),
    "typos_5pct": lambda t, rng: typos(t, 0.05, rng),
    "homoglyphs_10pct": lambda t, rng: homoglyphs(t, 0.10, rng),
}


def perturb_items(items: list[TaskItem], kind: str, seed: int = 0) -> list[TaskItem]:
    rng = random.Random(f"{kind}:{seed}")
    fn = PERTURBATIONS[kind]
    out = []
    for it in items:
        row = it.model_copy(deep=True)
        row.state = _walk(row.state, lambda t: fn(t, rng))
        row.permutation_id = f"pert_{kind}"
        row.controls.perturbation = kind
        out.append(row)
    return out
