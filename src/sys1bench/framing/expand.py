"""Framing expansion (Suite B), permutation (Suite D), short-circuit variants (meta-eval), decomposition combiners.

Paraphrases live in YAML keyed by framing_group:

    tickets.queue:
      paraphrases:
        - "Which queue should handle this ticket?"
        - "Route this ticket to the correct team."
      criteria_variants:
        label_only: true
        with_negatives:
          billing: "Do not use for cancellations or fraud."
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..schemas import Option, TaskItem


def load_framings(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _clone(item: TaskItem) -> TaskItem:
    return item.model_copy(deep=True)


def expand_framings(items: list[TaskItem], framings: dict[str, Any]) -> list[TaskItem]:
    """For each item, emit one row per framing variant of each question (all questions of a row share a
    framing index so pairs stay aligned). Row f0 is the canonical framing already in the manifest."""
    out: list[TaskItem] = []
    for it in items:
        out.append(it)
        max_var = 0
        for k, q in it.questions.items():
            spec = framings.get(q.framing_group or k, {})
            max_var = max(max_var, len(spec.get("paraphrases", [])))
        for v in range(max_var):
            row = _clone(it)
            changed = False
            for k, q in row.questions.items():
                spec = framings.get(q.framing_group or k, {})
                paras = spec.get("paraphrases", [])
                if v < len(paras):
                    q.instructions = paras[v]
                    q.framing_id = f"para{v+1}"
                    changed = True
            if changed:
                out.append(row)
        # adversarial paraphrases (semantically equivalent wordings contributed to lower accuracy); reported as a worst case
        max_adv = max((len(framings.get(q.framing_group or k, {}).get("adversarial", [])) for k, q in it.questions.items()), default=0)
        for v in range(max_adv):
            row = _clone(it)
            changed = False
            for k, q in row.questions.items():
                adv = framings.get(q.framing_group or k, {}).get("adversarial", [])
                if v < len(adv):
                    q.instructions = adv[v]
                    q.framing_id = f"adv{v+1}"
                    changed = True
            if changed:
                out.append(row)
        # criteria granularity variants for choice questions
        for name in ("label_only", "with_negatives", "vague"):
            row = _clone(it)
            changed = False
            for k, q in row.questions.items():
                if q.type != "choice":
                    continue
                spec = framings.get(q.framing_group or k, {})
                cv = spec.get("criteria_variants", {})
                if name == "label_only" and cv.get("label_only"):
                    q.criteria = [Option(key=o.key, description="") for o in q.criteria]  # type: ignore[union-attr]
                    changed = True
                elif name == "with_negatives" and cv.get("with_negatives"):
                    neg = cv["with_negatives"]
                    q.criteria = [Option(key=o.key, description=(o.description + " " + neg.get(o.key, "")).strip()) for o in q.criteria]  # type: ignore[union-attr]
                    changed = True
                elif name == "vague" and cv.get("vague"):
                    q.criteria = [Option(key=o.key, description="An option.") for o in q.criteria]  # type: ignore[union-attr]
                    changed = True
                if changed:
                    q.framing_id = f"crit_{name}"
            if changed:
                out.append(row)
    return out


def apply_corruption(items: list[TaskItem], seed: int = 0) -> list[TaskItem]:
    """Swap descriptions between two options (control arm). Ground truth unchanged; expect collapse."""
    rng = random.Random(seed)
    out = []
    for it in items:
        row = _clone(it)
        for q in row.questions.values():
            if q.type == "choice" and q.cardinality >= 2:
                i, j = rng.sample(range(q.cardinality), 2)
                a, b = q.criteria[i], q.criteria[j]  # type: ignore[index]
                q.criteria[i], q.criteria[j] = Option(key=a.key, description=b.description), Option(key=b.key, description=a.description)  # type: ignore[index]
                q.framing_id = "crit_swapped"
        out.append(row)
    return out


def permute_options(items: list[TaskItem], n_perms: int = 5, seed: int = 0) -> list[TaskItem]:
    """Emit n_perms shuffled-option copies of each item (permutation_id p1..pn); p0 is canonical."""
    rng = random.Random(seed)
    out = []
    for it in items:
        out.append(it)
        for p in range(1, n_perms + 1):
            row = _clone(it)
            row.permutation_id = f"p{p}"
            for q in row.questions.values():
                if q.type == "choice":
                    opts = list(q.criteria)  # type: ignore[arg-type]
                    rng.shuffle(opts)
                    q.criteria = opts
            out.append(row)
    return out


def strip_options(items: list[TaskItem]) -> list[TaskItem]:
    """Options-only short-circuit: replace the state with an empty placeholder."""
    out = []
    for it in items:
        row = _clone(it)
        row.state = "(no content)"
        row.permutation_id = "options_only"
        out.append(row)
    return out


def strip_state(items: list[TaskItem]) -> list[TaskItem]:
    """State-only short-circuit: blank all option descriptions and instructions."""
    out = []
    for it in items:
        row = _clone(it)
        row.permutation_id = "state_only"
        for q in row.questions.values():
            q.instructions = "Choose."
            if q.type == "choice":
                q.criteria = [Option(key=o.key, description="") for o in q.criteria]  # type: ignore[union-attr]
        out.append(row)
    return out


# ------------------------------------------------------------------ decomposition combiners (Suite B)


def decompose_fixed_rule(sub_probs: dict[str, float], rule: str = "any", threshold: float = 0.5) -> float:
    """Combine sub-question P(yes) values into a holistic P(yes) with no fitted parameters.
    any: noisy-OR; all: product; mean: average."""
    p = np.array(list(sub_probs.values()), float)
    if rule == "any":
        return float(1.0 - np.prod(1.0 - p))
    if rule == "all":
        return float(np.prod(p))
    if rule == "mean":
        return float(p.mean())
    if rule == "vote":
        return float((p >= threshold).mean())
    raise ValueError(rule)


def fit_decomposition_weights(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 500, lr: float = 0.5) -> np.ndarray:
    """Logistic regression on sub-question probabilities (fit on a disjoint half). Returns weights incl. bias."""
    X = np.hstack([np.asarray(X, float), np.ones((len(X), 1))])
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    return w


def apply_decomposition_weights(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.hstack([np.asarray(X, float), np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-(X @ w)))


# ------------------------------------------------------------------ prior shift (robustness)


def resample_prior_shift(items: list[TaskItem], question_key: str, target_label: str, share: float, n: int | None = None,
                         seed: int = 0) -> list[TaskItem]:
    """Resample items (with replacement where needed) so that `target_label` makes up `share` of the ground truth
    for `question_key`. Deploy-time priors never match the benchmark's; a calibrated model's probabilities should
    track the realised base rate, not the training prior."""
    rng = random.Random(seed)
    n = n or len(items)
    pos = [it for it in items if str(it.questions[question_key].ground_truth).lower() == target_label.lower()]
    neg = [it for it in items if str(it.questions[question_key].ground_truth).lower() != target_label.lower()]
    if not pos or not neg:
        return list(items)
    k = int(round(share * n))
    out = [rng.choice(pos) for _ in range(k)] + [rng.choice(neg) for _ in range(n - k)]
    rng.shuffle(out)
    result = []
    for i, it in enumerate(out):
        row = _clone(it)
        row.controls.prior_shift = f"{question_key}:{target_label}@{share}"
        row.metadata = {**row.metadata, "prior_shift_index": i}
        result.append(row)
    return result
