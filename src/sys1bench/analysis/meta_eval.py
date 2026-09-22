"""Benchmark audits: short-circuit, leakage, label-order bias, label-noise control."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..metrics.consistency import positional_bias_chi2
from ..schemas import PredictionRow, TaskItem


def _acc(rows: list[PredictionRow]) -> float:
    c = [r.correct for r in rows if r.correct is not None]
    return float(np.mean(c)) if c else float("nan")


def short_circuit_report(full: list[PredictionRow], state_only: list[PredictionRow], options_only: list[PredictionRow],
                         prior_acc: float, margin: float = 0.10) -> dict:
    a_full, a_state, a_opts = _acc(full), _acc(state_only), _acc(options_only)
    return {"accuracy_full": a_full, "accuracy_state_only": a_state, "accuracy_options_only": a_opts,
            "accuracy_prior": prior_acc,
            "flag_state_only": bool(a_state > prior_acc + margin), "flag_options_only": bool(a_opts > prior_acc + margin)}


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def leakage_report(items: list[TaskItem], n: int = 1) -> dict:
    """Fraction of items where the true option's key or description n-grams appear in the state."""
    hits, total, per_q = 0, 0, Counter()
    for it in items:
        state = it.state_text()
        sg = _ngrams(state, n)
        for k, q in it.questions.items():
            if q.type != "choice" or q.ground_truth is None:
                continue
            total += 1
            opt = next((o for o in q.criteria if o.key == q.ground_truth), None)  # type: ignore[union-attr]
            if opt is None:
                continue
            key_words = set(re.findall(r"[a-z0-9]+", opt.key.lower()))
            leak = bool(sg & {(w,) for w in key_words}) if n == 1 else bool(sg & _ngrams(opt.description, n))
            if leak:
                hits += 1
                per_q[k] += 1
    return {"n_choice_questions": total, "leak_rate": hits / total if total else float("nan"), "per_question": dict(per_q)}


def label_order_report(rows: list[PredictionRow]) -> dict:
    """Over permuted rows, test whether argmax index prefers a position."""
    by_k: dict[int, list[int]] = {}
    for r in rows:
        if r.primitive != "choice" or r.error or not r.argmax:
            continue
        by_k.setdefault(r.cardinality, []).append(r.option_keys.index(r.argmax))
    return {k: positional_bias_chi2(np.array(v), k) for k, v in by_k.items()}


def label_noise_control(rows_clean: list[PredictionRow], rows_noisy: list[PredictionRow]) -> dict:
    """On corrupted labels a model should be ~wrong at the corruption rate and NOT more confident."""
    conf = lambda rs: float(np.mean([max(r.probs) for r in rs if not r.error])) if rs else float("nan")
    return {"accuracy_clean": _acc(rows_clean), "accuracy_noisy_labels": _acc(rows_noisy),
            "confidence_clean": conf(rows_clean), "confidence_noisy_labels": conf(rows_noisy)}
