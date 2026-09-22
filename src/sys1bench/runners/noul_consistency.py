"""Suite G: noul consistency.

G1 complement consistency: P(yes | Q) + P(yes | not-Q) should be 1. Negated wordings come from the framing YAML
   (`negations:` list under the framing group) or a generic "Is it NOT the case that ..." fallback.
G2 choice-vs-noul agreement: the same binary judgment posed as a 2-option choice and as a noul; JSD between them.
G3 threshold portability: fit the accuracy-optimal P(yes) threshold on one generator's noul question and apply it to
   another generator's nominally similar question (or another seed of the same generator); report the accuracy lost
   relative to the locally optimal threshold.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..adapters.base import BaseAdapter
from ..metrics.consistency import complement_consistency, jsd
from ..schemas import Option, PredictionRow, Question, TaskItem
from .benchmark_runner import run_items
from .cache import ResponseCache


def _negate(instr: str) -> str:
    return f"Is it NOT the case that the following holds: {instr.rstrip('?')}?"


def _rows_for(items: list[TaskItem], adapter, cache, arm: str, concurrency: int, key: str) -> dict[str, PredictionRow]:
    return {r.task_id: r for r in run_items(items, adapter, cache, suite="G", arm=arm, concurrency=concurrency) if r.question_key == key and not r.error}


def complement_test(adapter: BaseAdapter, items: list[TaskItem], key: str, negations: list[str] | None = None,
                    cache: ResponseCache | None = None, concurrency: int = 1) -> dict[str, Any]:
    base = []
    for it in items:
        b = copy.deepcopy(it)
        b.questions = {key: it.questions[key]}
        b.permutation_id = "noul_pos"
        base.append(b)
    pos = _rows_for(base, adapter, cache, "complement_pos", concurrency, key)
    variants = negations or [_negate(items[0].questions[key].instructions)]
    out: dict[str, Any] = {"question": key, "n": len(pos), "negations": []}
    for i, neg in enumerate(variants):
        negs = []
        for it in items:
            b = copy.deepcopy(it)
            q = copy.deepcopy(it.questions[key])
            q.instructions = neg
            q.ground_truth = (not q.ground_truth) if isinstance(q.ground_truth, bool) else None
            q.framing_id = f"neg{i}"
            b.questions = {key: q}
            b.permutation_id = f"noul_neg{i}"
            negs.append(b)
        nrows = _rows_for(negs, adapter, cache, f"complement_neg{i}", concurrency, key)
        ids = [t for t in pos if t in nrows]
        p_q = np.array([pos[t].probs[0] for t in ids])
        p_nq = np.array([nrows[t].probs[0] for t in ids])
        cc = complement_consistency(p_q, p_nq)
        acc_neg = float(np.mean([nrows[t].correct for t in ids if nrows[t].correct is not None])) if ids else float("nan")
        out["negations"].append({"wording": neg, "n": len(ids), **cc, "accuracy_negated": acc_neg,
                                 "frac_both_yes_over_0.5": float(np.mean((p_q > 0.5) & (p_nq > 0.5))) if ids else float("nan")})
    out["accuracy_positive"] = float(np.mean([r.correct for r in pos.values() if r.correct is not None])) if pos else float("nan")
    return out


def choice_vs_noul(adapter: BaseAdapter, items: list[TaskItem], key: str, cache: ResponseCache | None = None,
                   concurrency: int = 1) -> dict[str, Any]:
    nouls, choices = [], []
    for it in items:
        a = copy.deepcopy(it)
        a.questions = {key: it.questions[key]}
        a.permutation_id = "as_noul"
        nouls.append(a)
        b = copy.deepcopy(it)
        q = it.questions[key]
        cq = Question(type="choice", instructions=q.instructions,
                      criteria=[Option(key="yes", description="The answer is yes."), Option(key="no", description="The answer is no.")],
                      ground_truth=("yes" if q.ground_truth else "no") if isinstance(q.ground_truth, bool) else None, framing_group=q.framing_group)
        b.questions = {key: cq}
        b.permutation_id = "as_choice"
        choices.append(b)
    n = _rows_for(nouls, adapter, cache, "as_noul", concurrency, key)
    c = _rows_for(choices, adapter, cache, "as_choice", concurrency, key)
    ids = [t for t in n if t in c]
    js = [jsd(np.array(n[t].probs), np.array(c[t].probs)) for t in ids]  # both ordered [yes, no]
    flips = [n[t].argmax != ("true" if c[t].argmax == "yes" else "false") for t in ids]
    return {"question": key, "n": len(ids), "mean_jsd": float(np.mean(js)) if js else float("nan"),
            "argmax_disagreement": float(np.mean(flips)) if flips else float("nan"),
            "accuracy_noul": float(np.mean([n[t].correct for t in ids])) if ids else float("nan"),
            "accuracy_choice": float(np.mean([c[t].correct for t in ids])) if ids else float("nan"),
            "mean_p_yes_noul": float(np.mean([n[t].probs[0] for t in ids])) if ids else float("nan"),
            "mean_p_yes_choice": float(np.mean([c[t].probs[0] for t in ids])) if ids else float("nan")}


def _best_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        acc = float(((p >= t) == y).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t, best_acc


def threshold_portability(rows_a: list[PredictionRow], rows_b: list[PredictionRow]) -> dict[str, Any]:
    """Fit the accuracy-optimal threshold on A, apply on B; compare with B's own optimum and with 0.5."""
    def xy(rows):
        rs = [r for r in rows if not r.error and r.ground_truth in ("true", "false")]
        return np.array([r.probs[0] for r in rs]), np.array([r.ground_truth == "true" for r in rs])

    pa, ya = xy(rows_a)
    pb, yb = xy(rows_b)
    if len(pa) < 20 or len(pb) < 20:
        return {"n_a": len(pa), "n_b": len(pb), "note": "too few rows"}
    ta, acc_a = _best_threshold(pa, ya)
    tb, acc_b = _best_threshold(pb, yb)
    acc_b_with_ta = float(((pb >= ta) == yb).mean())
    acc_b_half = float(((pb >= 0.5) == yb).mean())
    return {"n_a": len(pa), "n_b": len(pb), "threshold_fit_on_a": ta, "accuracy_a_at_own": acc_a,
            "threshold_b_own": tb, "accuracy_b_at_own": acc_b, "accuracy_b_at_a_threshold": acc_b_with_ta,
            "accuracy_b_at_0.5": acc_b_half, "portability_loss": acc_b - acc_b_with_ta}
