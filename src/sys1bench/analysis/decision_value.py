"""Suite I: downstream decision value from the cost matrices shipped in every Tier G manifest."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..metrics.decision_value import expected_cost, value_of_calibration
from ..schemas import PredictionRow, TaskItem


def decision_value_report(rows: list[PredictionRow], items: list[TaskItem], escalate_cost: float | None = None,
                          thresholds: tuple[float, ...] = (0.5, 0.7, 0.9)) -> dict[str, Any]:
    """Per question with a cost matrix: mean realised cost per 10k decisions under argmax, Bayes (uses probabilities)
    and act-or-escalate at each threshold, plus regret vs oracle and the value of calibration (argmax minus Bayes)."""
    by_task = {it.task_id: it for it in items}
    out: dict[str, Any] = {}
    keys = sorted({r.question_key for r in rows})
    for k in keys:
        cm = next((it.cost_matrix.get(k) for it in items if it.cost_matrix and k in it.cost_matrix), None)
        if not cm:
            continue
        rs = [r for r in rows if r.question_key == k and not r.error and r.ground_truth in r.option_keys and r.task_id in by_task
              and r.permutation_id == "p0" and r.framing_id == "f0"]
        if len(rs) < 20:
            continue
        labels = rs[0].option_keys
        C = np.array([[float(cm.get(str(t), {}).get(str(p), 1.0 if t != p else 0.0)) for p in labels] for t in labels])
        P = np.array([r.probs for r in rs])
        y = np.array([labels.index(r.ground_truth) for r in rs])
        esc = escalate_cost if escalate_cost is not None else float(np.median(C[C > 0])) if (C > 0).any() else 1.0
        res = {"n": len(rs), "labels": labels, "escalate_cost": esc,
               "argmax": expected_cost(P, y, C, "argmax"), "bayes": expected_cost(P, y, C, "bayes"),
               "value_of_calibration_per_10k": value_of_calibration(P, y, C) * 10_000}
        for t in thresholds:
            res[f"escalate@{t}"] = expected_cost(P, y, C, "threshold", escalate_cost=esc, threshold=t)
        # trivial policies for reference
        prior = np.bincount(y, minlength=len(labels)) / len(y)
        Pprior = np.tile(prior, (len(y), 1))
        res["majority_prior_bayes"] = expected_cost(Pprior, y, C, "bayes")
        out[k] = res
    return out
