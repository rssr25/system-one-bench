"""Suite B decomposition analysis: holistic question vs sub-questions combined by a fixed rule vs fitted weights.

Rows must contain, per task_id, one holistic noul row (question_key == target) and one row per sub-question
(metadata.decomposition_of == target). Weights are fitted on the first half of task_ids and evaluated on the second.
"""

from __future__ import annotations

import numpy as np

from ..framing.expand import apply_decomposition_weights, decompose_fixed_rule, fit_decomposition_weights
from ..metrics.calibration import brier, ece
from ..schemas import PredictionRow


def decomposition_report(rows: list[PredictionRow], target: str, rule: str = "any") -> dict:
    hol = {r.task_id: r for r in rows if r.question_key == target and not r.error and r.framing_id == "f0" and r.permutation_id == "p0"}
    subs: dict[str, dict[str, PredictionRow]] = {}
    for r in rows:
        if r.metadata.get("decomposition_of") == target and not r.error and r.framing_id == "f0" and r.permutation_id == "p0":
            subs.setdefault(r.task_id, {})[r.question_key] = r
    tids = sorted(t for t in hol if t in subs)
    if len(tids) < 20:
        return {"n": len(tids), "note": "too few paired items"}
    sub_keys = sorted({k for t in tids for k in subs[t]})
    X = np.array([[subs[t][k].probs[0] if k in subs[t] else 0.5 for k in sub_keys] for t in tids])
    y = np.array([1.0 if hol[t].ground_truth == "true" else 0.0 for t in tids])
    p_hol = np.array([hol[t].probs[0] for t in tids])
    p_rule = np.array([decompose_fixed_rule(dict(zip(sub_keys, x)), rule) for x in X])
    half = len(tids) // 2
    w = fit_decomposition_weights(X[:half], y[:half])
    p_fit = apply_decomposition_weights(X[half:], w)

    def summarise(p: np.ndarray, yy: np.ndarray) -> dict:
        pred = p >= 0.5
        P2 = np.stack([p, 1 - p], 1)
        yi = (1 - yy).astype(int)  # index 0 == "true"
        return {"accuracy": float((pred == yy.astype(bool)).mean()), "brier": brier(P2, yi),
                "ece15": ece(P2.max(1), (P2.argmax(1) == yi).astype(float))}

    out = {"n": len(tids), "sub_questions": sub_keys, "rule": rule,
           "holistic": summarise(p_hol, y), "fixed_rule": summarise(p_rule, y),
           "holistic_eval_half": summarise(p_hol[half:], y[half:]), "fitted_weights_eval_half": summarise(p_fit, y[half:]),
           "weights": dict(zip(sub_keys + ["bias"], [float(v) for v in w]))}
    out["decomposition_gain_fixed"] = out["fixed_rule"]["accuracy"] - out["holistic"]["accuracy"]
    out["decomposition_gain_fitted"] = out["fitted_weights_eval_half"]["accuracy"] - out["holistic_eval_half"]["accuracy"]
    return out
