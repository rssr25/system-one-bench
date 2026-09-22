"""Hybrid router sweep: primary System One model, escalate to a fallback below a confidence threshold. Reports
accuracy, latency, cost and escalation rate as a function of the threshold, from cached primary and fallback runs
(each model is called once per item; the routing is simulated offline for every threshold)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schemas import PredictionRow


def hybrid_curve(primary: list[PredictionRow], fallback: list[PredictionRow], thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
                 primary_latency_ms: float | None = None, fallback_latency_ms: float | None = None) -> dict[str, Any]:
    P = {(r.task_id, r.question_key): r for r in primary if not r.error and r.correct is not None}
    F = {(r.task_id, r.question_key): r for r in fallback if not r.error and r.correct is not None}
    keys = sorted(set(P) & set(F))
    if not keys:
        return {"n": 0}
    conf = np.array([max(P[k].probs) for k in keys])
    cp = np.array([P[k].correct for k in keys], float)
    cf = np.array([F[k].correct for k in keys], float)
    lp = np.array([P[k].latency_ms if primary_latency_ms is None else primary_latency_ms for k in keys], float)
    lf = np.array([F[k].latency_ms if fallback_latency_ms is None else fallback_latency_ms for k in keys], float)
    cost_p = np.array([P[k].cost_usd or 0.0 for k in keys])
    cost_f = np.array([F[k].cost_usd or 0.0 for k in keys])
    out: dict[str, Any] = {"n": len(keys), "primary_only": {"accuracy": float(cp.mean()), "p50_ms": float(np.nanmedian(lp)), "cost_per_100k": float(cost_p.mean() * 1e5)},
                           "fallback_only": {"accuracy": float(cf.mean()), "p50_ms": float(np.nanmedian(lf)), "cost_per_100k": float(cost_f.mean() * 1e5)}, "curve": {}}
    for t in thresholds:
        esc = conf < t
        acc = np.where(esc, cf, cp)
        lat = np.where(esc, lp + lf, lp)
        cost = np.where(esc, cost_p + cost_f, cost_p)
        out["curve"][t] = {"escalation_rate": float(esc.mean()), "accuracy": float(acc.mean()), "p50_ms": float(np.nanmedian(lat)),
                           "mean_ms": float(np.nanmean(lat)), "cost_per_100k": float(cost.mean() * 1e5)}
    return out
