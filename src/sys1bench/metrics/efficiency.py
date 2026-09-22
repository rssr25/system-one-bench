"""Latency, throughput, and cost. Hosted and local models are summarised separately by the report layer."""

from __future__ import annotations

import numpy as np


def latency_summary(ms: np.ndarray) -> dict[str, float]:
    ms = np.asarray(ms, float)
    ms = ms[np.isfinite(ms)]
    if len(ms) == 0:
        return {k: float("nan") for k in ("p50", "p90", "p95", "p99", "mean", "n")}
    return {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
            "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)),
            "mean": float(ms.mean()), "n": float(len(ms))}


def throughput(n_decisions: int, n_questions: int, wall_seconds: float) -> dict[str, float]:
    return {"decisions_per_s": n_decisions / wall_seconds if wall_seconds > 0 else float("nan"),
            "questions_per_s": n_questions / wall_seconds if wall_seconds > 0 else float("nan")}


def cost_summary(cost_per_call: np.ndarray, questions_per_call: np.ndarray | None = None,
                 gpu_hour_usd: float | None = None, wall_seconds: float | None = None) -> dict[str, float]:
    c = np.asarray(cost_per_call, float)
    c = c[np.isfinite(c)]
    out: dict[str, float] = {}
    if len(c):
        out["cost_per_100k_decisions"] = float(c.mean() * 100_000)
        if questions_per_call is not None:
            q = np.asarray(questions_per_call, float)
            out["cost_per_100k_questions"] = float(c.sum() / max(q.sum(), 1) * 100_000)
    if gpu_hour_usd is not None and wall_seconds is not None and len(c):
        out["amortised_gpu_cost_per_100k_decisions"] = float(gpu_hour_usd * wall_seconds / 3600 / len(c) * 100_000)
    return out
