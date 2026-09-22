"""Downstream decision value: expected cost of acting on the model's answers under a cost matrix,
with and without an abstain/escalate option. This is the metric calibration is supposed to buy."""

from __future__ import annotations

import numpy as np


def expected_cost(probs: np.ndarray, truth_idx: np.ndarray, cost: np.ndarray, policy: str = "argmax",
                  escalate_cost: float | None = None, threshold: float | None = None) -> dict[str, float]:
    """cost[t, p] = cost of predicting p when truth is t. Policies:
    - argmax: act on argmax.
    - bayes: choose p minimising sum_t probs[t] * cost[t, p] (uses the probabilities).
    - threshold: argmax if max prob >= threshold else escalate (pays escalate_cost, assumed correct).
    """
    probs, truth_idx, cost = np.asarray(probs, float), np.asarray(truth_idx, int), np.asarray(cost, float)
    n = len(truth_idx)
    if policy == "argmax":
        pred = probs.argmax(1)
        realised = cost[truth_idx, pred]
        esc = np.zeros(n, bool)
    elif policy == "bayes":
        pred = (probs @ cost).argmin(1)
        realised = cost[truth_idx, pred]
        esc = np.zeros(n, bool)
    elif policy == "threshold":
        assert threshold is not None and escalate_cost is not None
        pred = probs.argmax(1)
        esc = probs.max(1) < threshold
        realised = np.where(esc, escalate_cost, cost[truth_idx, pred])
    else:
        raise ValueError(policy)
    oracle = cost[truth_idx, truth_idx]
    return {"mean_cost": float(realised.mean()), "regret_vs_oracle": float((realised - oracle).mean()),
            "escalation_rate": float(esc.mean()), "cost_per_10k": float(realised.mean() * 10_000)}


def value_of_calibration(probs: np.ndarray, truth_idx: np.ndarray, cost: np.ndarray) -> float:
    """Cost saved by acting on probabilities (Bayes) rather than argmax. Zero for symmetric costs;
    positive when calibrated probabilities carry usable information about asymmetric risk."""
    a = expected_cost(probs, truth_idx, cost, "argmax")["mean_cost"]
    b = expected_cost(probs, truth_idx, cost, "bayes")["mean_cost"]
    return float(a - b)
