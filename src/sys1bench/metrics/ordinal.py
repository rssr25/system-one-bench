"""Ordinal metrics for the `score` primitive. Levels are integers; `probs` is (N, L) over levels."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def expected_level(probs: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.asarray(probs, float) @ np.asarray(levels, float)


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(np.asarray(pred, float) - np.asarray(true, float)).mean())


def off_by_one_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((np.abs(np.asarray(pred, float) - np.asarray(true, float)) <= 1).mean())


def quadratic_weighted_kappa(pred: np.ndarray, true: np.ndarray, levels: np.ndarray) -> float:
    levels = list(np.asarray(levels))
    k = len(levels)
    idx = {lv: i for i, lv in enumerate(levels)}
    o = np.zeros((k, k))
    for p, t in zip(pred, true):
        o[idx[int(t)], idx[int(p)]] += 1
    n = o.sum()
    if n == 0:
        return float("nan")
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    e = np.outer(o.sum(1), o.sum(0)) / n
    denom = (w * e).sum()
    return float(1.0 - (w * o).sum() / denom) if denom > 0 else float("nan")


def ranked_probability_score(probs: np.ndarray, true_idx: np.ndarray) -> float:
    """Proper scoring rule for ordinal forecasts: mean squared difference of cumulative distributions,
    normalised by L-1 so it lies in [0, 1]."""
    probs = np.asarray(probs, float)
    n, L = probs.shape
    cdf = np.cumsum(probs, 1)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), np.asarray(true_idx, int)] = 1.0
    ocdf = np.cumsum(onehot, 1)
    return float(((cdf - ocdf) ** 2).sum(1).mean() / (L - 1))


def monotonicity_rate(ladders: list[np.ndarray]) -> float:
    """Each ladder is the expected level along a sequence of stimuli with increasing true severity.
    Returns the fraction of ladders that are monotone non-decreasing."""
    if not ladders:
        return float("nan")
    return float(np.mean([bool(np.all(np.diff(np.asarray(l, float)) >= -1e-9)) for l in ladders]))


def ordinal_summary(probs: np.ndarray, true_levels: np.ndarray, levels: np.ndarray) -> dict:
    probs, levels = np.asarray(probs, float), np.asarray(levels)
    true_levels = np.asarray(true_levels)
    idx = {lv: i for i, lv in enumerate(levels)}
    true_idx = np.array([idx[int(t)] for t in true_levels])
    argmax_lv = levels[probs.argmax(1)]
    exp_lv = expected_level(probs, levels)
    exp_round = levels[np.abs(exp_lv[:, None] - levels[None, :]).argmin(1)]
    if len(set(true_levels.tolist())) > 1 and np.ptp(exp_lv) > 1e-12:
        rho = spearmanr(exp_lv, true_levels).statistic
    else:
        rho = float("nan")
    return {
        "n": len(true_levels),
        "exact_accuracy_argmax": float((argmax_lv == true_levels).mean()),
        "exact_accuracy_expected": float((exp_round == true_levels).mean()),
        "off_by_one_accuracy": off_by_one_accuracy(argmax_lv, true_levels),
        "mae_argmax": mae(argmax_lv, true_levels),
        "mae_expected": mae(exp_lv, true_levels),
        "qwk_argmax": quadratic_weighted_kappa(argmax_lv, true_levels, levels),
        "rps": ranked_probability_score(probs, true_idx),
        "spearman_expected": float(rho) if rho == rho else float("nan"),
        "argmax_vs_expected_disagreement": float((argmax_lv != exp_round).mean()),
    }
