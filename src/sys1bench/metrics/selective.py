"""Selective prediction: risk-coverage curves, AURC, coverage at risk, abstention quality."""

from __future__ import annotations

import numpy as np


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by descending confidence; coverage[i] = (i+1)/N, risk[i] = error rate among the i+1 most confident."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - correct[order]
    n = len(err)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return coverage, risk


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    cov, risk = risk_coverage(conf, correct)
    return float(np.trapezoid(risk, cov)) if len(cov) else float("nan")


def e_aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Excess AURC over the optimal ranking (all errors last)."""
    correct = np.asarray(correct, float)
    n = len(correct)
    if n == 0:
        return float("nan")
    k = int(n - correct.sum())
    opt_risk = np.concatenate([np.zeros(n - k), np.cumsum(np.ones(k)) / np.arange(n - k + 1, n + 1)]) if k else np.zeros(n)
    opt = float(np.trapezoid(opt_risk, np.arange(1, n + 1) / n))
    return aurc(conf, correct) - opt


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    cov, risk = risk_coverage(conf, correct)
    ok = cov[risk <= max_risk]
    return float(ok.max()) if len(ok) else 0.0


def threshold_for_risk(conf: np.ndarray, correct: np.ndarray, max_risk: float) -> float | None:
    conf = np.asarray(conf, float)
    cov, risk = risk_coverage(conf, correct)
    order = np.argsort(-conf, kind="stable")
    idx = np.where(risk <= max_risk)[0]
    return float(conf[order][idx.max()]) if len(idx) else None


def abstention_quality(abstained: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """For native abstain paths: how often the model abstains, and how often an abstention would have been wrong."""
    abstained, correct = np.asarray(abstained, bool), np.asarray(correct, float)
    rate = float(abstained.mean()) if len(abstained) else float("nan")
    prec = float(1.0 - correct[abstained].mean()) if abstained.any() else float("nan")
    acc_answered = float(correct[~abstained].mean()) if (~abstained).any() else float("nan")
    return {"abstain_rate": rate, "abstain_precision": prec, "accuracy_when_answering": acc_answered}


def selective_summary(conf: np.ndarray, correct: np.ndarray, abstained: np.ndarray | None = None) -> dict:
    out = {"aurc": aurc(conf, correct), "e_aurc": e_aurc(conf, correct)}
    for r in (0.01, 0.05, 0.10):
        out[f"coverage_at_risk_{int(r*100)}"] = coverage_at_risk(conf, correct, r)
        thr = threshold_for_risk(conf, correct, r)
        out[f"threshold_at_risk_{int(r*100)}"] = thr if thr is not None else float("nan")
    if abstained is not None:
        out.update(abstention_quality(abstained, correct))
    return out
