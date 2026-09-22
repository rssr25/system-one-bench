"""OOD / negative detection and perturbation deltas."""

from __future__ import annotations

import numpy as np


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probs, float), eps, 1.0)
    return -(p * np.log(p)).sum(-1)


def normalised_entropy(probs: np.ndarray) -> np.ndarray:
    k = np.asarray(probs).shape[-1]
    return entropy(probs) / np.log(k) if k > 1 else np.zeros(len(probs))


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUROC of `scores` separating positives (in-distribution) from negatives (OOD), via rank statistic."""
    from scipy.stats import rankdata

    pos, neg = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ood_summary(probs_in: np.ndarray, probs_ood: np.ndarray) -> dict[str, float]:
    conf_in, conf_ood = np.asarray(probs_in).max(1), np.asarray(probs_ood).max(1)
    h_in, h_ood = normalised_entropy(probs_in), normalised_entropy(probs_ood)
    z = (h_ood.mean() - h_in.mean()) / (h_in.std(ddof=1) + 1e-12) if len(h_in) > 1 else float("nan")
    return {
        "auroc_confidence": auroc(conf_in, conf_ood),
        "auroc_entropy": auroc(h_ood, h_in),
        "mean_conf_in": float(conf_in.mean()),
        "mean_conf_ood": float(conf_ood.mean()),
        "frac_ood_conf_over_0.7": float((conf_ood > 0.7).mean()),
        "entropy_shift_z": float(z),
        "mean_norm_entropy_ood": float(h_ood.mean()),
    }


def perturbation_delta(acc_clean: float, acc_pert: float, jsd_mean: float) -> dict[str, float]:
    return {"accuracy_delta": float(acc_pert - acc_clean), "mean_jsd_to_clean": float(jsd_mean)}
