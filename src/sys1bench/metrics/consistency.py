"""Distribution-stability kernels: JSD, flip rates, complement consistency, interference matrices."""

from __future__ import annotations

import numpy as np


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p, q = np.asarray(p, float) + eps, np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def pairwise_mean_jsd(dists: list[np.ndarray]) -> float:
    if len(dists) < 2:
        return float("nan")
    vals = [jsd(dists[i], dists[j]) for i in range(len(dists)) for j in range(i + 1, len(dists))]
    return float(np.mean(vals))


def flip_rate(argmaxes: list[str]) -> float:
    """Fraction of variants whose argmax differs from the canonical (first) variant."""
    if len(argmaxes) < 2:
        return float("nan")
    return float(np.mean([a != argmaxes[0] for a in argmaxes[1:]]))


def complement_consistency(p_yes_q: np.ndarray, p_yes_not_q: np.ndarray) -> dict[str, float]:
    dev = np.asarray(p_yes_q, float) + np.asarray(p_yes_not_q, float) - 1.0
    return {"complement_mad": float(np.abs(dev).mean()), "complement_bias": float(dev.mean())}


def framing_summary(acc_by_framing: dict[str, float], per_item_jsd: list[float], per_item_flip: list[float]) -> dict:
    accs = np.array(list(acc_by_framing.values()), float)
    return {
        "n_framings": len(accs),
        "accuracy_median": float(np.median(accs)) if len(accs) else float("nan"),
        "accuracy_min": float(accs.min()) if len(accs) else float("nan"),
        "accuracy_max": float(accs.max()) if len(accs) else float("nan"),
        "accuracy_range": float(accs.max() - accs.min()) if len(accs) else float("nan"),
        "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else float("nan"),
        "mean_pairwise_jsd": float(np.nanmean(per_item_jsd)) if per_item_jsd else float("nan"),
        "argmax_flip_rate": float(np.nanmean(per_item_flip)) if per_item_flip else float("nan"),
    }


def interference_matrix(alone: dict[str, np.ndarray], batched: dict[tuple[str, str], np.ndarray]) -> dict[str, dict[str, float]]:
    """alone[q] = distribution when q asked alone; batched[(q, other)] = distribution of q when asked with `other`.
    Returns M[q][other] = JSD(alone[q], batched[(q, other)])."""
    out: dict[str, dict[str, float]] = {}
    for (q, other), d in batched.items():
        out.setdefault(q, {})[other] = jsd(alone[q], d)
    return out


def positional_bias_chi2(argmax_indices: np.ndarray, k: int) -> dict[str, float]:
    """Chi-square test that argmax index is uniform over positions under random permutations."""
    from scipy.stats import chisquare

    counts = np.bincount(np.asarray(argmax_indices, int), minlength=k)[:k]
    if counts.sum() == 0:
        return {"chi2": float("nan"), "p": float("nan"), "index0_share": float("nan")}
    stat, p = chisquare(counts)
    return {"chi2": float(stat), "p": float(p), "index0_share": float(counts[0] / counts.sum())}
