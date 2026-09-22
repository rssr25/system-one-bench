"""Item-level statistics. The item (or framing group) is the unit; seeds are not."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import binomtest


def paired_bootstrap(a: np.ndarray, b: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean,
                     n_boot: int = 10_000, seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """Bootstrap CI for stat(a) - stat(b) over paired items."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    point = float(stat(a) - stat(b))
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_sign_holds": float((np.sign(diffs) == np.sign(point)).mean()) if point != 0 else 0.5,
            "n": int(n)}


def bootstrap_ci(x: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 10_000, seed: int = 0,
                 ci: float = 0.95) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    vals = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(stat(x)), float(lo), float(hi)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    if n01 + n10 == 0:
        return {"n_a_only": n01, "n_b_only": n10, "p": 1.0}
    p = binomtest(n01, n01 + n10, 0.5).pvalue
    return {"n_a_only": n01, "n_b_only": n10, "p": float(p)}


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean,
                      n_boot: int = 5_000, seed: int = 0) -> tuple[float, float, float]:
    """Resample clusters (e.g. task_id when paraphrases are pooled) rather than rows."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [values[clusters == c] for c in uniq]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[i] for i in pick])))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(stat(values)), float(lo), float(hi)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stop = {}, False
    for i, (k, p) in enumerate(items):
        if not stop and p <= alpha / (m - i):
            out[k] = True
        else:
            stop = True
            out[k] = False
    return out


def power_note(n: int) -> str:
    half_width = 1.96 * np.sqrt(0.25 / n) * 100
    return f"n={n}: accuracy 95% CI half-width about ±{half_width:.1f} points"
