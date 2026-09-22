"""Calibration kernels. All functions take confidence/probability arrays and 0/1 correctness or one-hot labels.

Conventions: `probs` is (N, K); `y` is integer labels (N,); `conf` is max-probability (N,); `correct` is 0/1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _bins(conf: np.ndarray, m: int, scheme: str) -> np.ndarray:
    if scheme == "width":
        edges = np.linspace(0.0, 1.0, m + 1)
    elif scheme == "mass":
        edges = np.quantile(conf, np.linspace(0, 1, m + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        raise ValueError(scheme)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    return idx


def ece(conf: np.ndarray, correct: np.ndarray, m: int = 15, scheme: str = "width") -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    idx = _bins(conf, m, scheme)
    total = 0.0
    for b in np.unique(idx):
        mask = idx == b
        total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def mce(conf: np.ndarray, correct: np.ndarray, m: int = 15, scheme: str = "width") -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = _bins(conf, m, scheme)
    gaps = [abs(conf[idx == b].mean() - correct[idx == b].mean()) for b in np.unique(idx)]
    return float(max(gaps)) if gaps else float("nan")


def ece_noise_floor(conf: np.ndarray, m: int = 15, scheme: str = "width", resamples: int = 200,
                    seed: int = 0) -> tuple[float, float]:
    """Expected ECE (mean, sd) of a *perfectly calibrated* model with this confidence histogram:
    resample correctness ~ Bernoulli(conf). Report measured ECE / floor, not ECE alone."""
    rng = np.random.default_rng(seed)
    conf = np.asarray(conf, float)
    vals = [ece(conf, rng.random(len(conf)) < conf, m, scheme) for _ in range(resamples)]
    return float(np.mean(vals)), float(np.std(vals))


def smooth_ece(conf: np.ndarray, correct: np.ndarray, bandwidth: float = 0.05) -> float:
    """Binning-free calibration error: kernel-smoothed |E[correct|conf] - conf| integrated over the
    empirical confidence distribution (Gaussian kernel)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    d = (conf[:, None] - conf[None, :]) / bandwidth
    w = np.exp(-0.5 * d**2)
    smoothed_acc = (w * correct[None, :]).sum(1) / w.sum(1)
    return float(np.mean(np.abs(smoothed_acc - conf)))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    probs = np.asarray(probs, float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), np.asarray(y, int)] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def brier_decomposition(conf: np.ndarray, correct: np.ndarray, m: int = 15) -> dict[str, float]:
    """Murphy decomposition on the top-label (confidence vs correctness) problem:
    BS = reliability - resolution + uncertainty."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = _bins(conf, m, "width")
    base = correct.mean()
    rel = res = 0.0
    for b in np.unique(idx):
        mask = idx == b
        rel += mask.mean() * (conf[mask].mean() - correct[mask].mean()) ** 2
        res += mask.mean() * (correct[mask].mean() - base) ** 2
    unc = base * (1 - base)
    return {"reliability": float(rel), "resolution": float(res), "uncertainty": float(unc),
            "brier_top": float(((conf - correct) ** 2).mean())}


def nll(probs: np.ndarray, y: np.ndarray, clip: float | None = 1e-4) -> float:
    p = np.asarray(probs, float)[np.arange(len(y)), np.asarray(y, int)]
    if clip is not None:
        p = np.clip(p, clip, 1.0)
    with np.errstate(divide="ignore"):
        return float(-np.log(p).mean())


def zero_prob_on_truth(probs: np.ndarray, y: np.ndarray) -> int:
    p = np.asarray(probs, float)[np.arange(len(y)), np.asarray(y, int)]
    return int((p <= 0.0).sum())


def quantisation_report(probs: np.ndarray) -> dict[str, float | None]:
    flat = np.asarray(probs, float).ravel()
    flat = flat[np.isfinite(flat)]
    step = None
    for s in (0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5):
        if np.allclose(np.round(flat / s) * s, flat, atol=s * 1e-3):
            step = s
            break
    return {"step": step, "frac_exact_zero": float((flat == 0).mean()), "frac_exact_one": float((flat == 1).mean()),
            "distinct_values": float(len(np.unique(np.round(flat, 6))))}


def fit_temperature(probs: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Single temperature minimising NLL on held-out data. T>1 means the model was over-confident."""
    probs = np.clip(np.asarray(probs, float), eps, 1.0)
    logits = np.log(probs)
    y = np.asarray(y, int)

    def loss(t: float) -> float:
        z = logits / t
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    res = minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(probs: np.ndarray, t: float, eps: float = 1e-6) -> np.ndarray:
    z = np.log(np.clip(np.asarray(probs, float), eps, 1.0)) / t
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def calibration_summary(probs: np.ndarray, y: np.ndarray, m: int = 15, floor_resamples: int = 200) -> dict:
    probs, y = np.asarray(probs, float), np.asarray(y, int)
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    floor_mean, floor_sd = ece_noise_floor(conf, m, "width", floor_resamples)
    e_w = ece(conf, correct, m, "width")
    out = {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "ece_width_15": e_w,
        "ece_mass_15": ece(conf, correct, m, "mass"),
        "ece_width_10": ece(conf, correct, 10, "width"),
        "ece_width_25": ece(conf, correct, 25, "width"),
        "ece_floor_mean": floor_mean,
        "ece_floor_sd": floor_sd,
        "ece_over_floor": (e_w / floor_mean) if floor_mean > 0 else float("nan"),
        "mce_15": mce(conf, correct, m),
        "smooth_ece": smooth_ece(conf, correct),
        "brier": brier(probs, y),
        "nll_clipped": nll(probs, y, 1e-4),
        "nll_raw": nll(probs, y, None),
        "zero_prob_on_truth": zero_prob_on_truth(probs, y),
        "overconfidence": float((conf - correct).mean()),
    }
    out.update(brier_decomposition(conf, correct, m))
    out.update({f"quant_{k}": v for k, v in quantisation_report(probs).items()})
    return out
