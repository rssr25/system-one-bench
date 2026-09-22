"""Scoring for the control arms: unknowable (calibration ceiling), label-noise, corruption, short-circuit."""

from __future__ import annotations

import numpy as np

from ..schemas import PredictionRow, TaskItem


def unknowable_report(rows: list[PredictionRow], question_key: str = "priority") -> dict:
    """On items whose label depends on removed information, a calibrated model should spread mass. For the ticket
    priority arm the tier is removed, so the truth is one of two adjacent levels: report mass on the two candidates,
    max-probability, and how often confidence exceeds 0.7."""
    rs = [r for r in rows if r.question_key == question_key and not r.error and r.controls.unknowable]
    if not rs:
        return {"n": 0}
    conf = np.array([max(r.probs) for r in rs])
    return {"n": len(rs), "mean_max_prob": float(conf.mean()), "frac_conf_over_0.7": float((conf > 0.7).mean()),
            "frac_conf_over_0.9": float((conf > 0.9).mean()),
            "accuracy_vs_hidden_truth": float(np.mean([r.correct for r in rs if r.correct is not None])),
            "mean_entropy_bits": float(np.mean([-(np.array(r.probs) + 1e-12).dot(np.log2(np.array(r.probs) + 1e-12)) for r in rs]))}


def label_noise_report(rows: list[PredictionRow], items: list["TaskItem"]) -> dict:
    """Manifest with a corrupted-label fraction: accuracy on corrupted rows should be near zero (the label is wrong, the
    model should still answer the true thing) and confidence on them should not exceed confidence on clean rows.
    Joined on task_id against the manifest so it works for rows produced by any runner version."""
    noisy_ids = {it.task_id for it in items if it.label_provenance.noise_injected}
    clean = [r for r in rows if not r.error and r.task_id not in noisy_ids]
    noisy = [r for r in rows if not r.error and r.task_id in noisy_ids]

    def f(rs):
        acc = float(np.mean([r.correct for r in rs if r.correct is not None])) if rs else float("nan")
        conf = float(np.mean([max(r.probs) for r in rs])) if rs else float("nan")
        return acc, conf

    ac, cc = f(clean)
    an, cn = f(noisy)
    return {"n_clean": len(clean), "n_noisy": len(noisy), "accuracy_clean": ac, "accuracy_on_corrupted_labels": an,
            "confidence_clean": cc, "confidence_on_corrupted": cn,
            "flag_more_confident_on_corrupted": bool(cn > cc + 0.02) if noisy else False}


def arm_accuracy(rows: list[PredictionRow], arm: str, question_key: str | None = None) -> float:
    rs = [r for r in rows if r.arm == arm and r.correct is not None and (question_key is None or r.question_key == question_key)]
    return float(np.mean([r.correct for r in rs])) if rs else float("nan")
