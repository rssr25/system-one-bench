"""Suite D runner: distractor resistance, none-of-the-above detection with and without an abstain option,
surface perturbations, and prior shift. Every arm reuses the same seed so items are paired with the clean run."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..adapters.base import BaseAdapter
from ..framing.expand import resample_prior_shift
from ..framing.perturb import PERTURBATIONS, perturb_items
from ..generators import get_generator
from ..metrics.calibration import ece
from ..metrics.consistency import jsd
from ..metrics.robustness import ood_summary
from ..schemas import PredictionRow
from .benchmark_runner import run_items
from .cache import ResponseCache


def _acc(rows: list[PredictionRow]) -> float:
    c = [r.correct for r in rows if r.correct is not None and r.error is None]
    return float(np.mean(c)) if c else float("nan")


def _conf(rows: list[PredictionRow]) -> float:
    c = [max(r.probs) for r in rows if r.error is None and r.probs]
    return float(np.mean(c)) if c else float("nan")


def _paired_jsd(a: list[PredictionRow], b: list[PredictionRow]) -> float:
    A = {(r.task_id, r.question_key): r for r in a if r.error is None}
    vals = [jsd(np.array(A[k].probs), np.array(r.probs)) for r in b if r.error is None and (k := (r.task_id, r.question_key)) in A
            and len(A[k].probs) == len(r.probs)]
    return float(np.mean(vals)) if vals else float("nan")


def robustness_suite(adapter: BaseAdapter, n: int = 300, seed: int = 42, generator: str = "support_tickets",
                     densities: tuple[float, ...] = (0.0, 0.25, 0.5), perturbations: tuple[str, ...] = tuple(PERTURBATIONS),
                     none_frac: float = 0.3, cache: ResponseCache | None = None, concurrency: int = 1) -> dict[str, Any]:
    out: dict[str, Any] = {"generator": generator, "n": n}
    clean_items = get_generator(generator, n=n, seed=seed).generate()
    clean = run_items(clean_items, adapter, cache, suite="D", arm="clean", concurrency=concurrency)
    by_q = sorted({r.question_key for r in clean})
    out["clean"] = {q: {"accuracy": _acc([r for r in clean if r.question_key == q]), "confidence": _conf([r for r in clean if r.question_key == q])} for q in by_q}

    # D2 distractors (regenerate with density; same seed so items pair by task_id)
    out["distractors"] = {}
    for d in densities:
        if d == 0.0:
            continue
        items = get_generator(generator, n=n, seed=seed, distractor_density=d).generate()
        rows = run_items(items, adapter, cache, suite="D", arm=f"distractor={d}", concurrency=concurrency)
        out["distractors"][d] = {q: {"accuracy": _acc([r for r in rows if r.question_key == q]),
                                     "accuracy_delta": _acc([r for r in rows if r.question_key == q]) - out["clean"][q]["accuracy"],
                                     "jsd_vs_clean": _paired_jsd([r for r in clean if r.question_key == q], [r for r in rows if r.question_key == q])}
                                 for q in by_q}

    # D3 none-of-the-above: truth removed from options for a fraction of items
    items_none = get_generator(generator, n=n, seed=seed, none_correct_frac=none_frac).generate()
    choice_q = next((k for k, q in items_none[0].questions.items() if q.type == "choice"), None)
    if choice_q:
        with_abst = run_items(items_none, adapter, cache, suite="D", arm="none_with_abstain", concurrency=concurrency)
        items_no = [it.model_copy(deep=True) for it in items_none]
        for it in items_no:
            it.questions[choice_q].allow_abstain = False
            if it.controls.none_correct:
                it.questions[choice_q].ground_truth = None  # nothing is correct; scored via confidence only
            it.permutation_id = "none_no_abstain"
        without = run_items(items_no, adapter, cache, suite="D", arm="none_without_abstain", concurrency=concurrency)
        ids_none = {it.task_id for it in items_none if it.controls.none_correct}

        def split(rows):
            rs = [r for r in rows if r.question_key == choice_q and r.error is None]
            return [r for r in rs if r.task_id not in ids_none], [r for r in rs if r.task_id in ids_none]

        in_w, ood_w = split(with_abst)
        in_wo, ood_wo = split(without)
        out["none_of_the_above"] = {
            "question": choice_q, "n_none": len(ids_none),
            "with_abstain_option": {"accuracy_in_dist": _acc(in_w), "abstain_rate_on_none": float(np.mean([r.abstained or r.argmax == "none_of_the_above" for r in ood_w])) if ood_w else float("nan"),
                                    "accuracy_on_none_items": _acc(ood_w), "conf_in": _conf(in_w), "conf_none": _conf(ood_w)},
            "without_abstain_option": {"accuracy_in_dist": _acc(in_wo), "conf_in": _conf(in_wo), "conf_none": _conf(ood_wo),
                                       **(ood_summary(np.array([r.probs for r in in_wo]), np.array([r.probs for r in ood_wo])) if in_wo and ood_wo else {})},
        }

    # D4 surface perturbations
    out["perturbations"] = {}
    for kind in perturbations:
        rows = run_items(perturb_items(clean_items, kind, seed), adapter, cache, suite="D", arm=f"perturb={kind}", concurrency=concurrency)
        out["perturbations"][kind] = {q: {"accuracy": _acc([r for r in rows if r.question_key == q]),
                                          "accuracy_delta": _acc([r for r in rows if r.question_key == q]) - out["clean"][q]["accuracy"],
                                          "jsd_vs_clean": _paired_jsd([r for r in clean if r.question_key == q], [r for r in rows if r.question_key == q]),
                                          "error_rate": float(np.mean([r.error is not None for r in rows if r.question_key == q]))}
                                      for q in by_q}

    # D5 prior shift on the first noul question
    noul_q = next((k for k, q in clean_items[0].questions.items() if q.type == "noul"), None)
    if noul_q:
        out["prior_shift"] = {}
        for share in (0.2, 0.5, 0.8):
            shifted = resample_prior_shift(clean_items, noul_q, "true", share, n=n, seed=seed)
            rows = [r for r in run_items(shifted, adapter, cache, suite="D", arm=f"prior={share}", concurrency=concurrency) if r.question_key == noul_q and r.error is None]
            if not rows:
                continue
            p_true = np.array([r.probs[0] for r in rows])
            truth = np.array([r.ground_truth == "true" for r in rows], float)
            out["prior_shift"][share] = {"realised_base_rate": float(truth.mean()), "mean_p_true": float(p_true.mean()),
                                         "accuracy": float(((p_true >= 0.5) == truth.astype(bool)).mean()),
                                         "ece15": ece(np.where(p_true >= 0.5, p_true, 1 - p_true), ((p_true >= 0.5) == truth.astype(bool)).astype(float), 15)}
    return out
