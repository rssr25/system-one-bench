"""Turn PredictionRows into per-(model, primitive, arm) scorecards. Every accuracy is reported as
median [min, max] over framings; every ECE next to its noise floor; hosted and local separated."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..metrics import calibration, consistency, efficiency, ordinal, selective
from ..schemas import PredictionRow


def group_rows(rows: Iterable[PredictionRow], *keys: str) -> dict[tuple, list[PredictionRow]]:
    out: dict[tuple, list[PredictionRow]] = defaultdict(list)
    for r in rows:
        out[tuple(getattr(r, k) if hasattr(r, k) else r.metadata.get(k) for k in keys)].append(r)
    return dict(out)


def _dedupe(rows: Iterable[PredictionRow]) -> list[PredictionRow]:
    """Framing expansion varies one question at a time, so a noul/score question is repeated verbatim in the rows
    created for a sibling choice question's criteria variants. Keep one row per (task, question, framing, permutation, arm)."""
    seen: set[tuple] = set()
    out = []
    for r in rows:
        k = (r.task_id, r.question_key, r.framing_id, r.permutation_id, r.arm)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _matrix(rows: list[PredictionRow]) -> tuple[np.ndarray, np.ndarray, list[PredictionRow]]:
    ok = [r for r in rows if r.error is None and r.ground_truth is not None and r.ground_truth in r.option_keys]
    if not ok:
        return np.zeros((0, 1)), np.zeros(0, int), ok
    k = ok[0].cardinality
    ok = [r for r in ok if r.cardinality == k]
    P = np.array([r.probs for r in ok], float)
    y = np.array([r.option_keys.index(r.ground_truth) for r in ok], int)
    return P, y, ok


def scorecard(rows: list[PredictionRow], floor_resamples: int = 200) -> dict[str, Any]:
    """Metrics for a homogeneous set of rows (same model, primitive, question). Framing variance is
    computed across `framing_id`; the headline accuracy is the median over framings."""
    rows = _dedupe(rows)
    n_all = len(rows)
    if n_all == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n_rows": n_all,
        "transport_failure_rate": float(np.mean([bool(r.error and r.error.startswith("transport")) for r in rows])),
        "schema_failure_rate": float(np.mean([bool(r.error and not r.error.startswith("transport")) for r in rows])),
        "truncation_rate": float(np.mean([r.truncated for r in rows])),
        "renormalised_rate": float(np.mean([r.renormalised for r in rows])),
        "abstain_rate": float(np.mean([r.abstained for r in rows])),
        "capability_issue_rate": float(np.mean([bool(r.metadata.get("capability_issues")) for r in rows])),
    }
    # accuracy per framing (canonical permutation only, to keep permutation as its own factor)
    by_framing = group_rows([r for r in rows if r.permutation_id == "p0"], "framing_id")
    acc_by_f = {}
    for f, rs in by_framing.items():
        c = [r.correct for r in rs if r.correct is not None]
        if c:
            acc_by_f[f[0]] = float(np.mean(c))
    # per-item JSD / flip across framings
    by_task = group_rows([r for r in rows if r.permutation_id == "p0" and not r.error], "task_id")
    jsds, flips = [], []
    for _, rs in by_task.items():
        if len(rs) > 1 and len({r.cardinality for r in rs}) == 1:
            rs = sorted(rs, key=lambda r: (r.framing_id != "f0", r.framing_id))
            jsds.append(consistency.pairwise_mean_jsd([np.array(r.probs) for r in rs]))
            flips.append(consistency.flip_rate([r.argmax for r in rs]))
    out["framing"] = consistency.framing_summary(acc_by_f, jsds, flips)
    out["accuracy"] = out["framing"]["accuracy_median"]
    out["accuracy_range"] = [out["framing"]["accuracy_min"], out["framing"]["accuracy_max"]]

    canon = [r for r in rows if r.permutation_id == "p0" and r.framing_id == "f0"]
    P, y, ok = _matrix(canon)
    if len(y) >= 10:
        out["calibration"] = calibration.calibration_summary(P, y, floor_resamples=floor_resamples)
        conf = P.max(1)
        correct = (P.argmax(1) == y).astype(float)
        out["selective"] = selective.selective_summary(conf, correct, np.array([r.abstained for r in ok]))
        if ok[0].primitive == "score":
            levels = np.array([int(k) for k in ok[0].option_keys])
            true_levels = np.array([int(r.ground_truth) for r in ok])
            out["ordinal"] = ordinal.ordinal_summary(P, true_levels, levels)
        if len(y) >= 100:
            half = len(y) // 2
            fit_acc = float((P[:half].argmax(1) == y[:half]).mean())
            t = calibration.fit_temperature(P[:half], y[:half])
            P2 = calibration.apply_temperature(P[half:], t)
            if fit_acc >= 0.999:
                direction = "degenerate (all correct on fit half; any T<1 lowers NLL)"
            else:
                direction = "overconfident" if t > 1.05 else ("underconfident" if t < 0.95 else "calibrated")
            out["temperature"] = {"T": t, "direction": direction, "fit_half_accuracy": fit_acc,
                                  "ece_after_refit": calibration.ece(P2.max(1), (P2.argmax(1) == y[half:]).astype(float)),
                                  "ece_before_same_half": calibration.ece(P[half:].max(1), (P[half:].argmax(1) == y[half:]).astype(float))}
    # permutation invariance
    by_task_perm = group_rows([r for r in rows if r.framing_id == "f0" and not r.error and r.primitive == "choice"], "task_id")
    pj, pf = [], []
    for _, rs in by_task_perm.items():
        if len(rs) > 1:
            rs = sorted(rs, key=lambda r: r.permutation_id)
            base = rs[0]
            aligned = []
            for r in rs:
                m = dict(zip(r.option_keys, r.probs))
                aligned.append(np.array([m.get(k, 0.0) for k in base.option_keys]))
            pj.append(consistency.pairwise_mean_jsd(aligned))
            pf.append(consistency.flip_rate([r.argmax for r in rs]))
    if pj:
        out["permutation"] = {"mean_jsd": float(np.mean(pj)), "argmax_flip_rate": float(np.mean(pf)), "n_items": len(pj)}
    out["latency"] = efficiency.latency_summary(np.array([r.latency_ms for r in rows]))
    costs = np.array([r.cost_usd for r in rows if r.cost_usd is not None], float)
    if len(costs):
        out["cost"] = efficiency.cost_summary(costs)
    out["quantisation_steps"] = sorted({r.quantisation_step for r in rows if r.quantisation_step is not None})
    return out


def framing_scorecard(rows: list[PredictionRow]) -> dict[str, Any]:
    """Corruption and criteria-variant deltas relative to canonical framing."""
    acc = {}
    for (f,), rs in group_rows([r for r in _dedupe(rows) if r.permutation_id == "p0"], "framing_id").items():
        c = [r.correct for r in rs if r.correct is not None]
        if c:
            acc[f] = float(np.mean(c))
    base = acc.get("f0", float("nan"))
    return {"accuracy_by_framing": acc,
            "corruption_drop": base - acc["crit_swapped"] if "crit_swapped" in acc else None,
            "label_only_delta": acc["crit_label_only"] - base if "crit_label_only" in acc else None,
            "negatives_delta": acc["crit_with_negatives"] - base if "crit_with_negatives" in acc else None,
            "vague_delta": acc["crit_vague"] - base if "crit_vague" in acc else None}


def markdown_table(cards: dict[str, dict[str, Any]], columns: list[tuple[str, str]] | None = None) -> str:
    """cards: name -> scorecard. columns: (header, dotted.path)."""
    columns = columns or [("acc (median)", "accuracy"), ("acc range", "accuracy_range"), ("ECE/floor", "calibration.ece_over_floor"),
                          ("ECE15", "calibration.ece_width_15"), ("Brier", "calibration.brier"), ("AURC", "selective.aurc"),
                          ("cov@5%", "selective.coverage_at_risk_5"), ("T", "temperature.T"), ("perm JSD", "permutation.mean_jsd"),
                          ("p50 ms", "latency.p50"), ("p95 ms", "latency.p95"), ("schema fail", "schema_failure_rate")]

    def get(d: dict, path: str):
        cur: Any = d
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    def fmt(v):
        if v is None:
            return "–"
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return str(v)

    head = "| model | " + " | ".join(h for h, _ in columns) + " |\n|" + "---|" * (len(columns) + 1) + "\n"
    body = "".join(f"| {name} | " + " | ".join(fmt(get(card, p)) for _, p in columns) + " |\n" for name, card in cards.items())
    return head + body
