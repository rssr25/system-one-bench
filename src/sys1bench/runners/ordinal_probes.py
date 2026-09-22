"""Suite F stimulus probes for the `score` primitive.

Monotonicity ladders: sequences of states that differ only in one controlled severity variable; the expected level
should be non-decreasing along the ladder. Scale invariance: the same states scored on 3-, 5- and 10-level rubrics
with aligned ground truth; the relative ordering (Spearman) and the mapped level should be preserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ..adapters.base import BaseAdapter
from ..metrics.ordinal import monotonicity_rate
from ..schemas import Controls, LabelProvenance, Level, Question, TaskItem
from .benchmark_runner import run_items
from .cache import ResponseCache

# (severity 0..4, text)  ladders of customer-impact statements with one variable: how broken things are
LADDERS = [
    ["The app shows a small typo in the settings menu.", "One report page loads slowly, about ten seconds.", "Exports fail for reports longer than 30 days; other features work.",
     "I cannot log in on any device since this morning.", "Our whole team is locked out and payroll runs in two hours."],
    ["A colour in the dashboard changed after the update.", "Notifications arrive a few minutes late.", "Search returns no results for about a third of queries.",
     "Payments are failing for all of our customers.", "The production database is unreachable and orders are being lost every minute."],
    ["The invoice PDF has our old logo.", "Autopay retried once and then succeeded.", "We were charged twice for one order and need a refund.",
     "Our card was charged for a plan we cancelled last month and the account is suspended.", "Fraudulent charges are appearing on our card every hour and we cannot reach anyone."],
    ["The mobile app icon looks blurry.", "The app asks me to re-login once a day.", "The app crashes when I open the reports tab.",
     "The app crashes on launch, so I cannot use it at all.", "The app deleted my local drafts and crashes on launch before every client meeting."],
]
RUBRICS = {
    3: ["Cosmetic or informational; no impact on work.", "Degraded: something is broken but there is a workaround.", "Blocking: core functionality unusable or money/data at stake."],
    5: ["Cosmetic; no functional impact.", "Minor annoyance; easy workaround.", "A feature is broken; workaround exists but costs time.",
        "Core functionality unusable for the customer.", "Business-critical outage, data loss or active financial harm."],
    10: ["Purely cosmetic.", "Trivial inconvenience.", "Minor slowdown or delay.", "A secondary feature is broken; workaround exists.",
         "A primary feature is degraded.", "A primary feature is broken; painful workaround.", "Core functionality unusable for some users.",
         "Core functionality unusable for all users.", "Outage with financial or data impact.", "Catastrophic: ongoing loss with no recourse."],
}
# severity 0..4 mapped to level index per rubric size
ALIGN = {3: [0, 0, 1, 2, 2], 5: [0, 1, 2, 3, 4], 10: [0, 2, 4, 7, 9]}


def ordinal_probe_items(levels: int, ladders: list[list[str]] = LADDERS) -> list[TaskItem]:
    rub = [Level(level=i, description=d) for i, d in enumerate(RUBRICS[levels])]
    items = []
    for li, ladder in enumerate(ladders):
        for si, text in enumerate(ladder):
            items.append(TaskItem(
                task_id=f"ladder{li}_step{si}_L{levels}", tier="G", domain="severity_probe", state={"customer_report": text},
                questions={"severity": Question(type="score", instructions=f"Rate the severity of the customer's problem on the {levels}-level rubric.",
                                                criteria=rub, ground_truth=ALIGN[levels][si], framing_group=f"probe.severity{levels}")},
                label_provenance=LabelProvenance(source="generator", generator="ordinal_probes@1.0.0", seed=0), controls=Controls(),
                metadata={"ladder": li, "step": si, "severity": si},
            ))
    return items


def ordinal_probes(adapter: BaseAdapter, cache: ResponseCache | None = None, concurrency: int = 1) -> dict[str, Any]:
    out: dict[str, Any] = {"by_levels": {}}
    exp_by_levels: dict[int, dict[str, float]] = {}
    for L in (3, 5, 10):
        items = ordinal_probe_items(L)
        rows = {r.task_id: r for r in run_items(items, adapter, cache, suite="F", arm=f"levels={L}", concurrency=concurrency) if not r.error}
        ladders = []
        exp: dict[str, float] = {}
        exact = []
        for li in range(len(LADDERS)):
            steps = []
            for si in range(5):
                r = rows.get(f"ladder{li}_step{si}_L{L}")
                if r is None:
                    continue
                e = float(np.dot(r.probs, range(L)))
                steps.append(e)
                exp[f"{li}_{si}"] = e / (L - 1)  # normalised position 0..1 for cross-rubric comparison
                exact.append(int(r.argmax) == ALIGN[L][si])
            ladders.append(np.array(steps))
        exp_by_levels[L] = exp
        out["by_levels"][L] = {"n": len(rows), "monotonicity_rate": monotonicity_rate(ladders), "exact_accuracy": float(np.mean(exact)) if exact else float("nan"),
                               "mean_expected_by_severity": [float(np.mean([l[s] for l in ladders if len(l) > s])) for s in range(5)]}
    # scale invariance: Spearman between normalised expected positions across rubrics
    keys = sorted(set.intersection(*[set(v) for v in exp_by_levels.values()]))
    inv = {}
    for a, b in ((3, 5), (5, 10), (3, 10)):
        xa = [exp_by_levels[a][k] for k in keys]
        xb = [exp_by_levels[b][k] for k in keys]
        inv[f"{a}v{b}"] = {"spearman": float(spearmanr(xa, xb).statistic), "mean_abs_position_diff": float(np.mean(np.abs(np.array(xa) - np.array(xb))))}
    out["scale_invariance"] = inv
    return out
