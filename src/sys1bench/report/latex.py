"""LaTeX export of the headline table per model (booktabs)."""

from __future__ import annotations

from typing import Any

COLS = [("question", None), ("prim.", None), ("acc", "accuracy"), ("range", "accuracy_range"), ("ECE/floor", "calibration.ece_over_floor"),
        ("Brier", "calibration.brier"), ("T", "temperature.T"), ("AURC", "selective.aurc"), ("p50 ms", "latency.p50")]


def _get(d: dict, path: str | None):
    if path is None:
        return None
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _fmt(v) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v).replace("_", r"\_")


def latex_table(cards: dict[str, dict[str, Any]], caption: str, label: str) -> str:
    """cards: '<manifest>.<question>' -> scorecard (with a 'primitive' key added by the caller)."""
    head = " & ".join(h for h, _ in COLS)
    rows = []
    for name, c in cards.items():
        vals = [name.replace("_", r"\_"), c.get("primitive", "")] + [_fmt(_get(c, p)) for _, p in COLS[2:]]
        rows.append(" & ".join(vals) + r" \\")
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{ll" + "r" * (len(COLS) - 2) + "}", r"\toprule",
                      head + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"])
