"""Plots for the results report (matplotlib, optional dependency): reliability diagrams with bootstrap bands,
risk-coverage curves, framing-range bars, cardinality/length scaling curves, interference heatmap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..metrics.selective import risk_coverage
from ..schemas import PredictionRow


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def reliability_diagram(rows_by_model: dict[str, list[PredictionRow]], out: Path, bins: int = 10, title: str = "") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--", label="perfect")
    edges = np.linspace(0, 1, bins + 1)
    rng = np.random.default_rng(0)
    for name, rows in rows_by_model.items():
        ok = [r for r in rows if r.error is None and r.correct is not None]
        if len(ok) < 20:
            continue
        conf = np.array([max(r.probs) for r in ok])
        corr = np.array([float(r.correct) for r in ok])
        idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, bins - 1)
        xs, ys, lo, hi = [], [], [], []
        for b in range(bins):
            m = idx == b
            if m.sum() < 5:
                continue
            xs.append(conf[m].mean())
            ys.append(corr[m].mean())
            boot = [corr[m][rng.integers(0, m.sum(), m.sum())].mean() for _ in range(300)]
            lo.append(np.quantile(boot, 0.025))
            hi.append(np.quantile(boot, 0.975))
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=name)
        ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xlabel("confidence (max probability)")
    ax.set_ylabel("accuracy in bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def risk_coverage_plot(rows_by_model: dict[str, list[PredictionRow]], out: Path, title: str = "") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for name, rows in rows_by_model.items():
        ok = [r for r in rows if r.error is None and r.correct is not None]
        if len(ok) < 20:
            continue
        cov, risk = risk_coverage(np.array([max(r.probs) for r in ok]), np.array([float(r.correct) for r in ok]))
        ax.plot(cov, risk, lw=1.4, label=name)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error rate among answered)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def framing_range_plot(acc_by_model_framing: dict[str, dict[str, float]], out: Path, title: str = "") -> Path:
    """acc_by_model_framing[model][framing_id] = accuracy. One row per model: dot per framing, bar for range."""
    plt = _plt()
    models = list(acc_by_model_framing)
    fig, ax = plt.subplots(figsize=(5.2, 0.5 * len(models) + 1.2))
    for i, m in enumerate(models):
        accs = acc_by_model_framing[m]
        vals = list(accs.values())
        ax.hlines(i, min(vals), max(vals), color="#999", lw=3, alpha=0.6)
        for f, v in accs.items():
            ax.plot(v, i, marker="o" if f == "f0" else ".", color="#c33" if f == "f0" else "#333", ms=7 if f == "f0" else 5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("accuracy  (red: canonical wording, dots: other framings)", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def scaling_plot(sweep: dict[int, dict[str, Any]], out: Path, xlabel: str, metric: str = "accuracy", title: str = "") -> Path:
    plt = _plt()
    xs = sorted(sweep)
    ys = [sweep[x].get(metric, np.nan) for x in xs]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(xs, ys, marker="o", lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def interference_heatmap(res: dict[str, Any], out: Path, metric: str = "mean_jsd_vs_alone", title: str = "") -> Path:
    plt = _plt()
    kinds = list(res["by_kind"])
    qs = sorted({int(q) for k in kinds for q in res["by_kind"][k]})
    M = np.array([[res["by_kind"][k].get(q, res["by_kind"][k].get(str(q), {})).get(metric, np.nan) for q in qs] for k in kinds], float)
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    im = ax.imshow(M, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds, fontsize=8)
    ax.set_xlabel("extra questions co-asked (Q)")
    for i in range(len(kinds)):
        for j in range(len(qs)):
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > np.nanmax(M) / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label=metric)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def multi_line_plot(series: dict[str, list[tuple[float, float]]], out: Path, xlabel: str, ylabel: str, title: str = "",
                    xlog: bool = False, ylim: tuple[float, float] | None = None) -> Path:
    """Several named series on one axis; used for cross-model sweep comparisons in the README."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, pts in series.items():
        pts = sorted(p for p in pts if p[1] == p[1])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, lw=1.6, label=name)
    if xlog:
        ax.set_xscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out
