"""Portable suite orchestration (replaces the shell scripts): `sys1bench suite A|C|D|E|G|F|I|all`.

Each suite writes into <out>/ with the same file names the report reader expects. Everything goes through the
response cache so re-running is free, and every step that returns a FatalAdapterError is retried a few times
(local models on a shared GPU can fail to get a CUDA context at process start).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from ..adapters.base import BaseAdapter, FatalAdapterError
from ..data import framings_path
from ..framing import apply_corruption, expand_framings, permute_options, strip_options, strip_state
from ..framing.expand import load_framings
from ..generators import get_generator
from ..report.scorecard import group_rows, markdown_table, scorecard
from .benchmark_runner import load_manifest, run_items, write_manifest, write_predictions
from .cache import ResponseCache

FRAMING_FOR = {"support_tickets": "support_tickets", "phishing_email": "phishing_email", "log_triage": "log_triage",
               "policy_compliance": "policy_compliance", "guardrail_intent": "guardrail_intent", "multilingual_tickets": "support_tickets"}
DEFAULT_GENERATORS = ("support_tickets", "phishing_email")


def _retry(fn: Callable[[], Any], attempts: int = 5, wait_s: float = 30.0, log=print):
    for i in range(attempts):
        try:
            return fn()
        except FatalAdapterError as e:
            log(f"attempt {i + 1} failed: {e}")
            if i + 1 == attempts:
                raise
            time.sleep(wait_s)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def suite_a(adapter: BaseAdapter, out: Path, n: int = 500, seed: int = 42, perms: int = 3, generators=DEFAULT_GENERATORS,
            concurrency: int = 1, manifests_from: Path | None = None, log=print) -> dict[str, Any]:
    """Suite A + B: framings, permutations, corruption, short circuits, unknowable and label-noise arms, scorecards."""
    out.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(out / "cache.sqlite")
    summary: dict[str, Any] = {}
    for g in generators:
        short = {"support_tickets": "tickets", "phishing_email": "phish"}.get(g, g)
        mpath = out / f"{short}.jsonl"
        if manifests_from and (manifests_from / f"{short}.jsonl").exists():
            items = load_manifest(manifests_from / f"{short}.jsonl")
            write_manifest(items, mpath)
        elif mpath.exists():
            items = load_manifest(mpath)
        else:
            gen = get_generator(g, n=n, seed=seed)
            items = gen.generate()
            write_manifest(items, mpath)
            (out / f"{short}_rules.yaml").write_text(yaml.safe_dump(gen.regex_rules(), sort_keys=False))
        if hasattr(adapter, "fit"):
            adapter.fit(items)
        fr = load_framings(framings_path(FRAMING_FOR.get(g, g))) if FRAMING_FOR.get(g) else {}
        expanded = permute_options(expand_framings(items, fr), perms)
        log(f"[A] {g}: {len(items)} items -> {len(expanded)} rows")
        rows = _retry(lambda: run_items(expanded, adapter, cache, suite="A", arm="main", concurrency=concurrency, progress=True), log=log)
        rows += _retry(lambda: run_items(apply_corruption(items), adapter, cache, suite="A", arm="corruption", concurrency=concurrency), log=log)
        rows += _retry(lambda: run_items(strip_state(items), adapter, cache, suite="A", arm="state_only", concurrency=concurrency), log=log)
        rows += _retry(lambda: run_items(strip_options(items), adapter, cache, suite="A", arm="options_only", concurrency=concurrency), log=log)
        write_predictions(rows, out / f"preds_{short}.jsonl")
        cards = {qk: scorecard(rs) for (qk,), rs in group_rows([r for r in rows if r.arm == "main"], "question_key").items()}
        summary[g] = cards
    # control arms on the first generator
    g0 = generators[0]
    for arm, knobs in (("unknowable", {"unknowable_frac": 1.0, "seed": seed + 1}), ("noisy", {"label_noise": 0.05, "seed": seed + 2})):
        mpath = out / f"tickets_{arm}.jsonl"
        if manifests_from and (manifests_from / f"tickets_{arm}.jsonl").exists():
            items = load_manifest(manifests_from / f"tickets_{arm}.jsonl")
            write_manifest(items, mpath)
        elif mpath.exists():
            items = load_manifest(mpath)
        else:
            kw = dict(n=500, **knobs)
            items = get_generator(g0, **kw).generate()
            write_manifest(items, mpath)
        rows = _retry(lambda: run_items(items, adapter, cache, suite="A", arm="main", concurrency=concurrency), log=log)
        write_predictions(rows, out / f"preds_{arm}.jsonl")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "table.md").write_text("\n".join(markdown_table({f"{g} / {q}": c for g, cs in summary.items() for q, c in cs.items()}).splitlines()))
    return summary


def suite_c(adapter: BaseAdapter, out: Path, n: int = 150, seed: int = 42, concurrency: int = 1, budget: bool = False, log=print) -> None:
    from .sweeps import budget_sweep, cardinality_sweep, length_sweep

    out.mkdir(parents=True, exist_ok=True)
    log("[C] cardinality (rag_relevance)")
    _write_json(out / "cardinality_rag.json", _retry(lambda: cardinality_sweep(adapter, [2, 5, 10, 20, 50, 100, 255], n=n, seed=seed, generator="rag_relevance",
                                                                                question_key="best_passage", cache=ResponseCache(out / "cache_card.sqlite"), concurrency=concurrency), log=log))
    log("[C] length (support_tickets)")
    lengths = [128, 256, 512, 1024, 2048, 4096, 8192] + ([16384, 30000] if adapter.capabilities.deployment == "hosted" else [])
    _write_json(out / "length_tickets.json", _retry(lambda: length_sweep(adapter, lengths, n=min(n, 100), seed=seed, cache=ResponseCache(out / "cache_len.sqlite"), concurrency=concurrency), log=log))
    if budget and adapter.adapter_id == "laya_local":
        from ..adapters import get_adapter

        ck = getattr(adapter, "checkpoint", "english")

        def factory(b):
            return get_adapter("laya_local", checkpoint=ck, head_max_len=b, max_len=1024, device=getattr(adapter, "device", "cuda"))

        log("[C] option budget (laya)")
        _write_json(out / "budget_K12.json", _retry(lambda: budget_sweep(factory, [64, 128, 192, 256, 384, 512], k=12, n=200, seed=seed, cache=ResponseCache(out / "cache_budget.sqlite")), log=log))


def suite_d(adapter: BaseAdapter, out: Path, n: int = 200, seed: int = 42, generator: str = "support_tickets", concurrency: int = 1, log=print) -> None:
    from .robustness import robustness_suite

    out.mkdir(parents=True, exist_ok=True)
    log(f"[D] robustness ({generator})")
    _write_json(out / "robustness_tickets.json", _retry(lambda: robustness_suite(adapter, n=n, seed=seed, generator=generator, cache=ResponseCache(out / "cache_robust.sqlite"), concurrency=concurrency), log=log))


def suite_e(adapter: BaseAdapter, out: Path, tickets: Path, n: int = 500, concurrency: int = 1, log=print) -> None:
    from .sweeps import interference_sweep

    out.mkdir(parents=True, exist_ok=True)
    items = load_manifest(tickets)[:n]
    for target in ("queue", "priority"):
        log(f"[E] interference ({target})")
        _write_json(out / f"interference_{target}.json", _retry(lambda: interference_sweep(adapter, items, target, qs=[0, 2, 5, 10, 20],
                                                                                              cache=ResponseCache(out / "cache_interf.sqlite"), concurrency=concurrency), log=log))


def suite_g(adapter: BaseAdapter, out: Path, tickets: Path, phish: Path | None, n: int = 500, concurrency: int = 1, log=print) -> None:
    from .noul_consistency import choice_vs_noul, complement_test, threshold_portability

    out.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(out / "cache_noul.sqlite")
    items = load_manifest(tickets)[:n]
    fr = load_framings(framings_path("support_tickets"))
    negs = fr.get("tickets.is_angry", {}).get("negations")
    log("[G] noul consistency (is_angry)")
    res = {"complement": _retry(lambda: complement_test(adapter, items, "is_angry", negs, cache, concurrency), log=log),
           "choice_vs_noul": _retry(lambda: choice_vs_noul(adapter, items, "is_angry", cache, concurrency), log=log)}
    if phish and phish.exists():
        other = load_manifest(phish)[:n]
        ra = run_items([_only(it, "is_angry") for it in items], adapter, cache, suite="G", arm="port_a", concurrency=concurrency)
        rb = run_items([_only(it, "is_phishing") for it in other], adapter, cache, suite="G", arm="port_b", concurrency=concurrency)
        res["threshold_portability"] = {"from": "tickets:is_angry", "to": "phish:is_phishing", **threshold_portability(ra, rb)}
    _write_json(out / "noul_is_angry.json", res)


def _only(item, key):
    import copy

    b = copy.deepcopy(item)
    b.questions = {key: item.questions[key]}
    return b


def suite_f(adapter: BaseAdapter, out: Path, concurrency: int = 1, log=print) -> None:
    from .ordinal_probes import ordinal_probes

    out.mkdir(parents=True, exist_ok=True)
    log("[F] ordinal probes")
    _write_json(out / "ordinal_probes.json", _retry(lambda: ordinal_probes(adapter, ResponseCache(out / "cache_probes.sqlite"), concurrency), log=log))


def suite_i(out_a: Path, out: Path, log=print) -> None:
    from ..analysis.decision_value import decision_value_report
    from .benchmark_runner import load_predictions

    out.mkdir(parents=True, exist_ok=True)
    for short in ("tickets", "phish"):
        p, m = out_a / f"preds_{short}.jsonl", out_a / f"{short}.jsonl"
        if p.exists() and m.exists():
            log(f"[I] decision value ({short})")
            _write_json(out / f"decision_value_{short}.json", decision_value_report([r for r in load_predictions(p) if r.arm in (None, "main")], load_manifest(m)))
