"""Markdown results document for Suite A/B runs. One section per deployment class; every accuracy shown as
median [min, max] over framings; every ECE next to its floor; control arms and audits included."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.arms import label_noise_report, unknowable_report
from ..analysis.decomposition import decomposition_report
from ..analysis.meta_eval import label_order_report, short_circuit_report
from ..runners.benchmark_runner import load_manifest, load_predictions
from ..schemas import PredictionRow
from .scorecard import framing_scorecard, group_rows, scorecard


def _f(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, list):
        return "[" + ", ".join(_f(x, nd) for x in v) + "]"
    return str(v)


def _load_dir(d: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"dir": d, "preds": {}, "manifests": {}}
    for name in ("tickets", "phish", "unknowable", "noisy"):
        p = d / f"preds_{name}.jsonl"
        if p.exists():
            out["preds"][name] = load_predictions(p)
    for name, fn in (("tickets", "tickets.jsonl"), ("phish", "phish.jsonl"), ("unknowable", "tickets_unknowable.jsonl"), ("noisy", "tickets_noisy.jsonl")):
        p = d / fn
        if p.exists():
            out["manifests"][name] = load_manifest(p)
    return out


def _model_meta(rows: list[PredictionRow]) -> dict[str, Any]:
    r0 = next((r for r in rows if r.error is None), rows[0])
    hw = {r.metadata.get("hardware") for r in rows if r.metadata.get("hardware")}
    deployment = "local" if r0.adapter_id in ("laya_local", "mock", "majority_prior", "regex_keyword", "embed_knn", "nli_zeroshot") else "hosted"
    return {"model_id": r0.model_id, "adapter": r0.adapter_id, "version_hash": r0.version_hash, "hardware": ", ".join(sorted(hw)) or "–",
            "deployment": deployment, "route": r0.metadata.get("route")}


def _main_rows(rows: list[PredictionRow]) -> list[PredictionRow]:
    return [r for r in rows if r.arm in (None, "main")]


def section_model(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    lines: list[str] = []
    all_rows = [r for rs in data["preds"].values() for r in rs]
    if not all_rows:
        return "", {}
    meta = _model_meta(all_rows)
    lines.append(f"### {meta['model_id']}  \n adapter `{meta['adapter']}` · version `{meta['version_hash']}` · hardware {meta['hardware']} · route {meta['route']}\n")
    # headline table over both manifests
    lines.append("| question | primitive | K | n | acc median | acc range (framings) | ECE15 | ECE floor | ECE/floor | Brier | NLL(clip) | zero-p on truth | T (refit) | AURC | cov@5% | perm JSD | flips | p50 ms | p95 ms | schema fail |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cards: dict[str, dict] = {}
    for mname in ("tickets", "phish"):
        rows = _main_rows(data["preds"].get(mname, []))
        for (qk,), rs in sorted(group_rows(rows, "question_key").items()):
            if qk.startswith("sub_"):
                continue
            c = scorecard(rs)
            cards[f"{mname}.{qk}"] = c
            cal, sel, perm, temp = c.get("calibration", {}), c.get("selective", {}), c.get("permutation", {}), c.get("temperature", {})
            lines.append("| " + " | ".join([
                f"{mname}.{qk}", rs[0].primitive, str(rs[0].cardinality), str(cal.get("n", "–")),
                _f(c.get("accuracy")), _f(c.get("accuracy_range")), _f(cal.get("ece_width_15")), _f(cal.get("ece_floor_mean")),
                _f(cal.get("ece_over_floor"), 2), _f(cal.get("brier")), _f(cal.get("nll_clipped")), _f(cal.get("zero_prob_on_truth")),
                _f(temp.get("T"), 2), _f(sel.get("aurc")), _f(sel.get("coverage_at_risk_5")), _f(perm.get("mean_jsd"), 4),
                _f(perm.get("argmax_flip_rate")), _f(c.get("latency", {}).get("p50"), 0), _f(c.get("latency", {}).get("p95"), 0),
                _f(c.get("schema_failure_rate")),
            ]) + " |")
    # ordinal detail
    ords = {k: c["ordinal"] for k, c in cards.items() if "ordinal" in c}
    if ords:
        lines.append("\n**Ordinal (`score`) fidelity**\n\n| question | exact (argmax) | exact (expected) | off-by-one | MAE (expected) | QWK | RPS | Spearman |\n|---|---|---|---|---|---|---|---|")
        for k, o in ords.items():
            lines.append(f"| {k} | {_f(o['exact_accuracy_argmax'])} | {_f(o['exact_accuracy_expected'])} | {_f(o['off_by_one_accuracy'])} | {_f(o['mae_expected'])} | {_f(o['qwk_argmax'])} | {_f(o['rps'])} | {_f(o['spearman_expected'])} |")
    # framing variants + corruption
    lines.append("\n**Framing (Suite B)**: accuracy by framing variant; `crit_swapped` is the corruption control (descriptions swapped between two options).\n")
    lines.append("| question | f0 | paraphrases (min..max) | label only | +negatives | vague | swapped | corruption drop | JSD across framings | flip rate |\n|---|---|---|---|---|---|---|---|---|---|")
    for mname in ("tickets", "phish"):
        rows = data["preds"].get(mname, [])
        for (qk,), rs in sorted(group_rows(rows, "question_key").items()):
            if qk.startswith("sub_"):
                continue
            fs = framing_scorecard(rs)
            acc = fs["accuracy_by_framing"]
            paras = [v for k, v in acc.items() if k.startswith("para")]
            c = cards.get(f"{mname}.{qk}", {}).get("framing", {})
            lines.append("| " + " | ".join([f"{mname}.{qk}", _f(acc.get("f0")), (f"{min(paras):.3f}..{max(paras):.3f}" if paras else "–"),
                                             _f(acc.get("crit_label_only")), _f(acc.get("crit_with_negatives")), _f(acc.get("crit_vague")),
                                             _f(acc.get("crit_swapped")), _f(fs.get("corruption_drop")), _f(c.get("mean_pairwise_jsd"), 4), _f(c.get("argmax_flip_rate"))]) + " |")
    # decomposition
    if "phish" in data["preds"]:
        dec = decomposition_report(data["preds"]["phish"], "is_phishing")
        if dec.get("n", 0) >= 20:
            lines.append(f"\n**Decomposition (is_phishing, n={dec['n']})**: holistic acc {_f(dec['holistic']['accuracy'])} (Brier {_f(dec['holistic']['brier'])}) · "
                         f"fixed noisy-OR over 5 sub-questions {_f(dec['fixed_rule']['accuracy'])} (Brier {_f(dec['fixed_rule']['brier'])}) · "
                         f"fitted weights on held-out half {_f(dec['fitted_weights_eval_half']['accuracy'])} vs holistic on same half {_f(dec['holistic_eval_half']['accuracy'])}. "
                         f"Gain fixed {dec['decomposition_gain_fixed']:+.3f}, fitted {dec['decomposition_gain_fitted']:+.3f}.")
    # arms
    arm_lines = []
    if "unknowable" in data["preds"]:
        u = unknowable_report(data["preds"]["unknowable"], "priority")
        if u.get("n"):
            arm_lines.append(f"- **Unknowable arm** (tier removed, priority truth hidden, n={u['n']}): mean max-prob {_f(u['mean_max_prob'])}, "
                             f"conf>0.7 on {_f(u['frac_conf_over_0.7'])} of items, conf>0.9 on {_f(u['frac_conf_over_0.9'])}, mean entropy {_f(u['mean_entropy_bits'])} bits. "
                             f"A calibrated model should sit near 0.5 max-prob (two adjacent levels are possible).")
    if "noisy" in data["preds"] and "noisy" in data["manifests"]:
        ln = label_noise_report([r for r in data["preds"]["noisy"] if r.question_key == "queue"], data["manifests"]["noisy"])
        arm_lines.append(f"- **Label-noise control** (queue, 5% corrupted labels, n_noisy={ln['n_noisy']}): accuracy vs corrupted labels {_f(ln['accuracy_on_corrupted_labels'])} "
                         f"(should be near 0), confidence on corrupted {_f(ln['confidence_on_corrupted'])} vs clean {_f(ln['confidence_clean'])}"
                         + (" **[flag: more confident on corrupted]**" if ln["flag_more_confident_on_corrupted"] else "") + ".")
    for mname in ("tickets", "phish"):
        rows = data["preds"].get(mname, [])
        full = [r for r in rows if r.arm == "main" and r.permutation_id == "p0" and r.framing_id == "f0"]
        so = [r for r in rows if r.arm == "state_only"]
        oo = [r for r in rows if r.arm == "options_only"]
        if so and oo:
            for qk in sorted({r.question_key for r in full if r.primitive == "choice"}):
                f_, s_, o_ = ([r for r in x if r.question_key == qk] for x in (full, so, oo))
                truths = [r.ground_truth for r in f_]
                prior = max(truths.count(t) for t in set(truths)) / len(truths) if truths else float("nan")
                sc = short_circuit_report(f_, s_, o_, prior)
                arm_lines.append(f"- **Short-circuit audit** {mname}.{qk}: full {_f(sc['accuracy_full'])}, state-only (blank descriptions) {_f(sc['accuracy_state_only'])}, "
                                 f"options-only (no state) {_f(sc['accuracy_options_only'])}, majority prior {_f(prior)}"
                                 + (" **[flag: state-only above prior+0.10]**" if sc["flag_state_only"] else "") + (" **[flag: options-only above prior+0.10]**" if sc["flag_options_only"] else "") + ".")
        lo = label_order_report([r for r in rows if r.arm == "main" and r.framing_id == "f0"])
        for k, v in lo.items():
            arm_lines.append(f"- **Positional bias** {mname} K={k}: chi² p={_f(v['p'], 4)}, share of index 0 = {_f(v['index0_share'])} (uniform would be {1/k:.3f}).")
    if arm_lines:
        lines.append("\n**Control arms and audits**\n")
        lines.extend(arm_lines)
    # cost & failures
    costs = [r.cost_usd for r in all_rows if r.cost_usd is not None]
    n_req = len({(r.row_id, r.arm) for r in all_rows})
    fail_t = np.mean([bool(r.error and r.error.startswith("transport")) for r in all_rows])
    fail_s = np.mean([bool(r.error and not r.error.startswith("transport")) for r in all_rows])
    lines.append(f"\nRequests {n_req:,}; question-rows {len(all_rows):,}; transport failure rate {fail_t:.4f}; schema failure rate {fail_s:.4f}"
                 + (f"; total billed ${sum(costs):.4f} (${sum(costs)/max(len(all_rows),1)*100_000:.2f} per 100k question-answers)" if costs else "") + ".")
    return "\n".join(lines) + "\n", {"meta": meta, "cards": cards}


def build_report(dirs: list[Path], title: str = "sys1bench results") -> str:
    hosted, local = [], []
    for d in dirs:
        data = _load_dir(d)
        if not data["preds"]:
            continue
        sec, info = section_model(data)
        if not sec:
            continue
        (hosted if info["meta"]["deployment"] == "hosted" else local).append(sec)
    out = [f"# {title}\n", "Generated by `sys1bench report`. Data tier G (generated, policy-dependent labels). "
           "Accuracies are medians over question framings with the range in brackets; ECE is shown with its resampled noise floor. "
           "Hosted and local models are listed separately and must not be compared on latency.\n"]
    if hosted:
        out.append("## Hosted models (client-observed latency includes network)\n")
        out.extend(hosted)
    if local:
        out.append("## Local models (latency is compute on the stated hardware)\n")
        out.extend(local)
    return "\n".join(out)
