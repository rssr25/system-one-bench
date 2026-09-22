"""Markdown results document for Suite A/B runs. One section per deployment class; every accuracy shown as
median [min, max] over framings; every ECE next to its floor; control arms and audits included."""

from __future__ import annotations

import json
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
    local_adapters = ("laya_local", "mock", "majority_prior", "regex_keyword", "embed_knn", "nli_zeroshot", "encoder_finetuned")
    deployment = "local" if (r0.adapter_id in local_adapters or hw) else "hosted"
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


def build_report(dirs: list[Path], title: str = "sys1bench results", sweep_dirs: dict[str, Path] | None = None) -> str:
    """sweep_dirs maps a results dir name to a directory holding its sweep JSON files (e.g. {"jev": Path("results/jev_sweeps")})."""
    hosted, local = [], []
    for d in dirs:
        data = _load_dir(d)
        if not data["preds"]:
            continue
        sec, info = section_model(data)
        if not sec:
            continue
        sd = (sweep_dirs or {}).get(d.name) or (d.parent / f"{d.name}_sweeps")
        if sd.exists():
            sec += sweeps_section(sd, info["meta"]["model_id"])
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


# ----------------------------------------------------------------------------------------- sweeps


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def sweeps_section(d: Path, model_label: str) -> str:
    """Render any sweep outputs found in `d`: cardinality_*.json, length_*.json, interference_*.json, robustness_*.json."""
    lines: list[str] = []
    for p in sorted(d.glob("cardinality_*.json")):
        res = {int(k): v for k, v in _load_json(p).items()}
        lines.append(f"\n**Cardinality sweep** ({p.stem.replace('cardinality_', '')}, {model_label})\n\n| K | n | acc | ECE15 | ECE/floor | mean conf | p50 ms | error rate |\n|---|---|---|---|---|---|---|---|")
        for k in sorted(res):
            v = res[k]
            lines.append(f"| {k} | {v.get('n')} | {_f(v.get('accuracy'))} | {_f(v.get('ece15'))} | {_f(v.get('ece_over_floor'), 2)} | {_f(v.get('mean_conf'))} | {_f(v.get('latency', {}).get('p50'), 0)} | {_f(v.get('error_rate'))} |")
    for p in sorted(d.glob("length_*.json")):
        res = {int(k): v for k, v in _load_json(p).items()}
        qs = sorted({q for v in res.values() for q in v["by_question"]})
        lines.append(f"\n**State-length sweep** ({p.stem.replace('length_', '')}, {model_label}); accuracy per question, then truncation / error rate and p50 latency\n\n| target tokens | realised tokens | " + " | ".join(qs) + " | error rate | p50 ms |\n|---|---|" + "---|" * len(qs) + "---|---|")
        for L in sorted(res):
            v = res[L]
            q0 = v["by_question"][qs[0]]
            err = _f(q0.get("error_rate"))
            if q0.get("error_kinds"):
                err += " (" + ", ".join(f"{k} {n}" for k, n in q0["error_kinds"].items()) + ")"
            lines.append(f"| {L} | {v['state_tokens_mean']:.0f} | " + " | ".join(_f(v["by_question"][q].get("accuracy")) for q in qs) + f" | {err} | {_f(q0.get('latency', {}).get('p50'), 0)} |")
    for p in sorted(d.glob("interference_*.json")):
        res = _load_json(p)
        lines.append(f"\n**Multi-question interference** (target `{res['target']}`, n={res['n_items']}, {model_label}); alone accuracy {_f(res['alone'].get('accuracy'))}\n\n| co-asked kind | Q | JSD vs alone | argmax flips | Δ accuracy | p50 ms | cost / question |\n|---|---|---|---|---|---|---|")
        for kind, dd in res["by_kind"].items():
            for Q, v in sorted(dd.items(), key=lambda kv: int(kv[0])):
                c = v.get("cost_per_question_usd")
                lines.append(f"| {kind} | {Q} | {_f(v['mean_jsd_vs_alone'], 4)} | {_f(v['argmax_flip_rate'])} | {v['accuracy_delta_vs_alone']:+.3f} | {_f(v['latency_p50_ms'], 0)} | {('$%.2e' % c) if c else '–'} |")
    for p in sorted(d.glob("robustness_*.json")):
        res = _load_json(p)
        qs = sorted(res["clean"])
        lines.append(f"\n**Robustness** ({res['generator']}, n={res['n']}, {model_label}); Δ accuracy vs clean and mean JSD to the clean answer\n\n| arm | " + " | ".join(f"{q} Δacc / JSD" for q in qs) + " |\n|---|" + "---|" * len(qs))
        lines.append("| clean (accuracy) | " + " | ".join(_f(res["clean"][q]["accuracy"]) for q in qs) + " |")
        for dens, dd in res.get("distractors", {}).items():
            lines.append(f"| distractors {dens} | " + " | ".join(f"{dd[q]['accuracy_delta']:+.3f} / {_f(dd[q]['jsd_vs_clean'], 4)}" for q in qs) + " |")
        for kind, dd in res.get("perturbations", {}).items():
            lines.append(f"| {kind} | " + " | ".join(f"{dd[q]['accuracy_delta']:+.3f} / {_f(dd[q]['jsd_vs_clean'], 4)}" for q in qs) + " |")
        na = res.get("none_of_the_above")
        if na:
            w, wo = na["with_abstain_option"], na["without_abstain_option"]
            lines.append(f"\nNone-of-the-above (`{na['question']}`, {na['n_none']} items whose true option was removed): with an explicit abstain option the model abstains on {_f(w['abstain_rate_on_none'])} of them "
                         f"(in-distribution accuracy {_f(w['accuracy_in_dist'])}); without it, mean confidence is {_f(wo['conf_none'])} on none-items vs {_f(wo['conf_in'])} in-distribution, "
                         f"AUROC(confidence) {_f(wo.get('auroc_confidence'))}, and {_f(wo.get('frac_ood_conf_over_0.7'))} of none-items get confidence above 0.7.")
        ps = res.get("prior_shift")
        if ps:
            lines.append("\nPrior shift (noul): | base rate | mean P(yes) | accuracy | ECE15 |\n|---|---|---|---|")
            for share, v in ps.items():
                lines.append(f"| {_f(v['realised_base_rate'], 2)} | {_f(v['mean_p_true'])} | {_f(v['accuracy'])} | {_f(v['ece15'])} |")
    return "\n".join(lines) + ("\n" if lines else "")
