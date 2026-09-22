import numpy as np
import pytest

from sys1bench.adapters import get_adapter, list_adapters
from sys1bench.adapters.base import finalize_answer
from sys1bench.adapters.jev_openrouter import parse_answer as jev_parse
from sys1bench.analysis import (
    label_order_report,
    leakage_report,
    mcnemar,
    paired_bootstrap,
    short_circuit_report,
)
from sys1bench.data import framings_path
from sys1bench.framing import expand_framings, permute_options, strip_options, strip_state
from sys1bench.framing.expand import load_framings
from sys1bench.generators import get_generator
from sys1bench.report import group_rows, markdown_table, scorecard
from sys1bench.runners import ResponseCache, run_items
from sys1bench.schemas import Answer, Level, Option, Question


def test_registry_has_core_adapters():
    names = list_adapters()
    for n in ("mock", "jev_openrouter", "laya_local", "generic_http", "hybrid_router", "majority_prior", "regex_keyword"):
        assert n in names


def test_contract_rejects_bad_vectors():
    q = Question(type="choice", instructions="x", criteria=[Option(key="a"), Option(key="b")])
    assert finalize_answer(q, [0.7, 0.2]).error is not None
    assert finalize_answer(q, [0.7, float("nan")]).error is not None
    assert finalize_answer(q, [0.7, 0.3]).ok and finalize_answer(q, [0.7, 0.3]).quantisation_step == 0.1


def test_jev_parse_shapes():
    q = Question(type="choice", instructions="x", criteria=[Option(key="a"), Option(key="b")])
    a = jev_parse(q, {"choice": "a", "probabilities": {"a": 0.73, "b": 0.27}, "confidence": 0.73})
    assert a.argmax == "a" and a.quantisation_step == 0.01
    a2 = jev_parse(q, {"probabilities": [{"key": "a", "probability": 0.2}, {"key": "b", "probability": 0.8}]})
    assert a2.argmax == "b"
    n = Question(type="noul", instructions="y?")
    assert jev_parse(n, {"probability": 0.9}).probs[0] == 0.9
    s = Question(type="score", instructions="s", criteria=[Level(level=0), Level(level=1), Level(level=2)])
    assert jev_parse(s, {"distribution": {"0": 0.1, "1": 0.2, "2": 0.7}}).argmax == "2"


def test_end_to_end_mock(tmp_path):
    items = get_generator("support_tickets", n=120, seed=1).generate()
    fr = load_framings(framings_path("support_tickets"))
    rows_items = permute_options(expand_framings(items, fr), 2)
    cache = ResponseCache(tmp_path / "c.sqlite")
    good = get_adapter("mock", skill=0.7, temperature=0.5, quantise=0.01)  # clearly over-confident
    rows = run_items(rows_items, good, cache, suite="A", concurrency=4)
    assert 0.99 * len(rows_items) <= len(cache) <= len(rows_items)  # identical requests (e.g. a shuffle equal to canonical) dedupe
    rows_again = run_items(rows_items, good, cache)
    assert [r.probs for r in rows] == [r.probs for r in rows_again]  # cache hit reproduces
    cards = {}
    for (qk,), rs in group_rows(rows, "question_key").items():
        cards[qk] = scorecard(rs, floor_resamples=20)
    c = cards["queue"]
    assert 0.6 < c["accuracy"] < 1.0
    assert c["framing"]["n_framings"] >= 6
    assert c["calibration"]["ece_over_floor"] > 0
    assert c["calibration"]["quant_step"] == 0.01
    assert c["temperature"]["direction"] == "overconfident"
    assert "permutation" in c and c["permutation"]["n_items"] == 120
    assert "ordinal" in cards["priority"] and "rps" in cards["priority"]["ordinal"]
    md = markdown_table(cards)
    assert "queue" in md and "ECE/floor" in md


def test_baselines_and_audits():
    items = get_generator("support_tickets", n=200, seed=2).generate()
    prior = get_adapter("majority_prior")
    prior.fit(items)
    rows_prior = run_items(items, prior)
    prior_acc = float(np.mean([r.correct for r in rows_prior if r.question_key == "queue"]))
    rules = get_generator("support_tickets").regex_rules()
    rx = get_adapter("regex_keyword", rules=rules)
    rows_rx = run_items(items, rx)
    rx_acc = float(np.mean([r.correct for r in rows_rx if r.question_key == "queue"]))
    assert rx_acc > prior_acc  # regex ceiling should beat priors on generated data
    m = get_adapter("mock", skill=0.85)
    full = [r for r in run_items(items, m) if r.question_key == "queue"]
    so = [r for r in run_items(strip_state(items), m) if r.question_key == "queue"]
    oo = [r for r in run_items(strip_options(items), m) if r.question_key == "queue"]
    rep = short_circuit_report(full, so, oo, prior_acc)
    assert set(rep) >= {"flag_state_only", "flag_options_only"}
    leak = leakage_report(items)
    assert 0 <= leak["leak_rate"] <= 1
    perm_rows = run_items(permute_options(items, 3), get_adapter("mock", position_bias=3.0))
    lo = label_order_report(perm_rows)
    assert any(v["index0_share"] > 0.4 for v in lo.values())
    pb = paired_bootstrap(np.array([r.correct for r in full], float), np.array([r.correct for r in rows_prior if r.question_key == "queue"], float), n_boot=500)
    assert pb["ci_low"] <= pb["diff"] <= pb["ci_high"]
    mc = mcnemar([r.correct for r in full], [r.correct for r in rows_prior if r.question_key == "queue"])
    assert 0 <= mc["p"] <= 1


def test_hybrid_router_escalates():
    h = get_adapter("hybrid_router", threshold=0.99, primary={"adapter": "mock", "skill": 0.5, "temperature": 3.0},
                    fallback={"adapter": "mock", "skill": 1.0, "latency_ms": 1000})
    items = get_generator("phishing_email", n=20, seed=1).generate()
    rows = run_items(items, h)
    assert all(r.latency_ms > 1000 for r in rows)


def test_decomposition_and_prior_shift():
    from sys1bench.analysis import decomposition_report
    from sys1bench.framing import resample_prior_shift

    items = get_generator("phishing_email", n=200, seed=5).generate()
    rows = run_items(items, get_adapter("mock", skill=0.9))
    rep = decomposition_report(rows, "is_phishing")
    assert rep["n"] == 200 and len(rep["sub_questions"]) == 5
    assert "decomposition_gain_fixed" in rep and "decomposition_gain_fitted" in rep
    shifted = resample_prior_shift(items, "is_phishing", "true", 0.8, n=100)
    share = np.mean([it.questions["is_phishing"].ground_truth for it in shifted])
    assert 0.75 <= share <= 0.85 and shifted[0].controls.prior_shift.startswith("is_phishing")


def test_typesafe_and_laya_parse_shapes():
    from sys1bench.adapters.jev_typesafe import parse_answer as ts_parse
    from sys1bench.adapters.jev_typesafe import to_vendor_question as ts_q
    from sys1bench.adapters.laya_local import parse_answer as laya_parse
    from sys1bench.adapters.laya_local import to_vendor_question as laya_q

    q = Question(type="choice", instructions="x", criteria=[Option(key="a", description="A"), Option(key="b")])
    assert ts_q(q)["criteria"] == {"a": "A", "b": None}
    a = ts_parse(q, {"type": "choice", "choice": "a", "probabilities": {"a": 0.88, "b": 0.12}, "confidence": 0.76})
    assert a.argmax == "a" and a.confidence == 0.76 and a.quantisation_step == 0.01
    s = Question(type="score", instructions="s", criteria=[Level(level=0, description="lo"), Level(level=2, description="mid"), Level(level=4, description="hi")])
    assert ts_q(s)["criteria"] == ["lo", "mid", "hi"]
    sa = ts_parse(s, {"type": "score", "score": 1.05, "legend": {"0": "lo", "1": "mid", "2": "hi"}, "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}, "confidence": 0.92})
    assert sa.argmax == "2" and sa.probs == [0.0, 0.95, 0.05]  # index 1 maps to level value 2
    n = Question(type="noul", instructions="y?")
    assert ts_parse(n, {"type": "noul", "noul": 0.95}).probs[0] == 0.95
    la = laya_parse(q, {"type": "choice", "choice": "b", "probabilities": {"a": 0.3, "b": 0.7}, "confidence": 0.4, "action": {"act_probability": 0.2}})
    assert la.argmax == "b" and la.abstained is True
    ln = laya_parse(n, {"type": "noul", "noul": 0.9707, "confidence": 0.9707, "action": {"act_probability": 1.0}})
    assert ln.argmax == "true" and ln.abstained is False
    assert laya_q(q)["criteria"] == {"a": "A", "b": "b"}


def test_sweeps_and_report(tmp_path):
    from sys1bench.report.results_doc import build_report
    from sys1bench.runners import write_manifest, write_predictions
    from sys1bench.runners.sweeps import cardinality_sweep, interference_sweep, length_sweep

    m = get_adapter("mock", skill=0.8)
    cs = cardinality_sweep(m, [2, 4], n=40)
    assert set(cs) == {2, 4} and cs[4]["realised_cardinality"] == 4
    ls = length_sweep(m, [128, 512], n=20)
    assert ls[512]["state_tokens_mean"] > ls[128]["state_tokens_mean"]
    items = get_generator("support_tickets", n=30, seed=3).generate()
    inter = interference_sweep(m, items, "queue", qs=[0, 2, 5])
    assert set(inter["by_kind"]) == {"relevant", "irrelevant", "adversarial"} and 5 in inter["by_kind"]["irrelevant"]
    # report over a fake results dir
    d = tmp_path / "mock"
    d.mkdir()
    write_manifest(items, d / "tickets.jsonl")
    fr = load_framings(framings_path("support_tickets"))
    rows = run_items(permute_options(expand_framings(items, fr), 2), m, suite="A", arm="main")
    rows += run_items(strip_state(items), m, arm="state_only") + run_items(strip_options(items), m, arm="options_only")
    write_predictions(rows, d / "preds_tickets.jsonl")
    noisy = get_generator("support_tickets", n=40, seed=4, label_noise=0.2).generate()
    write_manifest(noisy, d / "tickets_noisy.jsonl")
    write_predictions(run_items(noisy, m, arm="main"), d / "preds_noisy.jsonl")
    md = build_report([d])
    assert "Local models" in md and "tickets.queue" in md and "Label-noise control" in md and "Short-circuit audit" in md


def test_quantisation_slack_and_reparse(tmp_path):
    import json

    from sys1bench.adapters.jev_typesafe import JevTypeSafeAdapter
    from sys1bench.runners.benchmark_runner import to_request
    from sys1bench.schemas import DecisionResponse, LatencyRecord, ProviderRecord

    q = Question(type="score", instructions="s", criteria=[Level(level=i) for i in range(5)])
    a = finalize_answer(q, [0.04, 0.17, 0.67, 0.11, 0.0])  # sums to 0.99 (0.01 quantisation)
    assert a.ok and a.renormalised and abs(sum(a.probs) - 1) < 1e-9 and a.raw_prob_sum == pytest.approx(0.99)
    assert finalize_answer(q, [0.5, 0.5, 0.5, 0, 0]).error is not None  # 1.5 is a real failure
    f = Answer.failed(q, "x")
    assert f.probs == [] and json.loads(f.model_dump_json())["probs"] == []
    # legacy cache entry with NaN probs is repaired on read and rebuilt from raw by reparse
    item = get_generator("support_tickets", n=1, seed=1).generate()[0]
    req = to_request(item)
    raw = {"model": "jev-1.13.0", "answers": {
        "queue": {"type": "choice", "choice": item.questions["queue"].ground_truth,
                  "probabilities": {k: (0.99 if k == item.questions["queue"].ground_truth else 0.0) for k in item.questions["queue"].option_keys}, "confidence": 0.99},
        "is_angry": {"type": "noul", "noul": 0.1},
        "priority": {"type": "score", "score": 1.9, "probabilities": {"0": 0.04, "1": 0.17, "2": 0.67, "3": 0.11, "4": 0.0}, "confidence": 0.69}}}
    legacy = DecisionResponse(answers={k: Answer.failed(qq, "probabilities sum to 0.99") for k, qq in item.questions.items()},
                              latency=LatencyRecord(client_ms=1.0), provider=ProviderRecord(adapter_id="jev_typesafe", model_id_requested="jev-1.13.0"), raw=raw)
    payload = json.loads(legacy.model_dump_json())
    for ans in payload["answers"].values():
        ans["probs"] = [None] * 5  # what the old code wrote
    cache = ResponseCache(tmp_path / "c.sqlite")
    ad = JevTypeSafeAdapter(api_key="dummy")
    key = req.cache_key(ad.adapter_id, ad.model_id, ad.tunables)
    cache._db.execute("INSERT INTO responses VALUES (?,?,?,?,?)", (key, "jev_typesafe", "jev-1.13.0", 0.0, json.dumps(payload)))
    cache._db.commit()
    rows = run_items([item], ad, cache)  # no network: cache hit + reparse
    by = {r.question_key: r for r in rows}
    assert by["queue"].error is None and by["queue"].correct is True and by["queue"].renormalised
    assert by["priority"].error is None and by["priority"].argmax == "2" and by["priority"].renormalised
    assert by["is_angry"].error is None and by["is_angry"].probs[0] == 0.1
    assert cache.get(key).answers["priority"].ok  # repaired entry written back


def test_robustness_suite_mock(tmp_path):
    from sys1bench.framing.perturb import perturb_items
    from sys1bench.runners.robustness import robustness_suite

    items = get_generator("support_tickets", n=5, seed=1).generate()
    p = perturb_items(items, "homoglyphs_10pct")
    assert p[0].state != items[0].state and p[0].controls.perturbation == "homoglyphs_10pct"
    res = robustness_suite(get_adapter("mock", skill=0.8), n=40, seed=1, perturbations=("upper", "typos_2pct"), densities=(0.0, 0.5))
    assert set(res["perturbations"]) == {"upper", "typos_2pct"} and 0.5 in res["distractors"]
    assert res["none_of_the_above"]["n_none"] > 0 and "auroc_confidence" in res["none_of_the_above"]["without_abstain_option"]
    assert set(res["prior_shift"]) == {0.2, 0.5, 0.8}


def test_scorecard_dedupes_repeated_sibling_rows():
    items = get_generator("support_tickets", n=30, seed=2).generate()
    fr = load_framings(framings_path("support_tickets"))
    rows = run_items(expand_framings(items, fr), get_adapter("mock", skill=0.8), arm="main")
    angry = [r for r in rows if r.question_key == "is_angry"]
    assert len(angry) > 30 * 6  # raw rows include duplicates from queue criteria variants
    c = scorecard(angry, floor_resamples=10)
    assert c["calibration"]["n"] == 30


def test_fatal_adapter_error_is_not_swallowed():
    from sys1bench.adapters.base import FatalAdapterError
    from sys1bench.adapters.mock import MockAdapter

    class Broken(MockAdapter):
        def decide(self, request):
            raise FatalAdapterError("device unavailable")

    items = get_generator("support_tickets", n=2, seed=1).generate()
    with pytest.raises(FatalAdapterError):
        run_items(items, Broken())
