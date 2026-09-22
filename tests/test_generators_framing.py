
import numpy as np

from sys1bench.data import framings_path
from sys1bench.framing import (
    apply_corruption,
    decompose_fixed_rule,
    expand_framings,
    permute_options,
    strip_options,
    strip_state,
)
from sys1bench.framing.expand import apply_decomposition_weights, fit_decomposition_weights, load_framings
from sys1bench.generators import get_generator


def test_generators_deterministic_and_valid():
    a = get_generator("support_tickets", n=50, seed=7).generate()
    b = get_generator("support_tickets", n=50, seed=7).generate()
    assert [x.model_dump_json() for x in a] == [x.model_dump_json() for x in b]
    c = get_generator("support_tickets", n=50, seed=8).generate()
    assert a[0].state != c[0].state
    p = get_generator("phishing_email", n=30, seed=1).generate()
    assert all(q.ground_truth is not None for it in p for q in it.questions.values())
    assert any(q.decomposition_of == "is_phishing" for q in p[0].questions.values())


def test_priority_policy_applied():
    items = get_generator("support_tickets", n=200, seed=3).generate()
    for it in items:
        m = it.metadata
        expected = min(4, m["base_urgency"] + (1 if m["angry"] else 0) + (1 if m["tier"] in ("gold", "enterprise") else 0))
        assert it.questions["priority"].ground_truth == expected


def test_knobs():
    items = get_generator("support_tickets", n=100, seed=1, cardinality=3, target_tokens=600, unknowable_frac=1.0, none_correct_frac=1.0).generate()
    assert all(it.questions["queue"].cardinality <= 3 for it in items)
    assert all(it.state_tokens >= 500 for it in items)
    assert all(it.controls.unknowable for it in items)
    assert all(it.questions["queue"].ground_truth == "none_of_the_above" for it in items)
    assert all("tier" not in it.state.split("Context:")[1] for it in items)


def test_framing_expansion_and_controls():
    items = get_generator("support_tickets", n=5, seed=1).generate()
    fr = load_framings(framings_path("support_tickets"))
    rows = expand_framings(items, fr)
    ids = {q.framing_id for r in rows for q in r.questions.values()}
    assert {"f0", "para1", "para5", "crit_label_only", "crit_with_negatives", "crit_vague"} <= ids
    assert all(r.questions["queue"].ground_truth == items[0].questions["queue"].ground_truth for r in rows if r.task_id == items[0].task_id)
    perms = permute_options(items, 3)
    assert len(perms) == 5 * 4
    assert sorted(o.key for o in perms[1].questions["queue"].criteria) == sorted(o.key for o in perms[0].questions["queue"].criteria)
    cor = apply_corruption(items)
    assert cor[0].questions["queue"].framing_id == "crit_swapped"
    assert strip_state(items)[0].permutation_id == "state_only" and strip_options(items)[0].state == "(no content)"


def test_decomposition_combiners():
    assert decompose_fixed_rule({"a": 0.5, "b": 0.5}, "any") == 0.75
    X = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.7]] * 20)
    y = np.array([1, 0, 1, 0] * 20)
    w = fit_decomposition_weights(X, y)
    pred = apply_decomposition_weights(X, w) > 0.5
    assert (pred == y.astype(bool)).mean() == 1.0


def test_rag_relevance_generator_cardinality():
    for k in (2, 20, 255):
        items = get_generator("rag_relevance", n=5, seed=1, cardinality=k).generate()
        for it in items:
            q = it.questions["best_passage"]
            assert q.cardinality == k and q.ground_truth in it.state["passages"]
            assert it.state["passages"][q.ground_truth].count(it.metadata["entity"]) >= 1
            assert it.questions["relevance"].ground_truth in (0, 1, 2, 3)
            assert it.questions["is_relevant"].ground_truth == (it.metadata["focus_grade"] == 3)


def test_control_arms_pair_with_clean_manifest():
    for name in ("support_tickets", "phishing_email"):
        clean = get_generator(name, n=60, seed=5).generate()
        for knobs in (dict(distractor_density=0.5), dict(label_noise=0.2), dict(target_tokens=800)):
            arm = get_generator(name, n=60, seed=5, **knobs).generate()
            assert [a.task_id for a in arm] == [c.task_id for c in clean]
            if "label_noise" not in knobs:
                for a, c in zip(arm, clean):
                    assert {k: q.ground_truth for k, q in a.questions.items()} == {k: q.ground_truth for k, q in c.questions.items()}
                    # content identical up to the injected material
                    ca, cc = (a.state_text(), c.state_text())
                    assert ca.startswith(cc[:60])
            else:
                assert any(a.label_provenance.noise_injected for a in arm)


def test_new_generators_valid_and_paired():
    for name in ("log_triage", "policy_compliance", "guardrail_intent", "multilingual_tickets"):
        a = get_generator(name, n=40, seed=2).generate()
        assert len(a) == 40 and all(q.ground_truth is not None for it in a for q in it.questions.values())
        b = get_generator(name, n=40, seed=2).generate()
        assert [x.model_dump_json() for x in a] == [x.model_dump_json() for x in b]
    logs = get_generator("log_triage", n=100, seed=1).generate()
    for it in logs:
        lvl = it.state["log"]["level"]
        assert it.questions["severity"].ground_truth == {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3}[lvl]
        assert it.questions["page_oncall"].ground_truth == (it.questions["severity"].ground_truth >= 2 and it.state["log"]["env"] == "production")
    pol = get_generator("policy_compliance", n=100, seed=1).generate()
    assert any(it.questions["violated_clause"].ground_truth == "none" for it in pol) and any(it.questions["violated_clause"].ground_truth != "none" for it in pol)
    ml = get_generator("multilingual_tickets", n=8, seed=1).generate()
    assert {it.language for it in ml} == {"de", "es", "fr", "hi"}
