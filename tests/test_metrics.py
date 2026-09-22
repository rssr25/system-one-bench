import numpy as np
import pytest

from sys1bench.metrics import calibration as C
from sys1bench.metrics import consistency as K
from sys1bench.metrics import decision_value as D
from sys1bench.metrics import ordinal as O
from sys1bench.metrics import robustness as R
from sys1bench.metrics import selective as S


def test_ece_perfect_and_worst():
    conf = np.array([0.9] * 10 + [0.6] * 10)
    correct = np.array([1] * 9 + [0] + [1] * 6 + [0] * 4)
    assert C.ece(conf, correct, 10) == pytest.approx(0.0, abs=1e-9)
    assert C.ece(np.ones(10), np.zeros(10), 10) == pytest.approx(1.0)


def test_ece_noise_floor_positive_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = C.ece_noise_floor(rng.uniform(0.5, 1, 100), resamples=50)[0]
    large = C.ece_noise_floor(rng.uniform(0.5, 1, 5000), resamples=50)[0]
    assert small > large > 0


def test_brier_and_decomposition():
    probs = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    y = np.array([0, 1, 0])
    assert C.brier(probs, y) == pytest.approx((0 + 0 + 0.5) / 3)
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    d = C.brier_decomposition(conf, correct, 10)
    assert d["brier_top"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-9)


def test_nll_clip_and_zero_count():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1, 1])
    assert np.isinf(C.nll(probs, y, None))
    assert np.isfinite(C.nll(probs, y, 1e-4))
    assert C.zero_prob_on_truth(probs, y) == 1


def test_quantisation_detects_step():
    probs = np.array([[0.73, 0.27], [0.01, 0.99]])
    assert C.quantisation_report(probs)["step"] == 0.01


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4000, 4))
    y = np.array([rng.choice(4, p=np.exp(l) / np.exp(l).sum()) for l in logits])
    over = np.exp(logits * 2)
    over /= over.sum(1, keepdims=True)
    t = C.fit_temperature(over, y)
    assert 1.6 < t < 2.4


def test_aurc_ordering_and_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    good = np.array([1, 1, 0, 0])
    bad = np.array([0, 0, 1, 1])
    assert S.aurc(conf, good) < S.aurc(conf, bad)
    assert S.coverage_at_risk(conf, good, 0.0) == 0.5
    assert S.e_aurc(conf, good) == pytest.approx(0.0, abs=1e-9)


def test_ordinal_metrics():
    levels = np.array([0, 1, 2, 3, 4])
    probs = np.eye(5)
    truth = levels.copy()
    s = O.ordinal_summary(probs, truth, levels)
    assert s["exact_accuracy_argmax"] == 1.0 and s["mae_argmax"] == 0.0 and s["qwk_argmax"] == pytest.approx(1.0)
    assert s["rps"] == pytest.approx(0.0)
    assert O.monotonicity_rate([np.array([0, 1, 1, 2]), np.array([2, 1])]) == 0.5


def test_jsd_and_complement():
    assert K.jsd([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)
    assert K.jsd([1, 0], [0, 1]) == pytest.approx(1.0, abs=1e-6)
    assert K.complement_consistency([0.7, 0.2], [0.3, 0.8])["complement_mad"] == pytest.approx(0.0, abs=1e-9)


def test_ood_auroc():
    p_in = np.array([[0.95, 0.05]] * 50)
    p_ood = np.array([[0.55, 0.45]] * 50)
    s = R.ood_summary(p_in, p_ood)
    assert s["auroc_confidence"] == pytest.approx(1.0)


def test_decision_value_bayes_beats_argmax_on_asymmetric_cost():
    probs = np.array([[0.6, 0.4]] * 100)
    truth = np.array([0] * 60 + [1] * 40)
    cost = np.array([[0, 1], [10, 0]])  # predicting 0 when truth is 1 costs 10
    assert D.value_of_calibration(probs, truth, cost) > 0
    r = D.expected_cost(probs, truth, cost, "threshold", escalate_cost=2.0, threshold=0.7)
    assert r["escalation_rate"] == 1.0
