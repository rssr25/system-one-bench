# Changelog

## 0.3.0 (2026-09-22)

First installable release.

- Model-agnostic contract (`DecisionRequest` / `DecisionResponse` / `ModelCapabilities`); adapters for TypeSafe Jev (first-party API), Convai Laya (local Router), a config-only `generic_http` adapter for future vendors, `hybrid_router`, and baselines (majority prior, regex, embedding kNN, zero-shot NLI, constrained-decoding LLM with logprobs).
- Metric kernels: ECE with resampled noise floor, smooth calibration error, Brier decomposition, clipped NLL, quantisation report, per-half temperature refit; AURC / coverage at risk / abstention; ordinal MAE, QWK, RPS, monotonicity; JSD, flip rates, interference; OOD AUROC; latency, cost; decision value under cost matrices.
- Tier G generators with policy-dependent labels and paired control arms: `support_tickets`, `phishing_email` (with 5-way decomposition), `rag_relevance` (2 to 255 options).
- Suites A (calibration, selective prediction), B (framing: paraphrases, criteria variants, corruption, decomposition), C (cardinality, length, option budget), D (distractors, none-of-the-above, perturbations, prior shift), E (multi-question interference); audits (short-circuit, leakage, positional bias, label noise, unknowable arm).
- Cached runner with reparse-from-raw, drift canary (packaged 200-item set), item-level statistics, markdown report with hosted and local models separated, plots.
- First live results for Jev 1.13.0 and Laya 0.3.4 (english, typed-decisions) on identical manifests.
