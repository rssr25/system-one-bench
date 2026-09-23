# Changelog

## 0.3.2 (2026-09-23)

- The TypeSafe adapter serves any API-compatible server: `url`, `api_key`, `allow_alias`, `deployment`, and a
  `/v1/models` version probe recorded on every row. Packaged configs `kev_0.8b`, `kev_4b`, `kev_9b` for
  [Kev](https://github.com/jaredpalmer/kev); Kev-0.8B/4B/9B benchmarked on the same manifests as Jev and Laya.
- README with results figures across models; `multi_line_plot` for cross-model sweep charts.
- Publish workflow skips files that already exist on PyPI.

## 0.3.1 (2026-09-22)

- CLI fails fast with a clear message when no TypeSafe key is found, when the key is rejected (401/403), when a
  moving alias is used, or when an adapter is unknown; `run` warns when rows carry errors.
- Packaged example model configs (`sys1bench configs`, `--config jev_1.13`); `.env` is read from the working
  directory only.

## Unreleased

- `sys1bench suite` orchestrates suites A, C, D, E, F, G, I portably with retries; `report --latex` and `report --html`
  (self-contained dashboard); sweep figures in `plots`.
- New suites: F ordinal probes (monotonicity ladders, scale invariance), G noul consistency (complement, choice-vs-noul,
  threshold portability), I decision value (cost matrices), hybrid threshold sweep; adversarial framings reported as a
  worst case.
- New generators: `log_triage`, `policy_compliance`, `guardrail_intent`, `multilingual_tickets`; `support_tickets` 1.1.0
  adds surface diversity (openers, register, sign-offs, typos); token estimate calibrated against billed tokens.
- Baselines: fine-tuned encoder (`encoder_finetuned`, trained on a disjoint seed); NLI and encoder baselines run.
- Laya adapter: lazy single-checkpoint load, strict device check (FatalAdapterError), truncation by estimate,
  package refusals classified as `rejected_by_model`.
- `scripts/canary_daily.sh` appends to a drift history and redraws the drift curve.

## 0.3.0 (2026-09-22)

First installable release.

- Model-agnostic contract (`DecisionRequest` / `DecisionResponse` / `ModelCapabilities`); adapters for TypeSafe Jev (first-party API), Convai Laya (local Router), a config-only `generic_http` adapter for future vendors, `hybrid_router`, and baselines (majority prior, regex, embedding kNN, zero-shot NLI, constrained-decoding LLM with logprobs).
- Metric kernels: ECE with resampled noise floor, smooth calibration error, Brier decomposition, clipped NLL, quantisation report, per-half temperature refit; AURC / coverage at risk / abstention; ordinal MAE, QWK, RPS, monotonicity; JSD, flip rates, interference; OOD AUROC; latency, cost; decision value under cost matrices.
- Tier G generators with policy-dependent labels and paired control arms: `support_tickets`, `phishing_email` (with 5-way decomposition), `rag_relevance` (2 to 255 options).
- Suites A (calibration, selective prediction), B (framing: paraphrases, criteria variants, corruption, decomposition), C (cardinality, length, option budget), D (distractors, none-of-the-above, perturbations, prior shift), E (multi-question interference); audits (short-circuit, leakage, positional bias, label noise, unknowable arm).
- Cached runner with reparse-from-raw, drift canary (packaged 200-item set), item-level statistics, markdown report with hosted and local models separated, plots.
