# Review of spec v1 against the public state of the art (2026-09-22)

## Prior independent work

- **dhruvmehra/jevbench**: SST-2, AG News, Banking77 at n=500; Jev 1.13, Laya, DistilBERT-ft, BART-MNLI zero-shot, gpt-5-mini, claude-sonnet-5. Findings used here: fine-tuned DistilBERT beats Jev and Laya on AG News (91.0 vs 84.3 vs 90.6) and Banking77 (88.0 vs 76.4 vs 38.2); Jev p50 through OpenRouter 376 to 389 ms; Laya ECE 0.511 at 77 labels.
- **scienthoon/jev-ood-calibration**: 900 rule-generated tickets with a policy-dependent priority label; ECE 0.107 vs noise floor 0.024; per-type temperature refit (choice 1.30, score 1.92, boolean 0.66); probabilities quantised to 0.01 with many exact zeros; public benchmark rows (OpenBookQA 94.2%) flagged as contamination-suspect.
- **NandhaKishorM/laya BENCHMARKS.md**: option token budget (`head_max_len` 192/256) causes collapse above ~20 options; both checkpoints over-confident as shipped; English checkpoint collapses outside English at 0.952 confidence; ordinal `score` weakest (SST-5 0.372).
- **XenoSpectrum**: Jev 62.6% asked once vs 95.0% decomposed into five questions on 2,000 phishing emails; regex baseline 91.8%.
- **TypeSafe's own evals** reference the average of two frontier LLMs as ground truth.

## v1 strengths retained

Three decision primitives and a shared manifest; selective prediction (AURC); permutation JSD; state-only and options-only short-circuit audits; label-order chi-square; `schema_failure_rate`; bootstrap CIs; label-description granularity ablation; LaTeX export.

## v1 gaps and the v2 response

| Gap in v1 | v2 change |
|---|---|
| No framing-sensitivity metric | Suite B: paraphrase sets, decomposition, criteria corruption; all accuracies reported as median with range over framings |
| ECE at one binning, no floor | ECE/floor ratio, smooth CE, Brier decomposition, clipped NLL, quantisation report, per-primitive temperature refit |
| All datasets public and contamination-suspect | Three data tiers; headline numbers from generated (policy-dependent labels) and private human-labelled data |
| Label provenance unaddressed | `label_provenance` field, inter-annotator agreement, label-noise control arm, unknowable arm |
| Latency tiers put hosted and local on one scale | Separate tables by deployment class; route, hardware, and server-side timing recorded |
| Cardinality sweep confounded by Laya option budget | `head_max_len` and `max_len` as sweep factors; truncation as a metric |
| `score` treated as classification | Suite F: MAE, off-by-one, QWK, RPS, monotonicity, scale invariance |
| Multi-question batching only as a size knob | Suite E: interference JSD, flip rate, interference matrix, amortisation curve |
| OOD by entropy z-score only | AUROC of confidence, none-of-the-above present vs absent, language shift, confidence on collapse |
| No noul-specific tests | Suite G: complement consistency, choice-noul agreement, threshold portability |
| Baselines miss the winners | Fine-tuned encoder, regex, embedding-kNN, LLM with first-token logprobs added |
| Seed-level Wilcoxon at p<0.01 (unreachable with 5 seeds) | Item-level paired bootstrap, McNemar, cluster bootstrap over framing groups, Holm |
| Halt at 1% dropped calls | Retry, report failure rates, continue |
| No version pinning for hosted model | Pin `jev-1.13`, canary set every session, version hash on every row |
| Arbitrary absolute tier thresholds | Tiers relative to noise floor, fine-tuned encoder, and deployment class |
| English only | Multilingual generator and language-shift arm |
