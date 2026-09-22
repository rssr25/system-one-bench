# sys1-bench v2: Benchmark Specification for Typed System One Decision Models

**Status:** draft v2, 2026-09-22. Supersedes `spec_v1_original.md`.
**Targets:** TypeSafe Jev (hosted, `typesafe/jev-1.13` via OpenRouter Decisions API) and Convai Laya (`laya`, `laya-multilingual`, `laya-typed-decisions`, local), against fine-tuned encoder, zero-shot NLI, embedding-kNN, regex, majority-prior and constrained-decoding LLM baselines.

---

## 0. Positioning: what this benchmark adds

Three independent efforts already exist (as of 2026-09-22):

| Effort | Covers | Does not cover |
|---|---|---|
| dhruvmehra/jevbench | SST-2, AG News, Banking77; acc, F1, ECE(10 bins), latency, cost; 6 classifiers | calibration floor, framing, selective prediction, ordinal, interference, contamination-free data |
| scienthoon/jev-ood-calibration | 900 rule-generated tickets with policy-dependent labels; ECE vs noise floor; per-type temperature refit; quantization | Laya, baselines, robustness, efficiency, scale |
| NandhaKishorM/laya BENCHMARKS.md | typed-decisions, MASSIVE, XNLI, AG News, Emotion, Banking77, 6 workflows; ECE after temp fit; permutation | vendor-run; Jev numbers via single provider; no framing suite |

v2 is built so that **every headline number is (a) on contamination-resistant data, (b) reported relative to a noise floor or a trivial baseline, and (c) accompanied by its framing variance.** The novel suites are: Framing Sensitivity (B), Multi-Question Interference (E), Ordinal Fidelity (F), Noul Consistency (G), and Selective Prediction with native abstention (A2). Everything else is a hardened version of existing practice.

"System 1" in this document means *non-autoregressive typed decision model*: a model that reads state and a typed question and returns a probability distribution over a fixed answer space in one forward pass. It does not refer to dual-process psychology, and the benchmark does not test human cognitive biases.

---

## 1. Model interface contract

All adapters implement one contract. This is the unit everything else is built on.

```python
class DecisionRequest(BaseModel):
    state: str | dict | list                  # text or JSON; serialised deterministically
    questions: dict[str, Question]            # key -> Question
    meta: RequestMeta                         # task_id, framing_id, permutation_id, ...

class Question(BaseModel):
    type: Literal["choice", "score", "noul"]  # "noul" maps to OpenRouter "boolean"
    instructions: str
    criteria: list[Option] | list[Level] | None
    allow_abstain: bool = False               # adds a native "none_of_the_above" / escalate path where supported

class DecisionResponse(BaseModel):
    answers: dict[str, Answer]
    latency: LatencyRecord                    # see Suite H
    provider: ProviderRecord                  # model id string as returned, generation_id, route, version hash
    raw: dict                                 # verbatim payload, always stored

class Answer(BaseModel):
    probs: list[float] | float                # choice/score: vector over options; noul: P(true)
    argmax: str | int | bool
    confidence: float | None                  # vendor's own confidence, if any
    abstained: bool = False
    quantisation_step: float | None           # detected granularity (e.g. 0.01 for Jev)
    truncated: bool = False                   # state or options truncated by adapter or vendor
```

Adapter rules:
- **Identical criteria text** goes to every model (Jev criteria, Laya criteria, LLM prompt, NLI hypotheses, kNN label embeddings). Descriptions live in one place in the manifest.
- **Probabilities are validated.** Finite, within [0, 1], and summing to 1. Vendors quantise (Jev returns 0.01-rounded probabilities and about 0.6% of its vectors sum to 0.99), so sums within ±0.02 are rescaled and flagged `renormalised=True` with the raw sum kept; the `renormalised_rate` is a scorecard metric. Sums off by more than 0.02 are `schema_failure`. Nothing is silently corrected.
- **Nothing is silently fixed.** Truncation, renormalisation, and abstention are recorded as fields and become metrics.
- **Raw payloads are kept.** Every response's vendor payload is stored verbatim in the cache; when a parser changes, `adapter.reparse` rebuilds answers from the raw payload without a new API call.
- **Vendor limits are encoded as adapter capabilities** and checked at manifest-load time: Jev choice ≤ 255 options, score 2 to 10 levels, state ≤ 32k tokens, 64k per request; Laya context 512 (`laya`) or 1024 (`laya-multilingual`, `laya-typed-decisions`), option budget `head_max_len` 192/256 shared across options.

### 1.1 Adapters

| Adapter | Class | Notes |
|---|---|---|
| `jev_openrouter` | hosted | pin `typesafe/jev-1.13`, never `jev-latest`; store `generation_id`; record route (OpenRouter / Vercel AI Gateway / direct) because latency differs; exponential backoff with jitter; 1200 rpm ceiling |
| `laya_local` | local GPU | three checkpoints + Router; expose `head_max_len`, `max_len` as sweep parameters; batch size as parameter |
| `encoder_finetuned` | local | DistilBERT-base and ModernBERT-large fine-tuned per dataset on ≤10k examples; the strongest cheap baseline in jevbench |
| `nli_zeroshot` | local | `facebook/bart-large-mnli` or DeBERTa-v3-large-MNLI; hypothesis = label description |
| `embed_knn` | local | embed state and label descriptions (e.g. bge-m3), softmax over cosine with fitted temperature; the true apples-to-apples for "zero-shot from descriptions" |
| `regex_keyword` | local | hand-written rules per generated task; XenoSpectrum's regex hit 91.8% on phishing vs Jev 95.0% |
| `majority_prior` | local | class priors; sanity floor |
| `llm_constrained` | vLLM + Outlines/XGrammar | Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct; probabilities from first-token logits over option keys so LLMs get calibration metrics too |
| `llm_verbalised` | hosted | frontier LLM asked for label plus verbalised confidence; reference for "what a generalist does" |

### 1.2 Version drift and canaries

Hosted models move silently. Every session:
1. Runs a fixed 200-item **canary set** (mixed primitives) and stores the full probability vectors.
2. Compares to the last accepted canary run: if mean absolute probability shift > 0.01 or argmax disagreement > 1%, the session is tagged `model_drift_suspected` and results are stored under a new version hash rather than merged.
3. Records the exact model id string the provider returns, the generation id, and the route.

---

## 2. Data strategy

Three tiers. Headline numbers come from Tiers G and H only. Tier P is context.

### Tier P: public (contamination-suspect)
Banking77, CLINC150, AG News, SST-2, SST-5, DAIR Emotion, MASSIVE, XNLI, ToxiGen, jailbreak/prompt-injection sets. Used for comparability with prior work and for cardinality sweeps. Every result table marks Tier P rows with a contamination flag. Jev scores 94.2% on OpenBookQA and Laya's fine-tuned checkpoint was trained on typed-decisions; treat public scores accordingly.

### Tier G: generated, policy-dependent labels (contamination-free)
Rule-based generators where the correct label depends on a **policy that is stated in the question criteria but is not derivable from the state text alone**, following scienthoon's priority construction. Each generator emits state, questions, labels, and the generating parameters, so labels are exact and the noise rate is controllable.

Required generators (each ≥ 2,000 items, regenerable from seed):
- `support_tickets`: queue (choice 4 to 12), anger (noul), priority (score 0 to 4 with tier and SLA policy).
- `phishing_email`: phishing (noul), attack class (choice), urgency (score); decomposable into 5 sub-questions for Suite B.
- `log_triage`: JSON logs with injected distractor keys; severity (score), owning team (choice), page-on-call (noul).
- `rag_relevance`: query plus passage; relevant (noul), relevance grade (score 0 to 3), best passage of K (choice with K sweep).
- `policy_compliance`: a short policy plus a request; compliant (noul), violated clause (choice with a required "none" option), severity (score).
- `guardrail_intent`: benign vs jailbreak vs injection (choice), with paraphrase clusters.
- `multilingual_intent`: `support_tickets` templated in 8 languages for Laya router and Jev language behaviour.

Each generator has knobs: cardinality K, state length L (padding with realistic filler, not lorem ipsum), distractor density, label noise rate (default 0%, control arm 5%), and an **unknowable** arm where the label depends on information deliberately removed from the state (calibration ceiling test: a calibrated model must output near-uniform here).

### Tier H: human-labelled private held-out
- 1,000 to 2,000 real items per domain (support, moderation, RAG relevance) labelled by ≥ 3 annotators.
- Report Krippendorff's alpha; items below 0.6 agreement are kept but tagged `contested` and analysed separately.
- Never published in plain text. Distribute as encrypted archive with a public hash; publish only after a dated freeze period, then rotate.
- Every item carries a canary string in a metadata field (not the state) so future training-set scans can detect leakage.

### 2.1 Manifest schema v2 (JSONL)

```json
{
  "task_id": "gen_tickets_000412",
  "tier": "G",
  "domain": "support_triage",
  "language": "en",
  "state": "...",
  "state_tokens": 187,
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which queue should handle this ticket?",
      "criteria": [{"key": "billing", "description": "..."}, {"key": "shipping", "description": "..."}],
      "ground_truth": "billing",
      "framing_group": "queue_f0",
      "decomposition_of": null
    },
    "priority": {
      "type": "score",
      "instructions": "Priority 0 (lowest) to 4 (highest). Gold and enterprise tiers add one level.",
      "criteria": [{"level": 0, "description": "..."}, {"level": 4, "description": "..."}],
      "ground_truth": 3,
      "framing_group": "priority_f0"
    }
  },
  "label_provenance": {"source": "generator", "generator": "support_tickets@1.2.0", "seed": 42, "noise_injected": false},
  "controls": {"unknowable": false, "none_correct": false, "distractor_density": 0.0},
  "canary": "s1b-7f3a…"
}
```

Framing variants are separate manifest rows sharing `task_id` and `framing_group` with a distinct `framing_id`, so paired analysis is trivial.

---

## 3. Suites

### Suite A: Calibration and selective prediction

**A1 Calibration.** For every model that emits probabilities:
- ECE with equal-width and equal-mass bins at M ∈ {10, 15, 25}; report all three plus the **ECE noise floor** (expected ECE of a perfectly calibrated model with the same confidence histogram, 200 label resamplings). Report `ECE / floor`, not ECE alone.
- Smooth calibration error (kernel-based) as the binning-free primary.
- Brier score with Murphy decomposition into reliability, resolution, uncertainty.
- NLL with probability clipping at 1e-4, and unclipped NLL reported separately with the count of exact-zero probabilities assigned to the true label.
- **Quantisation report:** detected step size, fraction of probabilities that are exactly 0 or 1.
- **Per-primitive temperature refit:** fit T on a held-out half per (model, primitive, cardinality bucket), report T and post-fit ECE. Direction of miscalibration is reported per primitive because Jev is overconfident on choice and score and underconfident on noul.
- Reliability diagrams with bootstrap bands, per primitive.

**A2 Selective prediction.**
- Risk-coverage curve and AURC using max-probability, and separately using the vendor confidence field if it differs.
- Coverage at risk ≤ 1%, 5%, 10%.
- **Native abstention arm:** where a model exposes an abstain or escalate path (Laya act/escalate; a `none_of_the_above` option for Jev), run with `allow_abstain=true` and report abstention rate, precision of abstention (fraction of abstentions that would have been wrong), and the change in AURC.
- **Unknowable arm:** on Tier G `unknowable=true` items, report mean max-probability and fraction above 0.7. A calibrated model should sit near uniform.

### Suite B: Framing sensitivity (new)

The largest measured effect on Jev is how the question is written (62.6% vs 95.0% on the same emails). This suite makes that a first-class metric.

Variants per question, all sharing ground truth:
1. **Paraphrase set:** 5 human-written paraphrases of `instructions`, semantically equivalent.
2. **Criteria granularity:** label only; label plus one-sentence rubric; label plus rubric plus negative definitions ("do not use for ...").
3. **Decomposition:** single holistic question vs 3 to 5 sub-questions whose answers are combined by (a) fixed rule and (b) logistic weights fitted on a disjoint half.
4. **Criteria corruption (control):** descriptions swapped between two labels; descriptions replaced with vague text. Expect collapse; measures how much the model leans on descriptions vs state.
5. **Instruction position:** criteria before vs after state where the API permits.

Metrics:
- **Framing variance:** std of accuracy across paraphrases; **framing range:** max minus min.
- **Prediction instability:** mean pairwise JSD of probability vectors across paraphrases (per item), and argmax flip rate.
- **Decomposition gain:** accuracy(decomposed) minus accuracy(holistic).
- **Corruption drop:** accuracy(correct criteria) minus accuracy(swapped).
- Every headline accuracy in the report is shown as `median over paraphrases [min, max]`.

### Suite C: Cardinality, length and budget scaling

- K ∈ {2, 5, 10, 20, 50, 77, 150, 255} on Tier G `rag_relevance` (best passage of K) and Tier P Banking77/CLINC150.
- L ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000} tokens using realistic filler.
- **Budget ablation for Laya:** `head_max_len` ∈ {64, 128, 192, 256, 384, 512} × `max_len` ∈ {512, 1024}. Report tokens-per-option actually received. Without this, the 77-label collapse (0.38 to 0.425) measures a default, not the architecture.
- **Truncation as outcome:** items exceeding a model's context are still run; `truncated=true` is a metric (`truncation_rate`) and accuracy is reported both including and excluding truncated items.
- Report accuracy, ECE/floor and latency as functions of K and L with bootstrap bands.

### Suite D: Invariance and robustness

1. **Permutation invariance:** 5 random option orders per item; mean JSD to the canonical order; argmax flip rate; positional bias chi-square (index preference).
2. **Distractor resistance:** inject irrelevant JSON keys, prose logs, and near-miss content (text matching a wrong label's description) at densities {0, 0.25, 0.5}. Near-miss distractors are the informative case.
3. **OOD and negative detection:**
   - `none_correct=true` items with and without an explicit "none of the above" option.
   - Report AUROC of max-probability for in-distribution vs none-correct items (threshold-free), plus entropy shift as a secondary.
   - Language shift: English-trained checkpoints on non-English states. Laya collapses at 0.952 confidence on Khmer; this is a calibration failure, so report `confidence_on_collapse`.
4. **Surface perturbations:** casing, whitespace, typos at 2% character rate, unicode homoglyphs. Report accuracy delta and JSD.

### Suite E: Multi-question interference (new)

System One models answer many questions over one state in one pass. Measure whether answers are independent of co-asked questions.
- For each item, ask question q1 alone, then with Q ∈ {2, 5, 10, 20} additional questions (relevant, irrelevant, and adversarially phrased).
- Metrics: JSD between q1 alone and q1 in batch; argmax flip rate; accuracy delta; latency and cost per question as a function of Q (amortisation curve).
- Include the reverse: does asking q1 change q2's answer? Report an interference matrix on a small fixed question set.

### Suite F: Ordinal fidelity for `score` (new)

`score` is Laya's weakest primitive (SST-5 at 0.372) and Jev's most overconfident (refit T = 1.92). Treat it as ordinal, not categorical.
- Metrics: exact accuracy, off-by-one accuracy, MAE, quadratic weighted kappa, ranked probability score (the proper scoring rule for ordinal distributions), Spearman with ground truth.
- **Monotonicity test:** generated stimuli with a single controlled severity variable; report the fraction of stimulus ladders where expected score is monotone non-decreasing.
- **Scale invariance:** same items on 3-, 5-, and 10-level rubrics with aligned ground truth; report whether relative ordering is preserved.
- **Expected vs argmax:** report both, since vendors differ on which they surface.

### Suite G: Noul consistency (new)

- **Complement consistency:** P(yes | Q) + P(yes | not-Q) should be ≈ 1. Report mean absolute deviation.
- **Paraphrase stability:** std of P(yes) across paraphrases (from Suite B).
- **Choice-noul agreement:** for binary choice questions also posed as noul, JSD between the two.
- **Threshold portability:** fit a decision threshold on one Tier G generator, apply to another with the same nominal question; report accuracy loss. This is what "you cannot set one threshold and walk away" measures.

### Suite H: Efficiency and economics

Report hosted and local models in **separate tables**; never rank across deployment classes on one latency scale.
- Hosted: client-observed p50, p90, p95, p99 at concurrency {1, 8, 32}; route recorded; time-of-day recorded; server-side timing header if exposed.
- Local: GPU compute latency (CUDA events) and end-to-end wall clock, batch sizes {1, 8, 32, 64, 128}, on stated hardware. Primary hardware: one T4 (matches Convai's published numbers) and one A100 or 4090; report both.
- Throughput in decisions per second and **questions per second** (multi-question batching changes the unit).
- Cost per 100k decisions: hosted from billed tokens (Jev bills input only, once per state regardless of Q); local from amortised GPU-hour at a stated price. Show cost as a function of Q.

---

## 4. Statistics

- **Unit of analysis is the item, not the seed.** Hosted models are deterministic given input; seeds only affect data sampling. Variance sources are: dataset sample, option order, paraphrase, distractor draw, and (hosted) version drift. Each is a factor, not noise to average away.
- **Paired comparisons:** item-level paired bootstrap (10,000 resamples) for accuracy, ECE, Brier, AURC differences; McNemar for argmax agreement. Cluster bootstrap over `framing_group` when paraphrases are pooled. Report 95% CIs and the fraction of resamples where the sign holds.
- **No seed-level Wilcoxon.** With 5 seeds the two-sided minimum p is 0.0625, so a p < 0.01 criterion is unreachable.
- **Power:** at n = 500 the accuracy CI is about ± 2.5 points; suites that claim differences below 3 points use n ≥ 2,000.
- **Multiple comparisons:** Holm correction within each suite's headline table.
- **Missing data:** failed calls are retried (backoff, 5 attempts); remaining failures are reported as `schema_failure_rate` and `transport_failure_rate` and excluded from accuracy with the exclusion count shown. No pipeline halt at 1%; an alpha API will exceed that.

---

## 5. Meta-evaluation (auditing the benchmark)

1. **Short-circuit tests:** state-only and options-only runs; flag any split where either exceeds majority-prior + 10 points.
2. **Label-order bias audit:** chi-square on argmax index over permutations.
3. **Surface leakage:** n-gram overlap between state and label descriptions; report per-split leakage rate and accuracy on the zero-overlap subset.
4. **Label noise control:** the 5% corrupted arm; a model's accuracy on corrupted items should be near the corruption rate, and its confidence on them should not be higher than on clean items.
5. **Contamination probes:** run Tier P items with minimal edits (entity swaps that preserve label); a large accuracy drop relative to Tier G behaviour suggests memorisation.
6. **Regex ceiling:** for each Tier G generator, report the hand-written regex baseline; a System One model that does not beat it is not adding value on that task.
7. **Canary scan protocol:** documented procedure for checking future model releases against the Tier H canary strings.

---

## 6. Reporting

- Each model gets a **scorecard** per suite with: headline metric as `median [min, max] over framings`, CI, noise floor where applicable, and the trivial baseline on the same rows.
- **No absolute tier tables across deployment classes.** Tiers in v1 (e.g. p95 < 15 ms as SOTA) placed Jev in "critical failure" by construction. Where tiers are wanted, define them relative to (a) the noise floor for calibration, (b) the fine-tuned encoder for accuracy, (c) the deployment class for latency.
- Every table row carries: model id string as returned, version hash, route, hardware, date, tier (P/G/H), contamination flag.
- Exports: `results/summary.json` (hierarchical), `results/tables/*.md`, `results/tables/*.tex`, `results/plots/` (reliability diagrams, risk-coverage, scaling curves, interference matrix).

---

## 7. Reproducibility

- Pin: Python 3.11, PyTorch 2.4, Transformers ≥ 4.45, vLLM version, Outlines/XGrammar version, CUDA 12.4; `uv.lock` committed.
- Every request and raw response cached in SQLite keyed by (adapter, model id string, request hash); cache is shipped with results so numbers can be re-derived offline without API access.
- Generators are pure functions of (version, seed); the manifest for every published table is regenerable and its hash is in the table footer.
- Hosted runs record date, route, generation ids; the canary comparison from §1.2 is stored alongside.
- Docker: `Dockerfile.gpu` (CUDA 12.4, non-root, weight volume) and `Dockerfile.cpu` for the hosted-only path so contributors without a GPU can run Suites A, B, D, E, G against Jev.

---

## 8. Repository structure

```text
sys1-bench/
├── configs/
│   ├── default_eval.yaml
│   ├── models/                     # one YAML per adapter; pins model id strings
│   └── suites/                     # A..H suite definitions
├── data/
│   ├── generators/                 # Tier G generators (pure, seeded, versioned)
│   ├── manifests/public/           # Tier P manifests
│   ├── manifests/generated/        # Tier G manifests (regenerable; hashes committed)
│   ├── manifests/private/          # Tier H (encrypted; hash committed)
│   ├── framings/                   # paraphrase sets, decompositions, corruptions
│   └── canary/                     # 200-item drift canary
├── src/sys1bench/
│   ├── adapters/                   # base, jev_openrouter, laya_local, encoder_finetuned, nli_zeroshot,
│   │                               # embed_knn, regex_keyword, majority_prior, llm_constrained, llm_verbalised
│   ├── metrics/
│   │   ├── calibration.py          # ECE (ew/em), smooth CE, noise floor, Brier decomposition, clipped NLL, quantisation
│   │   ├── selective.py            # risk-coverage, AURC, coverage@risk, abstention precision
│   │   ├── ordinal.py              # MAE, off-by-one, QWK, RPS, monotonicity
│   │   ├── consistency.py          # JSD, flip rate, complement consistency, interference matrix
│   │   ├── robustness.py           # OOD AUROC, entropy shift, perturbation deltas
│   │   └── efficiency.py           # latency percentiles, throughput, cost curves
│   ├── framing/                    # paraphrase expansion, decomposition combiners, corruption
│   ├── runners/                    # benchmark_runner, sweeps (K, L, head_max_len, Q), canary
│   ├── analysis/                   # paired bootstrap, McNemar, cluster bootstrap, Holm, meta_eval
│   ├── report/                     # scorecards, tables, plots
│   └── schemas.py                  # Pydantic v2: manifest v2, request/response contract
├── docker/
├── scripts/
├── tests/                          # metric unit tests against closed-form cases; adapter contract tests
├── pyproject.toml
└── README.md
```

---

## 9. Phased roadmap

| Phase | Scope | Exit criterion |
|---|---|---|
| 0 | Contract, schemas, metric kernels with unit tests (closed-form ECE, Brier, RPS cases) | `pytest` green; metrics match reference implementations on synthetic data |
| 1 | Jev and Laya adapters, canary, two Tier G generators (`support_tickets`, `phishing_email`), Suites A and B | First scorecards with framing ranges and noise floors |
| 2 | Baseline adapters (encoder-ft, NLI, kNN, regex, prior, LLM-constrained), Tier P manifests, Suite C and H | Reproduce jevbench numbers within CI as a sanity check |
| 3 | Suites D, E, F, G; remaining generators; meta-eval audits | All audits pass on Tier G; interference matrix published |
| 4 | Tier H collection, encryption, canary protocol; public release with frozen manifests | Dated freeze; external reproduction by one outside group |

---

## 10. Open design questions

Recorded here so decisions are traceable. See discussion thread for current thinking.

1. Should Tier H be collected at all in v2, or deferred until Tier G results show where human labels change conclusions?
2. Decomposition combiner: fixed rule only (no fitting, cleaner) or also fitted weights (matches XenoSpectrum, but introduces a training step)?
3. Which multilingual scope: Laya's router makes it interesting; Jev's language support is undocumented. Eight languages or English plus two?
4. Should the LLM baselines include a frontier model with logprobs, or is the 3B constrained model sufficient to make the efficiency argument?
5. How to treat Laya `head_max_len` in headline tables: vendor default, best-of-sweep, or both?
6. Public leaderboard or static report? A leaderboard needs Tier H rotation and a submission protocol.
7. Governance for hosted drift: rerun everything on every canary alarm, or freeze results per version hash and let tables accumulate?

---

## 11. Additions after brainstorm (2026-09-22)

All accepted. Implementation status is tracked in `README.md`.

- **Downstream decision value (Suite I).** Each Tier G generator ships a cost matrix per question (`cost_matrix` in the manifest). Report mean realised cost and regret vs oracle under three policies: act on argmax, act on the Bayes-optimal option given the probabilities, and act-or-escalate at a threshold. `value_of_calibration` = cost(argmax) minus cost(Bayes): zero for a model whose probabilities carry no information about asymmetric risk. Metric kernel: `metrics/decision_value.py`.
- **Prior-shift stress test (Suite D4).** Resample a manifest so one label holds 20%, 50%, 80% of the ground truth and re-score calibration. A model whose probabilities do not move with the base rate is reproducing a training prior. Utility: `framing.resample_prior_shift`.
- **Hybrid router baseline.** `hybrid_router` adapter: a System One model answers; questions below a confidence threshold escalate to a second adapter. Sweep the threshold and plot accuracy, latency and cost against escalation rate. This is the production comparison the vendors' marketing avoids.
- **Adversarial framing.** Suite B gains a worst-case row: accept externally contributed paraphrases that preserve meaning but lower accuracy, report `accuracy_min` over the adversarial set separately from the friendly paraphrase range.
- **Agentic closed-loop task (Suite J, phase 3).** A 20-step tool-selection episode where each step is a `choice` over tools plus a `noul` "is the task complete". Score episode success, compounding error, and calibration of the completion probability. Compares System One models as the decision layer inside an agent, which is how they are marketed.
- **Public drift tracker.** The 200-item canary (`data/canary/canary.jsonl`) is run daily against each hosted model and its drift curve published. Cheap, and the reference other people cite when a version changes silently.

## 12. Future models

The harness must outlive Jev 1.13 and Laya 1.0. Rules:

1. **Nothing vendor-specific outside `adapters/`.** Suites, metrics and reports see only `DecisionRequest`, `DecisionResponse` and `ModelCapabilities`.
2. **Capabilities are declared, then measured.** An adapter declares limits (options, levels, state tokens, primitives, native abstain, batching, languages). The runner records violations per row as `capability_issues`; the report turns them into rates. A model with a smaller envelope is not penalised by a crash, it is scored on what it accepted and its coverage is reported.
3. **New hosted model = a YAML file.** `generic_http` maps a conventional JSON decisions API via config (`configs/models/example_future_vendor.yaml`). Only unusual shapes need code, and those register with `@register("id")` or the `sys1bench.adapters` entry-point group from a separate package.
4. **New primitive = a schema change, not a rewrite.** Primitives are a literal in `schemas.py`; metrics dispatch on it. A future `rank` or `extract` primitive adds a branch in the contract and a metric module, with the existing suites unchanged.
5. **Version is data.** Every prediction row carries the model id string the provider returned and a version hash. Results across versions are never merged; the canary decides when a new version starts.
6. **Manifests are frozen by hash, generators by version.** A future model is scored on the same bytes as today's, and generator upgrades produce a new manifest hash rather than silently changing old numbers.
