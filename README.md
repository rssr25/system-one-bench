# system-one-bench

Benchmark harness for typed **System One decision models**: non-autoregressive models that read a block of state and typed questions (`choice`, `score`, `noul`) and return calibrated probability distributions in one forward pass. First targets are TypeSafe **Jev** (hosted) and Convai **Laya** (open weights); the harness is model-agnostic so future models plug in through an adapter or a YAML config.

## Documents

- [`docs/SPEC_v2.md`](docs/SPEC_v2.md): benchmark specification (suites, data tiers, metrics, statistics, audits, roadmap, future-model rules).
- [`docs/REVIEW_v1.md`](docs/REVIEW_v1.md): review of the original spec against the public state of the art (2026-09-22).
- [`docs/spec_v1_original.md`](docs/spec_v1_original.md): the original specification, kept for diffing.

## Design principles

1. Headline numbers come from contamination-resistant data: generated items whose labels depend on a stated policy, or private human-labelled sets.
2. Every calibration number is reported relative to its noise floor; every accuracy next to a trivial baseline.
3. Every accuracy is a median with a range over question framings.
4. Hosted and local models are never ranked on one latency scale.
5. The item, not the seed, is the unit of statistical analysis.
6. Nothing vendor-specific lives outside `adapters/`.

## Quickstart (offline, no API key or GPU)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q

sys1bench generate support_tickets tickets.jsonl --n 500 --seed 42 --rules-out rules.yaml
sys1bench run tickets.jsonl preds_mock.jsonl --adapter mock \
    --framings data/framings/support_tickets.yaml --permutations 5 --corruption --short-circuit
sys1bench run tickets.jsonl preds_prior.jsonl --adapter majority_prior
sys1bench score preds_mock.jsonl preds_prior.jsonl --out results/summary.json --table results/table.md
```

## Running real models

```bash
export OPENROUTER_API_KEY=...
sys1bench canary jev_openrouter --model typesafe/jev-1.13          # drift check first, every session
scripts/run_suite_A.sh configs/models/jev_1.13.yaml results/jev     # Suite A end to end

pip install -e ".[local]" laya                                       # GPU box
scripts/run_suite_A.sh configs/models/laya_en.yaml results/laya_en
```

A new hosted model with a conventional JSON API needs only a config file: copy `configs/models/example_future_vendor.yaml`. Anything else subclasses `BaseAdapter` and registers with `@register("id")` or the `sys1bench.adapters` entry-point group.

## Layout

```
src/sys1bench/
  schemas.py        manifest v2, DecisionRequest/Response contract, ModelCapabilities
  adapters/         mock, jev_openrouter, laya_local, generic_http, hybrid_router,
                    majority_prior, regex_keyword, embed_knn, nli_zeroshot, llm_constrained
  metrics/          calibration (ECE + noise floor, smooth CE, Brier decomposition, clipped NLL,
                    quantisation, temperature), selective (AURC, coverage@risk, abstention),
                    ordinal (MAE, QWK, RPS, monotonicity), consistency (JSD, flips, interference),
                    robustness (OOD AUROC, entropy), efficiency, decision_value (cost matrices)
  generators/       Tier G: support_tickets, phishing_email (policy-dependent labels, knobs, regex rules, cost matrices)
  framing/          paraphrase/criteria expansion, corruption, permutation, short-circuit, decomposition, prior shift
  runners/          cached runner (SQLite, thread-safe), drift canary
  analysis/         paired bootstrap, McNemar, cluster bootstrap, Holm; short-circuit/leakage/label-order audits; decomposition
  report/           scorecards (median [min,max] over framings, ECE/floor, T, permutation JSD, latency, cost), markdown tables
  cli.py            generate | run | score | canary | adapters | generators
data/framings/      paraphrase sets and criteria variants per question group
data/canary/        fixed 200-item drift canary
configs/            model configs (Jev, Laya, hybrid, future-vendor template), suite definitions
```

## Status

| Component | State |
|---|---|
| Contract, schemas, capability declarations | done |
| Metric kernels with closed-form tests | done |
| Generators: support_tickets, phishing_email | done (template diversity is low; see roadmap) |
| Framing expansion, corruption, permutation, short-circuit, decomposition, prior shift | done |
| Runner with cache, canary, CLI, scorecards | done |
| Adapters: mock, majority_prior, regex_keyword, hybrid_router | done, tested offline |
| Adapters: jev_openrouter, laya_local, generic_http | implemented, parsing unit-tested; **not yet run against the live API or weights** |
| Adapters: embed_knn, nli_zeroshot, llm_constrained | implemented, need optional deps; untested |
| Generators: log_triage, rag_relevance (K up to 255), policy_compliance, guardrail_intent, multilingual | todo |
| Suite runners for C (scaling sweeps), E (interference), J (agentic) | todo |
| Tier P public manifests, Tier H collection | todo |
| Plots (reliability, risk-coverage, scaling, interference matrix), LaTeX export | todo |

Known limitation: the two generators use fixed templates, so a regex baseline saturates on `queue`. Next generator iteration adds surface variation (paraphrased templates, typos, multilingual) while keeping rule-derived labels.
