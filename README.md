# system-one-bench

Benchmark for typed **System One decision models**: non-autoregressive models that read a block of state and typed questions (`choice`, `score`, `noul`) and return calibrated probability distributions in one forward pass. Primary targets are TypeSafe **Jev** (hosted) and Convai **Laya** (open weights), compared against fine-tuned encoders, zero-shot NLI, embedding-kNN, regex, and constrained-decoding LLM baselines.

## Documents

- [`docs/SPEC_v2.md`](docs/SPEC_v2.md): current benchmark specification (suites, data tiers, metrics, statistics, audits, roadmap).
- [`docs/spec_v1_original.md`](docs/spec_v1_original.md): the original harness specification v2 supersedes, kept for diffing.
- [`docs/REVIEW_v1.md`](docs/REVIEW_v1.md): review of v1 against the public state of the art as of 2026-09-22, with the changes it motivated.

## Design principles (v2)

1. Headline numbers come from contamination-resistant data (generated with policy-dependent labels, or private human-labelled).
2. Every calibration number is reported relative to its noise floor; every accuracy number next to a trivial baseline.
3. Every accuracy is reported as a median with a range over question framings.
4. Hosted and local models are never ranked on one latency scale.
5. The item, not the seed, is the unit of statistical analysis.

Status: specification stage. No code yet.
