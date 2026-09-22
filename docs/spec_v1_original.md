# TASK SPECIFICATION FOR CLAUDE CODE: SYSTEM 1 DECISION MODEL BENCHMARK HARNESS

You are an expert ML Systems and Benchmarking Engineer. Build a production-ready, modular, reproducible benchmark harness to evaluate non-autoregressive, typed **System 1 decision models** (specifically evaluating [PRIMARY_TARGET_MODEL_A, e.g., TypeSafe Jev API] and [PRIMARY_TARGET_MODEL_B, e.g., ConvAI Laya / ModernBERT-large / mmBERT-base]) alongside traditional generative LLM constrained-decoding baselines.

Generate the full repository structure, executable Python harness code, configuration schemas, evaluation runners, and metric reporting engines according to the modular specification below.

---

## 1. REPOSITORY STRUCTURE & DEPENDENCY GRAPH

Scaffold and implement the benchmark framework following this directory structure:

```text
sys1-bench/
├── configs/
│   ├── default_eval.yaml          # Global execution, seed, run-count settings
│   ├── models/                    # Model adapter configs (APIs, endpoints, weights)
│   └── suites/                    # Task suite definitions & dataset splits
├── data/
│   ├── manifests/                 # Standardized JSONL task manifests
│   └── synthetic/                 # Scripts to generate adversarial/OOD inputs
├── src/
│   ├── adapters/                  # Model interface abstraction layer
│   │   ├── base.py                # Abstract Base Adapter
│   │   ├── typesafe_jev.py        # TypeSafe Jev client adapter
│   │   ├── convai_laya.py         # ConvAI Laya HF/vLLM local adapter
│   │   └── baseline_llm.py        # Outlines/vLLM structured decoding baseline
│   ├── metrics/                   # Mathematical evaluation kernels
│   │   ├── calibration.py         # ECE, MCE, Brier score, NLL
│   │   ├── selective.py           # Risk-Coverage curves, AURC, selective accuracy
│   │   └── performance.py         # Latency (p50, p95, p99), throughput, token cost
│   ├── runners/                   # Execution orchestrators
│   │   ├── benchmark_runner.py    # Main evaluation loop & concurrency manager
│   │   └── ablations.py           # Ablation experiment driver
│   ├── analysis/                  # Statistical validation & meta-eval
│   │   ├── aggregator.py          # Multi-run aggregation & bootstrap CI
│   │   └── meta_eval.py           # Label leakage, gaming, & metric sensitivity tests
│   └── utils/
│       ├── schemas.py             # Pydantic v2 schemas for tasks and predictions
│       └── env_check.py           # Hardware/dependency sanity auditor
├── docker/
│   ├── Dockerfile.gpu             # CUDA 12.x / PyTorch container specification
│   └── docker-compose.yml         # Local vLLM/Laya runner orchestration
├── scripts/
│   ├── run_full_suite.sh          # End-to-end execution script
│   └── export_reports.py          # Markdown/LaTeX/JSON artifact exporter
├── pyproject.toml
└── README.md
```

---

## 2. FORMAL BENCHMARK CATEGORIES & TASK SUITES

Implement standardized task definitions supporting the three core decision primitives:
- `choice`: Multi-class categorical selection with probability distributions.
- `score`: Ordinal/rubric evaluation with monotonic scale properties.
- `noul`: Binary hypothesis validation ($P(\text{true}) \in [0, 1]$).

Implement 5 core benchmark suites:

### Suite A: Probability Calibration & Selective Prediction
*   **Goal:** Validate whether confidence scores correspond to true empirical accuracy.
*   **Datasets:**
    *   [CALIBRATION_DATASET_CHOICE, e.g., Banking77, CLINC150, or AG News]
    *   [CALIBRATION_DATASET_NOUL, e.g., ToxiGen, Prompt Injection / Guardrail sets]
    *   [CALIBRATION_DATASET_SCORE, e.g., Customer Support Severity (0–4)]
*   **Metrics:** 
    *   Expected Calibration Error (ECE) with equal-frequency and equal-width binning ($M = 15$).
    *   Maximum Calibration Error (MCE).
    *   Brier Score: $\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$.
    *   Area Under Risk-Coverage Curve (AURC) for selective abstention thresholds $\tau \in [0.5, 0.99]$.

### Suite B: Cardinality & Complexity Scaling
*   **Goal:** Evaluate degradation in accuracy and calibration as $|C_k|$ grows.
*   **Test Matrices:**
    *   Cardinality scale: $K \in \{2, 5, 10, 25, 50, 77, 150\}$ options.
    *   State length scale: Token sequence length $L \in \{128, 512, 1024, 2048, 4096, 8192\}$.
    *   Multi-Question batching: $Q \in \{1, 5, 10, 20\}$ simultaneous evaluations evaluated against identical state.

### Suite C: State Invariance & Robustness
*   **Goal:** Measure vulnerability to input perturbations without semantic shifts.
*   **Scenarios:**
    1.  *Permutation Invariance:* Measure Jensen-Shannon Divergence ($D_{JS}$) between predictions under original vs. randomly shuffled choice orders.
    2.  *Distractor Resistance:* Inject irrelevant structured JSON keys and unstructured prose logs into the state without altering ground truth.
    3.  *Out-of-Distribution (OOD) / Negative Detection:* Input queries where no candidate option is correct. Measure prediction entropy vs. calibration over-confidence.

### Suite D: Latency, Throughput & Economic Profiling
*   **Goal:** Quantify the computational efficiency advantage of System 1 forward passes.
*   **Measurements:**
    *   Latency percentiles: p50, p90, p95, p99 across batch sizes $B \in \{1, 8, 32, 64\}$.
    *   Throughput: Decisions/second (DPS).
    *   Cost per $100{,}000$ decisions ($) based on hosted token pricing vs. self-hosted GPU amortization.

---

## 3. BASELINE MODELS & REFERENCE IMPLEMENTATIONS

Implement adapters in `src/adapters/` with uniform input/output schemas:

1.  **Evaluated System 1 Models:**
    *   `TypeSafeJevAdapter`: Hosted API integration with rate-limiting, retries, and token-cost tracking.
    *   `ConvAILayaAdapter`: Local PyTorch/Hugging Face pipeline loaded via `ModernBERT-large` and `mmBERT-base`.
2.  **Baselines for Direct Comparison:**
    *   *Constrained Autoregressive SOTA:* `vLLM` + `Outlines` (or `XGrammar`) running [BASELINE_MODEL_SMALL, e.g., Qwen2.5-3B-Instruct / Llama-3.2-3B-Instruct] constrained to the exact JSON schema / regex grammar.
    *   *Zero-Shot Open Encoder:* DeBERTa-v3-large fine-tuned on MNLI / NLI zero-shot classification.
    *   *Empirical Majority Baseline:* Predicts dataset class priors (sanity-check baseline).

---

## 4. QUANTITATIVE SCORING RUBRICS & THRESHOLDS

Define a machine-readable validation suite that grades candidate models across 5 tiers:

| Dimension | Critical Failure | Substandard | Acceptable | Production Grade | SOTA / Superior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ECE ($M=15$)** | $> 0.20$ | $0.12 - 0.20$ | $0.06 - 0.12$ | $0.02 - 0.06$ | $< 0.02$ |
| **Permutation $D_{JS}$** | $> 0.15$ | $0.08 - 0.15$ | $0.03 - 0.08$ | $0.005 - 0.03$ | $< 0.005$ |
| **Latency p95 ($B=1$)** | $> 350\text{ ms}$ | $150 - 350\text{ ms}$ | $50 - 150\text{ ms}$ | $15 - 50\text{ ms}$ | $< 15\text{ ms}$ |
| **AURC (Risk $\le 5\%$)** | Coverage $< 50\%$ | Coverage $50-65\%$ | Coverage $65-80\%$ | Coverage $80-90\%$ | Coverage $> 90\%$ |
| **OOD Entropy Spike** | No shift ($z < 1.0$) | Low shift ($1.0 \le z < 2.0$) | Moderate ($2.0 \le z < 3.5$) | Sharp ($z \ge 3.5$) | Near Uniform Distribution |

---

## 5. REPRODUCIBILITY, ENVIRONMENT & CONTAINERIZATION

Ensure deterministic, production-grade test runs:

1.  **Environment Pinning:**
    *   OS: Ubuntu 22.04 LTS (x86_64).
    *   CUDA: 12.4 | Python: 3.11.8 | PyTorch: 2.4.x | Hugging Face Transformers: >= 4.45.0.
    *   Hardware Minimum: 1x NVIDIA A100 (80GB SXM4) or 1x RTX 4090 (24GB).
2.  **RNG & Seed Control:**
    *   Pin seeds globally across Python `random`, `numpy`, `torch.manual_seed`, and CUDA deterministic flags:
        ```python
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        ```
    *   Default seeds: `[42, 1337, 2026, 80085, 99999]`. All experiments must run across minimum **$N=5$ seeds**.
3.  **Dockerization:** Provide a production-grade multi-stage `Dockerfile.gpu` with isolated non-root user execution, model weight volume caching, and pre-compiled FlashAttention-2.

---

## 6. DATA MANIFEST SPECIFICATION

All test cases must parse into strict Pydantic models. Validate every record against this schema:

```json
{
  "task_id": "routing_telecom_0042",
  "primitive": "choice",
  "state": "Customer: 'I got charged $45 twice on my card ending in 4021 for the roaming plan.' Context: Billing cycle active, autopay enabled.",
  "options": [
    {"key": "dispute_charge", "description": "Duplicate billing or disputed line items"},
    {"key": "cancel_service", "description": "Terminating account or line"},
    {"key": "plan_upgrade", "description": "Adding features or increasing tiers"},
    {"key": "technical_support", "description": "Network connectivity issues"}
  ],
  "ground_truth": "dispute_charge",
  "metadata": {
    "difficulty": "medium",
    "domain": "telecom_billing",
    "is_adversarial": false,
    "input_tokens": 48
  }
}
```

---

## 7. EXECUTION ENGINE & FAILURE MODES

The harness execution engine must explicitly test and report on:
1.  **Rate Limiting & Transient Network Faults:** Implement exponential backoff with jitter for remote APIs (`TypeSafeJev`).
2.  **Length Truncation:** Gracefully trap inputs exceeding context windows without dropping batch execution.
3.  **Malformed Outputs:** Trap non-conforming JSON schemas, NaN probability vectors, or probabilities not summing to $1.0 \pm 10^{-4}$.
4.  **Schema Rejection Handling:** Track the percentage of failed model queries as a first-class metric (`schema_failure_rate`).

---

## 8. ABLATION STUDY SUITE

Implement an automated parameter sweep engine in `src/runners/ablations.py` to systematically isolate:
1.  **State Context Window:** $L \in [256, 512, 1024, 2048, 4096]$ tokens.
2.  **Label Set Description Granularity:**
    *   Variant A: Label string only (e.g., `dispute_charge`).
    *   Variant B: Label + 1-sentence rubric (e.g., `dispute_charge: Duplicate billing or disputed line items`).
    *   Variant C: Label + negative definitions (e.g., `dispute_charge: ... Do not use for cancellations`).
3.  **Batch Concurrency:** Batch sizes $B \in [1, 4, 16, 64, 128]$.

---

## 9. AGGREGATION, STATISTICAL TESTING & OUTPUTS

1.  **Statistical Aggregation:**
    *   Aggregate metric scores across $N=5$ seed runs reporting: Mean, Median, Standard Deviation, and 95% Bootstrap Confidence Intervals (10,000 resamples).
    *   Use two-sided Wilcoxon signed-rank tests when comparing models; reject the null hypothesis of equivalence only when $p < 0.01$.
2.  **Missing Data & Outlier Rules:**
    *   If a remote API drops $< 1\%$ of runs, impute with worst-case performance ($p_{correct} = 0, \text{latency} = \text{timeout\_val}$).
    *   If drops $\ge 1\%$, mark run invalid and halt pipeline.
3.  **Export Artifacts:**
    *   `results/summary.json`: Complete hierarchical metric payload.
    *   `results/table_summary.md`: Clean Markdown comparison table.
    *   `results/latex_table.tex`: Camera-ready LaTeX table for academic/technical whitepapers.

---

## 10. BENCHMARK META-EVALUATION (AUDITING THE BENCHMARK)

Implement validation functions in `src/analysis/meta_eval.py` to prevent benchmark gaming or contamination:
1.  **Label Order Bias Audit:** Verify whether model choices correlate with positional indices (e.g., preference for index 0). Flag models with Chi-square test $p < 0.05$.
2.  **Surface Form Leakage:** Ensure test questions do not contain high word-overlap/n-gram leakage with target label names (e.g., token "dispute" directly occurring in the prompt text).
3.  **Short-Circuit Testing:** Run models on **state-only** (options stripped) and **options-only** (state stripped). If state-only or options-only accuracy exceeds random baseline $+ 15\%$, flag the dataset split for spurious correlation.

---

## INSTRUCTIONS FOR GENERATING THE CODEBASE

1. Generate fully functioning, type-annotated Python code (using `pydantic`, `numpy`, `scipy`, `pandas`, `torch`, `transformers`, `httpx`).
2. Write clean, self-documenting code with clear docstrings and error handling.
3. Replace all bracketed placeholders `[LIKE_THIS]` with sensible production defaults if not explicitly overridden.
4. Begin by scaffolding the core architecture, Pydantic schemas, and mathematical metric calculations.