#!/usr/bin/env bash
# Suite A (calibration + selective prediction) end to end for one model config.
# Usage: scripts/run_suite_A.sh configs/models/jev_1.13.yaml results/jev
set -euo pipefail
CFG=${1:?model config yaml}; OUT=${2:?output dir}; N=${N:-2000}
mkdir -p "$OUT"
sys1bench generate support_tickets "$OUT/tickets.jsonl" --n "$N" --seed 42 --rules-out "$OUT/tickets_rules.yaml"
sys1bench generate support_tickets "$OUT/tickets_unknowable.jsonl" --n 500 --seed 43 --unknowable-frac 1.0
sys1bench generate support_tickets "$OUT/tickets_noisy.jsonl" --n 500 --seed 44 --label-noise 0.05
sys1bench generate phishing_email "$OUT/phish.jsonl" --n "$N" --seed 42
for m in tickets phish; do
  sys1bench run "$OUT/$m.jsonl" "$OUT/preds_$m.jsonl" --config "$CFG" --framings "data/framings/${m/phish/phishing_email}.yaml" \
    --permutations 5 --corruption --short-circuit --concurrency "${CONC:-8}" --cache "$OUT/cache.sqlite" --suite A
done
sys1bench run "$OUT/tickets_unknowable.jsonl" "$OUT/preds_unknowable.jsonl" --config "$CFG" --cache "$OUT/cache.sqlite" --suite A
sys1bench run "$OUT/tickets_noisy.jsonl" "$OUT/preds_noisy.jsonl" --config "$CFG" --cache "$OUT/cache.sqlite" --suite A
sys1bench score "$OUT"/preds_tickets.jsonl "$OUT"/preds_phish.jsonl --out "$OUT/summary.json" --table "$OUT/table.md"
