#!/usr/bin/env bash
# Suite A (calibration + selective prediction) end to end for one model config.
# Usage: scripts/run_suite_A.sh configs/models/jev_1.13.yaml results/jev
set -euo pipefail
S1B=${SYS1BENCH:-sys1bench}   # override e.g. SYS1BENCH="env PYTHONPATH=src /path/to/venv/bin/python -m sys1bench.cli"
CFG=${1:?model config yaml}; OUT=${2:?output dir}; N=${N:-2000}
mkdir -p "$OUT"
$S1B generate support_tickets "$OUT/tickets.jsonl" --n "$N" --seed 42 --rules-out "$OUT/tickets_rules.yaml"
$S1B generate support_tickets "$OUT/tickets_unknowable.jsonl" --n 500 --seed 43 --unknowable-frac 1.0
$S1B generate support_tickets "$OUT/tickets_noisy.jsonl" --n 500 --seed 44 --label-noise 0.05
$S1B generate phishing_email "$OUT/phish.jsonl" --n "$N" --seed 42
declare -A FR=([tickets]=support_tickets [phish]=phishing_email)
for m in tickets phish; do
  $S1B run "$OUT/$m.jsonl" "$OUT/preds_$m.jsonl" --config "$CFG" --framings "data/framings/${FR[$m]}.yaml" \
    --permutations "${PERMS:-5}" --corruption --short-circuit --concurrency "${CONC:-4}" --cache "$OUT/cache.sqlite" --suite A
done
$S1B run "$OUT/tickets_unknowable.jsonl" "$OUT/preds_unknowable.jsonl" --config "$CFG" --cache "$OUT/cache.sqlite" --suite A
$S1B run "$OUT/tickets_noisy.jsonl" "$OUT/preds_noisy.jsonl" --config "$CFG" --cache "$OUT/cache.sqlite" --suite A
$S1B score "$OUT"/preds_tickets.jsonl "$OUT"/preds_phish.jsonl --out "$OUT/summary.json" --table "$OUT/table.md"
