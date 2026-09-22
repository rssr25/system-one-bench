#!/usr/bin/env bash
# Suites C, D, E for one model config. Usage: scripts/run_sweeps.sh configs/models/laya_en.yaml results/laya_en_sweeps [tickets manifest]
# Env: SYS1BENCH (command override), CONC (concurrency), N (items per sweep level), BUDGET=1 to also run the Laya option-budget sweep.
set -euo pipefail
S1B=${SYS1BENCH:-sys1bench}
CFG=${1:?model config}; OUT=${2:?out dir}; TICKETS=${3:-}
N=${N:-150}; CONC=${CONC:-2}
mkdir -p "$OUT"
[ -n "$TICKETS" ] || { TICKETS="$OUT/tickets.jsonl"; $S1B generate support_tickets "$TICKETS" --n 300 --seed 42; }
$S1B sweep-cardinality "$OUT/cardinality_rag.json" --config "$CFG" --generator rag_relevance --question-key best_passage \
  --ks 2,5,10,20,50,100,255 --n "$N" --cache "$OUT/cache_card.sqlite" --concurrency "$CONC" > "$OUT/cardinality_rag.log" 2>&1
$S1B sweep-length "$OUT/length_tickets.json" --config "$CFG" --lengths 128,256,512,1024,2048,4096,8192 --n 100 \
  --cache "$OUT/cache_len.sqlite" --concurrency "$CONC" > "$OUT/length_tickets.log" 2>&1
$S1B interference "$TICKETS" "$OUT/interference_queue.json" --config "$CFG" --target queue --qs 2,5,10,20 --limit "$N" \
  --cache "$OUT/cache_interf.sqlite" --concurrency "$CONC" > "$OUT/interference_queue.log" 2>&1
$S1B interference "$TICKETS" "$OUT/interference_priority.json" --config "$CFG" --target priority --qs 2,5,10,20 --limit "$N" \
  --cache "$OUT/cache_interf.sqlite" --concurrency "$CONC" > "$OUT/interference_priority.log" 2>&1
$S1B robustness "$OUT/robustness_tickets.json" --config "$CFG" --n 200 --cache "$OUT/cache_robust.sqlite" --concurrency "$CONC" > "$OUT/robustness_tickets.log" 2>&1
if [ "${BUDGET:-0}" = "1" ]; then
  CKPT=$(grep -E '^checkpoint:' "$CFG" | awk '{print $2}')
  $S1B sweep-budget "$OUT/budget_K12.json" --budgets 64,128,192,256,384,512 --k 12 --n 200 --checkpoint "${CKPT:-english}" --max-len 1024 \
    --cache "$OUT/cache_budget.sqlite" > "$OUT/budget_K12.log" 2>&1
fi
echo "sweeps done: $OUT"
