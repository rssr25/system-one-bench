#!/usr/bin/env bash
# Regenerate prediction files for an existing Suite A results dir from the cache (no manifest regeneration, so old
# manifests keep pairing with cached responses; parser changes are applied via adapter.reparse on cache hits).
# Usage: scripts/rebuild_preds.sh configs/models/jev_1.13.yaml results/jev [PERMS]
set -euo pipefail
S1B=${SYS1BENCH:-sys1bench}
CFG=${1:?model config yaml}; OUT=${2:?results dir}; PERMS=${3:-${PERMS:-3}}
declare -A FR=([tickets]=support_tickets [phish]=phishing_email)
for m in tickets phish; do
  [ -f "$OUT/$m.jsonl" ] || continue
  $S1B run "$OUT/$m.jsonl" "$OUT/preds_$m.jsonl" --config "$CFG" --framings "${FR[$m]}" \
    --permutations "$PERMS" --corruption --short-circuit --concurrency "${CONC:-4}" --cache "$OUT/cache.sqlite" --suite A
done
[ -f "$OUT/tickets_unknowable.jsonl" ] && $S1B run "$OUT/tickets_unknowable.jsonl" "$OUT/preds_unknowable.jsonl" --config "$CFG" --cache "$OUT/cache.sqlite" --suite A
[ -f "$OUT/tickets_noisy.jsonl" ] && $S1B run "$OUT/tickets_noisy.jsonl" "$OUT/preds_noisy.jsonl" --config "$CFG" --cache "$OUT/cache.sqlite" --suite A
$S1B score "$OUT"/preds_tickets.jsonl "$OUT"/preds_phish.jsonl --out "$OUT/summary.json" --table "$OUT/table.md" > /dev/null
echo "rebuilt $OUT"
