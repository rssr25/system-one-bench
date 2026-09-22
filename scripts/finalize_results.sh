#!/usr/bin/env bash
# Rebuild predictions from caches with the current parsers, then render report + plots.
# Usage: scripts/finalize_results.sh [results_root] [docs copy name]
set -euo pipefail
ROOT=${1:-results}; DOC=${2:-docs/RESULTS_$(date +%F).md}
[ -d "$ROOT/jev" ] && CONC=4 scripts/rebuild_preds.sh configs/models/jev_1.13.yaml "$ROOT/jev" 3 > /dev/null
sys1bench report "$ROOT" --out "$ROOT/REPORT.md" --title "sys1bench results ($(date +%F))"
sys1bench plots "$ROOT" --out "$ROOT/plots"
cp "$ROOT/REPORT.md" "$DOC"
echo "report: $ROOT/REPORT.md (copied to $DOC); plots: $ROOT/plots"
