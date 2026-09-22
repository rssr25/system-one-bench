#!/usr/bin/env bash
# Daily drift check for hosted models. Appends one line per run to results/canary/history.jsonl and redraws the drift
# curve. Cron example (09:00 daily):  0 9 * * * cd /path/to/system-one-bench && scripts/canary_daily.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p results/canary
for cfg in configs/models/jev_1.13.yaml; do
  name=$(basename "$cfg" .yaml)
  out=$(sys1bench canary jev_typesafe --config "$cfg" --store results/canary 2>&1 | tail -n 40)
  code=$?
  python3 - "$name" "$code" <<PY
import json, sys, time, re
name, code = sys.argv[1], int(sys.argv[2])
raw = """$out"""
m = re.search(r"\{.*\}", raw, re.S)
rec = json.loads(m.group(0)) if m else {"error": raw[-300:]}
rec.update({"config": name, "date": time.strftime("%Y-%m-%d"), "exit_code": code})
open("results/canary/history.jsonl", "a").write(json.dumps(rec) + "\n")
print(json.dumps(rec))
PY
done
python3 - <<'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("results/canary/history.jsonl").read_text().splitlines() if l.strip()]
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for cfg in sorted({r["config"] for r in rows}):
        rs = [r for r in rows if r["config"] == cfg and r.get("mean_abs_dp") is not None]
        ax.plot([r["date"] for r in rs], [r["mean_abs_dp"] for r in rs], marker="o", label=f"{cfg} mean |dp|")
        ax.plot([r["date"] for r in rs], [r["argmax_flip_rate"] for r in rs], marker="s", ls="--", label=f"{cfg} flip rate")
    ax.axhline(0.01, color="#c33", lw=1, ls=":", label="alarm 0.01")
    ax.set_ylabel("drift vs last accepted"); ax.legend(fontsize=7, frameon=False); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.tight_layout(); fig.savefig("results/canary/drift.png", dpi=150)
    print("drift curve: results/canary/drift.png")
except ImportError:
    print("matplotlib not installed; history in results/canary/history.jsonl")
PY
