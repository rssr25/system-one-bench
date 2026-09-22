"""Drift canary for hosted models: a fixed item set run at the start of every session and compared to the
last accepted run. Alarm thresholds: mean |dp| > 0.01 or argmax disagreement > 1%."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..adapters.base import BaseAdapter
from ..schemas import TaskItem
from .benchmark_runner import run_items


def run_canary(items: list[TaskItem], adapter: BaseAdapter, store: str | Path, prob_tol: float = 0.01,
               flip_tol: float = 0.01) -> dict:
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    rows = run_items(items, adapter)
    current = {f"{r.row_id}|{r.question_key}": {"probs": r.probs, "argmax": r.argmax, "model": r.model_id} for r in rows}
    prev_path = store / f"{adapter.adapter_id}__{adapter.model_id.replace('/', '_')}.json"
    result = {"n": len(current), "timestamp": time.time(), "model_id_returned": rows[0].model_id if rows else None,
              "drift_suspected": False, "mean_abs_dp": None, "argmax_flip_rate": None, "first_run": not prev_path.exists()}
    if prev_path.exists():
        prev = json.loads(prev_path.read_text())["answers"]
        dps, flips = [], []
        for k, cur in current.items():
            if k in prev and len(prev[k]["probs"]) == len(cur["probs"]):
                dps.append(np.abs(np.array(prev[k]["probs"]) - np.array(cur["probs"])).mean())
                flips.append(prev[k]["argmax"] != cur["argmax"])
        if dps:
            result["mean_abs_dp"] = float(np.mean(dps))
            result["argmax_flip_rate"] = float(np.mean(flips))
            result["drift_suspected"] = result["mean_abs_dp"] > prob_tol or result["argmax_flip_rate"] > flip_tol
            result["model_id_changed"] = json.loads(prev_path.read_text()).get("model_id_returned") != result["model_id_returned"]
    if not result["drift_suspected"]:
        prev_path.write_text(json.dumps({"answers": current, "timestamp": result["timestamp"],
                                         "model_id_returned": result["model_id_returned"]}, indent=1))
    else:
        (store / f"drift_{int(result['timestamp'])}.json").write_text(json.dumps({"answers": current, **result}, indent=1))
    return result
