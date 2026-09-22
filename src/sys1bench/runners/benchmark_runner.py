"""Main evaluation loop: manifest rows -> DecisionRequests -> adapter -> PredictionRows (one per question).

Failures never crash a run: transport and schema failures become rows with `error` set, which the report
layer turns into `transport_failure_rate` and `schema_failure_rate`.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..adapters.base import BaseAdapter
from ..schemas import DecisionRequest, PredictionRow, RequestMeta, TaskItem
from .cache import ResponseCache


def load_manifest(path: str | Path) -> list[TaskItem]:
    return [TaskItem.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_manifest(items: Iterable[TaskItem], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def write_predictions(rows: Iterable[PredictionRow], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


def load_predictions(path: str | Path) -> list[PredictionRow]:
    return [PredictionRow.model_validate_json(l) for l in Path(path).read_text().splitlines() if l.strip()]


def to_request(item: TaskItem, suite: str | None = None, arm: str | None = None) -> DecisionRequest:
    return DecisionRequest(
        state=item.state, questions=item.questions,
        meta=RequestMeta(task_id=item.task_id, row_id=item.row_id(), permutation_id=item.permutation_id,
                         framing_ids={k: q.framing_id for k, q in item.questions.items()}, suite=suite, arm=arm),
    )


def _truth_key(q) -> str | None:
    if q.ground_truth is None:
        return None
    return str(q.ground_truth).lower() if q.type == "noul" else str(q.ground_truth)


def flatten(item: TaskItem, resp, adapter: BaseAdapter, suite: str | None, arm: str | None) -> list[PredictionRow]:
    rows = []
    n_q = max(1, len(item.questions))
    for k, q in item.questions.items():
        a = resp.answers.get(k)
        truth = _truth_key(q)
        if a is None:
            continue
        err = a.error or (f"transport:{resp.transport_error}" if resp.transport_error else None)
        correct = (a.argmax == truth) if (truth is not None and err is None) else None
        rows.append(PredictionRow(
            row_id=item.row_id(), task_id=item.task_id, question_key=k, primitive=q.type, tier=item.tier,
            domain=item.domain, language=item.language, framing_id=q.framing_id, framing_group=q.framing_group,
            permutation_id=item.permutation_id, cardinality=q.cardinality, option_keys=a.option_keys, probs=a.probs,
            argmax=a.argmax, ground_truth=truth, correct=correct, confidence=a.confidence, abstained=a.abstained,
            truncated=a.truncated, error=err, quantisation_step=a.quantisation_step, controls=item.controls,
            latency_ms=resp.latency.client_ms, cost_usd=(resp.provider.cost_usd / n_q) if resp.provider.cost_usd is not None else None,
            adapter_id=adapter.adapter_id, model_id=resp.provider.model_id_returned or adapter.model_id,
            version_hash=resp.provider.version_hash, suite=suite, arm=arm,
            metadata={"decomposition_of": q.decomposition_of, "capability_issues": adapter.check_request(to_request(item)),
                      "generation_id": resp.provider.generation_id, "route": resp.provider.route},
        ))
    return rows


def run_items(items: list[TaskItem], adapter: BaseAdapter, cache: ResponseCache | None = None, suite: str | None = None,
              arm: str | None = None, concurrency: int = 1, progress: bool = False) -> list[PredictionRow]:
    def one(item: TaskItem):
        req = to_request(item, suite, arm)
        key = req.cache_key(adapter.adapter_id, adapter.model_id, adapter.tunables)
        resp = cache.get(key) if cache is not None else None
        if resp is None:
            try:
                resp = adapter.decide(req)
            except Exception as e:  # adapter bug or hard failure: record, don't crash
                from ..schemas import Answer, DecisionResponse, LatencyRecord, ProviderRecord

                resp = DecisionResponse(answers={k: Answer.failed(q, "adapter_exception") for k, q in item.questions.items()},
                                        latency=LatencyRecord(client_ms=float("nan")),
                                        provider=ProviderRecord(adapter_id=adapter.adapter_id, model_id_requested=adapter.model_id),
                                        transport_error=repr(e))
            if cache is not None and resp.transport_error is None:
                cache.put(key, adapter.adapter_id, adapter.model_id, resp)
        return flatten(item, resp, adapter, suite, arm)

    t0 = time.time()
    rows: list[PredictionRow] = []
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for i, r in enumerate(ex.map(one, items)):
                rows.extend(r)
                if progress and i % 100 == 0:
                    print(f"  {i}/{len(items)}", flush=True)
    else:
        for i, it in enumerate(items):
            rows.extend(one(it))
            if progress and i % 100 == 0:
                print(f"  {i}/{len(items)}", flush=True)
    wall = time.time() - t0
    for r in rows:
        r.metadata["wall_seconds_total"] = wall
        r.metadata["concurrency"] = concurrency
    return rows
