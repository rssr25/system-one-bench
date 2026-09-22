"""Convai Laya, run locally (Apache 2.0) through the `laya` package's Router (verified against laya 0.3.4).

    from laya import Router
    router = Router(preload=True, device="cuda")
    out = router.predict(state, questions, model=None | "english" | "multilingual" | "typed-decisions")

Question shape matches TypeSafe's: choice criteria is {option: description}, score criteria is an ordered list of
level descriptions, noul has no criteria. Answers:
    choice -> {"type","choice","probabilities":{opt: p},"confidence","action":{"act_probability"}}
    score  -> {"type","score","legend","probabilities":{"0": p,...},"confidence","action":{...}}
    noul   -> {"type","noul": P(yes),"confidence","action":{...}}
`action.act_probability` is Laya's act/escalate head; we treat act_probability < 0.5 as a native abstention.
Result also carries "routing": {"model","repo","reason",...} and "usage".

Tunables: `head_max_len` and `max_len` are set on the checkpoint's `agent.cfg` before inference (Suite C ablates them;
the 77-label collapse at the default budget is a configuration effect). Checkpoint defaults: english 512/192,
multilingual 1024/256, typed-decisions 1024/256. The package prints a warning and falls back to CPU when CUDA is
unavailable or out of memory; we record the device actually used in `hardware` so latency tables stay honest.
"""

from __future__ import annotations

import platform
import time
from typing import Any

from ..schemas import Answer, DecisionRequest, DecisionResponse, LatencyRecord, ModelCapabilities, ProviderRecord, Question
from .base import BaseAdapter, FatalAdapterError, finalize_answer, register

CHECKPOINTS = {"english": ("convaiinnovations/laya", 512, 192), "multilingual": ("convaiinnovations/laya/multilingual", 1024, 256),
               "typed-decisions": ("convaiinnovations/laya/typed-decisions", 1024, 256), "router": ("convaiinnovations/laya", 1024, 256)}
ALIASES = {"laya": "english", "laya-multilingual": "multilingual", "laya-typed-decisions": "typed-decisions"}


def to_vendor_question(q: Question) -> dict[str, Any]:
    if q.type == "noul":
        return {"type": "noul", "instructions": q.instructions}
    if q.type == "choice":
        crit = {c.key: (c.description or c.key) for c in q.criteria}  # type: ignore[union-attr]
        if q.allow_abstain:
            crit[q.abstain_key] = "None of the listed options applies."
        return {"type": "choice", "instructions": q.instructions, "criteria": crit}
    return {"type": "score", "instructions": q.instructions,
            "criteria": [(c.description or f"level {c.level}") for c in q.criteria]}  # type: ignore[union-attr]


def parse_answer(q: Question, payload: dict[str, Any], abstain_threshold: float = 0.5) -> Answer:
    keys = q.option_keys
    conf = payload.get("confidence")
    act = (payload.get("action") or {}).get("act_probability")
    abstained = act is not None and float(act) < abstain_threshold
    if q.type == "noul":
        p = payload.get("noul")
        if p is None:
            return Answer.failed(q, "no noul field")
        a = finalize_answer(q, [float(p), 1.0 - float(p)], confidence=None, abstained=abstained)
    else:
        dist = payload.get("probabilities")
        if not isinstance(dist, dict):
            return Answer.failed(q, "no probabilities map")
        if q.type == "score":
            probs = [float(dist.get(str(i), 0.0)) for i in range(len(keys))]
        else:
            probs = [float(dist.get(k, 0.0)) for k in keys]
            if q.allow_abstain and q.abstain_key in dist:
                abstained = abstained or float(dist[q.abstain_key]) >= max(probs)
                s = sum(probs)
                probs = [x / s for x in probs] if s > 0 else [1.0 / len(keys)] * len(keys)
        a = finalize_answer(q, probs, confidence=conf, abstained=abstained)
    if a.ok and act is not None:
        a = a.model_copy(update={"raw_prob_sum": a.raw_prob_sum})
    return a


@register("laya_local")
class LayaLocalAdapter(BaseAdapter):
    def __init__(self, model_id: str = "convaiinnovations/laya", checkpoint: str = "english", device: str = "cuda",
                 head_max_len: int | None = None, max_len: int | None = None, batch_size: int = 1,
                 hardware: str | None = None, abstain_threshold: float = 0.5, strict_device: bool = True, **tunables) -> None:
        super().__init__(model_id, **tunables)
        checkpoint = ALIASES.get(checkpoint, checkpoint)
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"unknown Laya checkpoint {checkpoint!r}; known: {sorted(CHECKPOINTS)} (+ aliases {sorted(ALIASES)})")
        self.checkpoint, self.device, self.batch_size, self.abstain_threshold = checkpoint, device, batch_size, abstain_threshold
        self.strict_device = strict_device
        _, dflt_max, dflt_head = CHECKPOINTS[checkpoint]
        self.head_max_len = head_max_len or dflt_head
        self.max_len = max_len or dflt_max
        self.hardware = hardware
        self.tunables.update({"head_max_len": self.head_max_len, "max_len": self.max_len, "checkpoint": checkpoint})
        self._router = None

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=f"laya:{self.checkpoint}", deployment="local", max_options=255,
                                 max_state_tokens=self.max_len - self.head_max_len, emits_confidence=True,
                                 supports_native_abstain=True, supports_batching=True,
                                 languages={"en"} if self.checkpoint == "english" else None,
                                 tunables={"head_max_len": self.head_max_len, "max_len": self.max_len})

    def _load(self):
        if self._router is not None:
            return self._router
        try:
            from laya import Router  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("install Laya: pip install laya (github.com/NandhaKishorM/laya)") from e
        # Lazy router: load only the checkpoint we evaluate (preloading all three needs ~6 GB and, on a shared GPU,
        # silently lands on CPU). "router" mode keeps the full preload because it dispatches per language.
        if self.checkpoint == "router":
            router = Router(preload=True, device=self.device)
            agents = list((getattr(router, "_agents", {}) or {}).values())
        else:
            router = Router(preload=False, device=self.device)
            agents = [router.load(self.checkpoint)]
        for ag in agents:
            if hasattr(ag, "cfg"):
                ag.cfg["head_max_len"] = self.head_max_len
                ag.cfg["max_len"] = self.max_len
            dev = str(getattr(ag, "device", ""))
            if self.device.startswith("cuda") and "cuda" not in dev and hasattr(ag, "model"):
                try:  # the package falls back to CPU when CUDA context creation fails at import; retry the move
                    import torch  # type: ignore

                    ag.model.to("cuda")
                    ag.device = torch.device("cuda") if hasattr(torch, "device") else "cuda"
                except Exception:
                    pass
            dev = str(getattr(ag, "device", ""))
            if self.device.startswith("cuda") and "cuda" not in dev and self.strict_device:
                del router, agents  # release the half-loaded model before aborting
                raise FatalAdapterError(f"Laya checkpoint {self.checkpoint!r} loaded on {dev or 'cpu'} although device={self.device!r} was requested; "
                                        "refusing to record CPU latency as GPU. Free GPU memory or pass strict_device=false.")
        ag = agents[0] if agents else None
        dev = str(getattr(ag, "device", self.device)) if ag is not None else self.device
        if self.hardware is None:
            gpu = None
            try:
                import torch  # type: ignore

                if "cuda" in dev and torch.cuda.is_available():
                    gpu = torch.cuda.get_device_name(0)
            except Exception:  # pragma: no cover
                pass
            self.hardware = f"{gpu} ({dev})" if gpu else f"CPU {platform.machine()} ({dev})"
        self._router = router  # only after every check passed
        return self._router

    def parse_raw_answer(self, q: Question, payload: dict) -> Answer:
        return parse_answer(q, payload, self.abstain_threshold)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        router = self._load()
        vq = {k: to_vendor_question(q) for k, q in request.questions.items()}
        model = None if self.checkpoint == "router" else self.checkpoint
        t0 = time.perf_counter()
        try:
            out = router.predict(request.state, vq, model=model)
        except Exception as e:
            # laya raises ValueError("... options exceed head_max_len=N") when the option strings do not fit the budget:
            # that is the model refusing the request, not a harness bug, so it is classified like a vendor 4xx.
            kind = "rejected_by_model" if isinstance(e, ValueError) else "adapter_exception"
            return DecisionResponse(
                answers={k: Answer.failed(q, kind) for k, q in request.questions.items()},
                latency=LatencyRecord(client_ms=float("nan"), timestamp=time.time()),
                provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, hardware=self.hardware),
                transport_error=repr(e)[:500])
        ms = (time.perf_counter() - t0) * 1000
        raw_answers = out.get("answers", {}) if isinstance(out, dict) else {}
        # laya truncates silently to max_len - head_max_len state tokens and reports nothing; estimate it (~4 chars/token)
        est_tokens = len(request.state_text()) / 4
        truncated = bool(out.get("truncated", False)) if isinstance(out, dict) else False
        truncated = truncated or est_tokens > (self.max_len - self.head_max_len)
        answers = {}
        for k, q in request.questions.items():
            a = parse_answer(q, raw_answers[k], self.abstain_threshold) if isinstance(raw_answers.get(k), dict) else Answer.failed(q, "missing answer")
            if a.ok and truncated:
                a = a.model_copy(update={"truncated": True})
            answers[k] = a
        routing = out.get("routing", {}) if isinstance(out, dict) else {}
        usage = out.get("usage", {}) if isinstance(out, dict) else {}
        return DecisionResponse(
            answers=answers,
            latency=LatencyRecord(client_ms=ms, compute_ms=ms, batch_size=self.batch_size, timestamp=time.time()),
            provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id,
                                    model_id_returned=f"laya:{routing.get('model', self.checkpoint)}",
                                    version_hash=f"{routing.get('repo', '')}|head{self.head_max_len}|max{self.max_len}",
                                    hardware=self.hardware, billed_input_tokens=usage.get("input_tokens"), cost_usd=0.0),
            raw=out if isinstance(out, dict) else {"out": str(out)},
        )
