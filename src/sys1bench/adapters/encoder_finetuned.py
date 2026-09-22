"""Fine-tuned encoder baseline: a small classifier (DistilBERT by default) trained per question on a *disjoint* seed of
the same generator, then evaluated on the benchmark manifest. This is the "strongest cheap baseline" from jevbench:
it has seen thousands of labelled examples where the System One models see only descriptions, and the report says so.

    adapter: encoder_finetuned
    model_id: distilbert-base-uncased
    train_generator: support_tickets
    train_n: 4000
    train_seed: 7                      # never the benchmark seed
    epochs: 2
Trains one head per question key at first use (cached under ~/.cache/sys1bench/encoders/<hash>).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..schemas import Answer, DecisionRequest, DecisionResponse, LatencyRecord, ModelCapabilities, ProviderRecord
from .base import BaseAdapter, finalize_answer, register


@register("encoder_finetuned")
class EncoderFinetunedAdapter(BaseAdapter):
    def __init__(self, model_id: str = "distilbert-base-uncased", train_generator: str = "support_tickets", train_n: int = 4000,
                 train_seed: int = 7, epochs: int = 2, batch_size: int = 32, lr: float = 5e-5, max_length: int = 256,
                 device: str | None = None, cache_dir: str | None = None, **tunables) -> None:
        super().__init__(model_id, **tunables)
        self.train_generator, self.train_n, self.train_seed, self.epochs = train_generator, train_n, train_seed, epochs
        self.batch_size, self.lr, self.max_length, self.device = batch_size, lr, max_length, device
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "sys1bench" / "encoders")
        self.tunables.update({"train_generator": train_generator, "train_n": train_n, "train_seed": train_seed, "epochs": epochs})
        self._models: dict[str, tuple] = {}
        self._tok = None

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=f"encoder_ft:{self.model_id}:{self.train_generator}", deployment="local", supports_batching=True,
                                 tunables={"train_n": self.train_n, "train_seed": self.train_seed, "note": "trained on labelled examples; not zero-shot"})

    def _key(self, qkey: str, labels: list[str]) -> str:
        return hashlib.sha256(json.dumps([self.model_id, self.train_generator, self.train_n, self.train_seed, self.epochs, qkey, labels]).encode()).hexdigest()[:16]

    def _ensure(self, qkey: str, labels: list[str], qtype: str):
        if qkey in self._models:
            return self._models[qkey]
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        from ..generators import get_generator

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
        path = self.cache_dir / self._key(qkey, labels)
        if path.exists():
            model = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
        else:
            items = get_generator(self.train_generator, n=self.train_n, seed=self.train_seed).generate()
            texts, ys = [], []
            for it in items:
                q = it.questions.get(qkey)
                if q is None or q.ground_truth is None:
                    continue
                gt = str(q.ground_truth).lower() if q.type == "noul" else str(q.ground_truth)
                if gt in labels:
                    texts.append(it.state_text())
                    ys.append(labels.index(gt))
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=len(labels)).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
            model.train()
            idx = np.arange(len(texts))
            rng = np.random.default_rng(self.train_seed)
            for _ in range(self.epochs):
                rng.shuffle(idx)
                for b in range(0, len(idx), self.batch_size):
                    bi = idx[b:b + self.batch_size]
                    enc = self._tok([texts[i] for i in bi], truncation=True, max_length=self.max_length, padding=True, return_tensors="pt").to(dev)
                    out = model(**enc, labels=torch.tensor([ys[i] for i in bi], device=dev))
                    out.loss.backward()
                    opt.step()
                    opt.zero_grad()
            model.eval()
            path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(path)
        self._models[qkey] = (model, dev)
        return self._models[qkey]

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        import torch  # type: ignore

        t0 = self._now_ms()
        answers = {}
        hw = None
        for k, q in request.questions.items():
            labels = q.option_keys
            try:
                model, dev = self._ensure(k, labels, q.type)
            except Exception as e:
                answers[k] = Answer.failed(q, f"train_failed:{type(e).__name__}")
                continue
            enc = self._tok([request.state_text()], truncation=True, max_length=self.max_length, padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits[0].float(), -1).cpu().numpy()
            answers[k] = finalize_answer(q, [float(x) for x in p], confidence=float(p.max()))
            hw = hw or (torch.cuda.get_device_name(0) if "cuda" in str(dev) else "cpu")
        ms = self._now_ms() - t0
        return DecisionResponse(answers=answers, latency=LatencyRecord(client_ms=ms, compute_ms=ms, timestamp=time.time()),
                                provider=ProviderRecord(adapter_id=self.adapter_id, model_id_requested=self.model_id, model_id_returned=f"encoder_ft:{self.model_id}",
                                                        version_hash=f"{self.train_generator}@n{self.train_n}s{self.train_seed}e{self.epochs}", hardware=hw, cost_usd=0.0))
