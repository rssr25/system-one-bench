"""Log triage generator: JSON log records with injected distractor keys.

  severity   (score 0..3)   -> from the log level and error class, per a stated policy
  owning_team (choice)      -> from the service name (a mapping is given in the state as `team_directory`)
  page_oncall (noul)        -> policy: severity >= 2 AND the environment is production
The team directory lives in the state, so the model must read a lookup table rather than guess; the paging rule
combines two fields. Distractors are irrelevant keys with plausible values.
"""

from __future__ import annotations

import json

from ..schemas import Controls, LabelProvenance, Level, Option, Question, TaskItem
from .base import BaseGenerator, canary_for, estimate_tokens, pad_to_tokens, register_generator

SERVICES = {"checkout-api": "payments", "ledger": "payments", "search-indexer": "discovery", "ranker": "discovery", "auth-gateway": "identity",
            "session-store": "identity", "notify-worker": "messaging", "mailer": "messaging", "cdn-edge": "platform", "k8s-autoscaler": "platform"}
TEAMS = [("payments", "Checkout, ledger, refunds"), ("discovery", "Search and ranking"), ("identity", "Auth, sessions, SSO"),
         ("messaging", "Email, push, in-app notifications"), ("platform", "Infra, CDN, orchestration")]
# (level, error class, base severity, message)
EVENTS = [
    ("INFO", "deploy", 0, "deployment {ver} rolled out to {pct}% of pods"),
    ("INFO", "cache", 0, "cache warmup complete in {ms} ms"),
    ("WARN", "latency", 1, "p99 latency {ms} ms exceeds SLO of 800 ms for {n} minutes"),
    ("WARN", "retry", 1, "upstream {dep} returned 503; retry {n}/5 succeeded"),
    ("ERROR", "exception", 2, "NullPointerException in {fn} handling request {rid}; {n} requests failed"),
    ("ERROR", "db", 2, "connection pool exhausted; {n} queries queued > 5 s"),
    ("CRITICAL", "outage", 3, "health check failing on all replicas for {n} minutes; traffic returning 5xx"),
    ("CRITICAL", "data", 3, "write path returned inconsistent results; {n} records may be lost"),
    ("ERROR", "auth", 2, "token validation failing for {pct}% of requests since {ver}"),
    ("WARN", "disk", 1, "disk usage at {pct}% on {n} nodes"),
]
ENVS = ["production", "production", "staging", "canary"]
DISTRACTORS = {"trace_sampling": "0.1", "build_sha": "a3f9c21", "region": "eu-central-1", "feature_flags": ["new-ranker", "dark-mode"],
               "previous_incident": "INC-2041 (resolved)", "runbook_url": "https://runbooks.internal/generic", "request_headers": {"x-ab": "B"}}
SEV = [Level(level=0, description="Informational; no action."), Level(level=1, description="Degraded; investigate within the day."),
       Level(level=2, description="Errors affecting users; fix within hours."), Level(level=3, description="Outage or data loss; all hands now.")]
SEV_INSTR = "Severity 0-3 from the log record. Policy: INFO=0, WARN=1, ERROR=2, CRITICAL=3."


@register_generator("log_triage")
class LogTriageGenerator(BaseGenerator):
    version = "1.0.0"

    def generate(self) -> list[TaskItem]:
        kb = self.knobs
        rng, ctl = self.rng("items"), self.rng("controls")
        items = []
        for i in range(kb.n):
            distract = ctl.random() < kb.distractor_density
            noise_draw = ctl.random() < kb.label_noise
            service = rng.choice(list(SERVICES))
            level, cls, sev, msg = rng.choice(EVENTS)
            env = rng.choice(ENVS)
            msg = msg.format(ver=f"v{rng.randrange(2, 9)}.{rng.randrange(0, 30)}", pct=rng.randrange(5, 100), ms=rng.randrange(120, 4000),
                             n=rng.randrange(1, 60), dep=rng.choice(list(SERVICES)), fn=rng.choice(["chargeCard", "buildIndex", "verifyToken", "renderEmail"]),
                             rid=f"req-{rng.randrange(10**5, 10**6)}")
            record = {"ts": f"2026-09-{rng.randrange(1, 28):02d}T{rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}:00Z", "level": level, "service": service,
                      "env": env, "message": msg, "error_class": cls}
            if distract:
                for k in ctl.sample(list(DISTRACTORS), 3):
                    record[k] = DISTRACTORS[k]
            state = {"log": record, "team_directory": SERVICES,
                     "paging_policy": "Page the on-call engineer only when severity is 2 or higher AND env is production."}
            if kb.target_tokens:
                state["context"] = pad_to_tokens(ctl, "", kb.target_tokens)
            team = SERVICES[service]
            page = sev >= 2 and env == "production"
            noise = False
            if kb.label_noise and noise_draw:
                noise = True
                team = ctl.choice([t for t, _ in TEAMS if t != team])
                sev = ctl.choice([x for x in range(4) if x != sev])
                page = not page
            tid = f"logs_{kb.seed}_{i:05d}"
            items.append(TaskItem(
                task_id=tid, tier="G", domain="log_triage", language=kb.language, state=state,
                state_tokens=estimate_tokens(json.dumps(state)),
                questions={
                    "severity": Question(type="score", instructions=SEV_INSTR, criteria=SEV, ground_truth=sev, framing_group="logs.severity"),
                    "owning_team": Question(type="choice", instructions="Which team owns the service in `log.service`? Use `team_directory`.",
                                            criteria=[Option(key=k, description=d) for k, d in TEAMS], ground_truth=team, framing_group="logs.owning_team"),
                    "page_oncall": Question(type="noul", instructions="Per `paging_policy`, should the on-call engineer be paged for this record?",
                                            ground_truth=page, framing_group="logs.page_oncall"),
                },
                label_provenance=LabelProvenance(source="generator", generator=self.generator_id, seed=kb.seed, noise_injected=noise),
                controls=Controls(distractor_density=kb.distractor_density), cost_matrix=self.cost_matrices(), canary=canary_for(tid),
                metadata={"level": level, "env": env, "service": service},
            ))
        return items

    def regex_rules(self) -> dict:
        return {"logs.severity": {"0": [r'"level": "INFO"'], "1": [r'"level": "WARN"'], "2": [r'"level": "ERROR"'], "3": [r'"level": "CRITICAL"']},
                "logs.owning_team": {t: [rf'"service": "{s}"' for s, tt in SERVICES.items() if tt == t] for t, _ in TEAMS},
                "logs.page_oncall": [r'"level": "(ERROR|CRITICAL)"[^}]*"env": "production"|"env": "production"[^}]*"level": "(ERROR|CRITICAL)"']}

    def cost_matrices(self):
        return {"page_oncall": {"true": {"true": 0.0, "false": 50.0}, "false": {"true": 5.0, "false": 0.0}},
                "severity": {str(t): {str(p): float(abs(t - p) * (3 if p < t else 1)) for p in range(4)} for t in range(4)}}
