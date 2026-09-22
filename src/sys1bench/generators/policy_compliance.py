"""Policy compliance generator: a short policy (3 to 5 numbered clauses) plus a request.

  compliant       (noul)  -> no clause is violated
  violated_clause (choice over clause ids plus a required "none")  -> the single violated clause, or none
  severity        (score 0..2) -> 0 compliant, 1 minor clause, 2 major clause (clause severities stated in the policy)
Requests are built from the same clause set, so the label is exact and the "none" option is correct for a stated
fraction of items. Nothing in the request text names the clause.
"""

from __future__ import annotations

import json

from ..schemas import Controls, LabelProvenance, Level, Option, Question, TaskItem
from .base import BaseGenerator, canary_for, estimate_tokens, pad_to_tokens, register_generator

# clause: (id, text, severity 1|2, violating request templates, compliant request templates)
CLAUSES = [
    ("C1", "Refunds are only issued within 30 days of purchase.", 2,
     ["I bought this {days} days ago and want a full refund.", "Please refund order #{order}; it was placed {days} days back."],
     ["I bought this {sdays} days ago and want a full refund.", "Order #{order} from {sdays} days ago, please refund."]),
    ("C2", "Account details may only be shared with the verified account holder.", 2,
     ["I'm calling for my husband; can you read me his last four transactions?", "My roommate's account: what email is on file? He asked me to check."],
     ["Can you read me my own last four transactions? I've verified via the security question.", "What email is on my account? I'm the holder and verified."]),
    ("C3", "Discount codes cannot be combined with other promotions.", 1,
     ["Apply code SAVE20 on top of the current summer sale, please.", "Stack my loyalty coupon with the Black Friday price."],
     ["Apply code SAVE20 to my order; no other promotions are on it.", "Use my loyalty coupon on this full-price item."]),
    ("C4", "Shipping address changes are not possible after dispatch.", 1,
     ["My order shipped this morning; change the address to my office.", "Tracking says dispatched; please redirect it to my parents' house."],
     ["My order hasn't shipped yet; change the address to my office.", "Order still processing, please update the delivery address."]),
    ("C5", "Support agents must not provide legal or tax advice.", 2,
     ["Should I declare this refund as income on my tax return?", "Is it legal for my employer to deduct this from my pay?"],
     ["Can you send me the invoice so my accountant can handle the taxes?", "Where do I download my annual statement?"]),
]
SEV = [Level(level=0, description="Compliant: no clause is violated."), Level(level=1, description="Minor violation (a clause marked minor)."),
       Level(level=2, description="Major violation (a clause marked major).")]


@register_generator("policy_compliance")
class PolicyComplianceGenerator(BaseGenerator):
    version = "1.0.0"

    def generate(self) -> list[TaskItem]:
        kb = self.knobs
        rng, ctl = self.rng("items"), self.rng("controls")
        none_frac = kb.none_correct_frac if kb.none_correct_frac else 0.3
        items = []
        for i in range(kb.n):
            noise_draw = ctl.random() < kb.label_noise
            k = rng.randrange(3, 6)
            clauses = rng.sample(CLAUSES, k)
            policy = "\n".join(f"{cid}. {txt} [{'major' if sev == 2 else 'minor'}]" for cid, txt, sev, _, _ in clauses)
            compliant = rng.random() < none_frac
            target = rng.choice(clauses)
            tmpl = rng.choice(target[4] if compliant else target[3])
            request = tmpl.format(days=rng.randrange(31, 120), sdays=rng.randrange(1, 29), order=rng.randrange(10000, 99999))
            state = {"policy": policy, "customer_request": request}
            if kb.target_tokens:
                state["history"] = pad_to_tokens(ctl, "", kb.target_tokens)
            violated = "none" if compliant else target[0]
            sev = 0 if compliant else target[2]
            noise = False
            if kb.label_noise and noise_draw:
                noise = True
                compliant = not compliant
                violated = "none" if compliant else target[0]
                sev = 0 if compliant else target[2]
            options = [Option(key=cid, description=txt) for cid, txt, _, _, _ in clauses] + [Option(key="none", description="No clause is violated.")]
            tid = f"policy_{kb.seed}_{i:05d}"
            items.append(TaskItem(
                task_id=tid, tier="G", domain="policy_compliance", language=kb.language, state=state, state_tokens=estimate_tokens(json.dumps(state)),
                questions={
                    "compliant": Question(type="noul", instructions="Can the request be fulfilled without violating any clause of `policy`?",
                                          ground_truth=compliant, framing_group="policy.compliant"),
                    "violated_clause": Question(type="choice", instructions="Which clause of `policy` would fulfilling `customer_request` violate? Choose none if it complies.",
                                                criteria=options, ground_truth=violated, framing_group="policy.violated_clause"),
                    "severity": Question(type="score", instructions="Violation severity per the [minor]/[major] tags in `policy`; 0 if compliant.",
                                         criteria=SEV, ground_truth=sev, framing_group="policy.severity"),
                },
                label_provenance=LabelProvenance(source="generator", generator=self.generator_id, seed=kb.seed, noise_injected=noise),
                controls=Controls(none_correct=compliant), cost_matrix=self.cost_matrices(), canary=canary_for(tid),
                metadata={"target_clause": target[0], "k": k},
            ))
        return items

    def regex_rules(self) -> dict:
        return {"policy.compliant": [r"(?!)"]}

    def cost_matrices(self):
        return {"compliant": {"true": {"true": 0.0, "false": 2.0}, "false": {"true": 10.0, "false": 0.0}}}
