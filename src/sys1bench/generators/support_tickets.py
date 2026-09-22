"""Support ticket generator (queue: choice, is_angry: noul, priority: score).

Priority policy (stated in the question, not derivable from the ticket alone):
    base urgency by template (0..2) + 1 if angry + 1 if customer tier is gold/enterprise, capped at 4.
The customer tier appears in a `Context:` line. In the *unknowable* arm the tier line is removed, so a
calibrated model must spread mass over two adjacent levels.
"""

from __future__ import annotations

from ..schemas import Controls, LabelProvenance, Level, Option, Question, TaskItem
from .base import BaseGenerator, canary_for, estimate_tokens, pad_to_tokens, register_generator

QUEUES = [
    ("billing", "Charges, refunds, invoices, duplicate payments, autopay problems"),
    ("shipping", "Delivery delays, tracking, lost or damaged packages, wrong address"),
    ("account", "Login, password, two-factor, email change, account lockout"),
    ("technical", "App crashes, errors, features not working, connectivity"),
    ("cancellation", "Ending a subscription or closing an account"),
    ("sales", "Pre-purchase questions, upgrades, quotes, discounts"),
    ("privacy", "Data export or deletion requests, consent, GDPR/CCPA"),
    ("returns", "Returning a product, exchanges, return labels, refund status after return"),
    ("fraud", "Unrecognised transactions, stolen card, suspicious login"),
    ("feedback", "Compliments or suggestions with no action requested"),
    ("accessibility", "Screen reader, captions, contrast or keyboard navigation problems"),
    ("partnership", "Business development, reseller or affiliate enquiries"),
]

# (queue, base_urgency, template)
TEMPLATES = [
    ("billing", 1, "I was charged ${amt} twice on {day} for order #{order}. Only one charge should exist."),
    ("billing", 0, "Can you send me the invoice for order #{order}? I need it for expenses."),
    ("billing", 1, "My autopay failed on {day} and now there's a late fee of ${amt}. I had funds available."),
    ("shipping", 1, "Order #{order} says delivered on {day} but nothing arrived. Tracking shows a photo of a door that isn't mine."),
    ("shipping", 0, "What's the estimated delivery for order #{order}? It's been {days} days."),
    ("shipping", 2, "Package #{order} arrived crushed and the contents are broken. I need this for an event on {day}."),
    ("account", 2, "I'm locked out. Two-factor codes go to a phone number I no longer have. Order #{order} is pending."),
    ("account", 0, "How do I change the email on my account? I want to switch to my work address."),
    ("technical", 2, "The app crashes on launch since the update on {day}. I can't access anything. Error code {code}."),
    ("technical", 1, "Export to CSV produces an empty file for reports over {days} days."),
    ("cancellation", 1, "Please cancel my subscription before the renewal on {day}. Reference #{order}."),
    ("cancellation", 0, "If I cancel, do I keep access until the end of the billing period?"),
    ("sales", 0, "Do you offer a discount for annual billing on the team plan? We'd be about {days} seats."),
    ("privacy", 1, "Please delete all my personal data under GDPR. Account reference #{order}."),
    ("returns", 1, "I returned order #{order} on {day} and tracking shows it was received. When is the refund?"),
    ("fraud", 2, "There's a ${amt} charge on {day} I did not make. My card ending {last4} may be compromised."),
    ("feedback", 0, "Just wanted to say the new dashboard is much better. No action needed."),
    ("accessibility", 1, "The checkout button has no label so my screen reader announces 'button'. Order #{order} stuck."),
    ("partnership", 0, "We're a reseller in {region} and would like to discuss a partnership."),
    ("technical", 1, "Notifications stopped arriving on {day}. Settings look correct. Device: {device}."),
]

# Surface variants applied to the customer text (labels unaffected): openers, register, sign-offs, light typos.
OPENERS = ["", "", "Hi,", "Hello team,", "Hey,", "Good morning.", "To whom it may concern,", "Quick one:", "ok so", "Hi there —"]
REGISTER = {
    "formal": lambda t: t, "casual": lambda t: t.replace("I was", "i was").replace("Please", "pls").replace("please", "pls").replace("Thanks", "thx"),
    "terse": lambda t: t.rstrip(".") + ".", "verbose": lambda t: t + " I have been a customer for a while and this has never happened before.",
}
SIGNOFFS = ["", "", "Regards, {name}", "— {name}", "Thanks, {name}", "Sent from my phone", "Best, {name}"]
NAMES = ["Priya", "Jonas", "Amara", "Luis", "Mei", "Tomasz", "Fatima", "Noah"]
ANGRY = [
    "This is completely unacceptable and I am furious.",
    "Honestly, this is ridiculous. Fix it today or I'm done.",
    "I've had it with this company. Absolutely appalling service.",
    "Unbelievable. Every single time. I want this escalated NOW.",
]
CALM = [
    "Thanks in advance for your help.",
    "Let me know if you need anything else from me.",
    "Appreciate you looking into it.",
    "No rush, but I'd like to get this sorted.",
]
TIERS = ["free", "standard", "gold", "enterprise"]
DISTRACTORS = [
    "P.S. I once had a delivery problem years ago with a different company; that got resolved.",
    "Also I noticed the password reset page has a typo, not important.",
    "By the way, a friend asked me whether you offer refunds, I told her I'd ask.",
    'Log excerpt (from another tab): {"event":"cancel_click","source":"pricing_page","ts":"2026-09-01T10:00:00Z"}',
]

PRIORITY_LEVELS = [
    Level(level=0, description="No action needed or informational."),
    Level(level=1, description="Routine request; respond within 2 business days."),
    Level(level=2, description="Customer blocked or money at stake; respond within 1 business day."),
    Level(level=3, description="Urgent: blocked plus angry or high-value customer; respond within 4 hours."),
    Level(level=4, description="Critical: blocked, high-value and angry, or active fraud; respond within 1 hour."),
]
PRIORITY_INSTRUCTIONS = (
    "Assign a priority from 0 (lowest) to 4 (highest). Policy: start from the urgency of the issue "
    "(informational 0, routine 1, blocked or money at stake 2). Add 1 if the customer is angry. "
    "Add 1 if the customer tier is gold or enterprise. Cap at 4."
)


@register_generator("support_tickets")
class SupportTicketGenerator(BaseGenerator):
    version = "1.1.0"

    def _queues(self) -> list[tuple[str, str]]:
        k = self.knobs.cardinality or 6
        k = max(2, min(k, len(QUEUES)))
        used = {t[0] for t in TEMPLATES}
        ordered = [q for q in QUEUES if q[0] in used] + [q for q in QUEUES if q[0] not in used]
        return ordered[:k]

    def generate(self) -> list[TaskItem]:
        kb = self.knobs
        rng = self.rng("items")          # content stream: identical across control arms
        ctl = self.rng("controls")       # control stream: distractors, padding, label noise, unknowable, none-correct
        queues = self._queues()
        qkeys = [q[0] for q in queues]
        templates = [t for t in TEMPLATES if t[0] in qkeys] or TEMPLATES
        items: list[TaskItem] = []
        for i in range(kb.n):
            queue, base, tmpl = rng.choice(templates)
            angry = rng.random() < 0.3
            tier = rng.choice(TIERS)
            unknowable = ctl.random() < kb.unknowable_frac
            none_correct = ctl.random() < kb.none_correct_frac
            distract = ctl.random() < kb.distractor_density
            noise_draw = ctl.random() < kb.label_noise
            body = tmpl.format(amt=rng.choice([19, 45, 89, 120, 249, 1200]), day=rng.choice(["Monday", "the 3rd", "yesterday", "Sept 14"]),
                               order=rng.randrange(10000, 99999), days=rng.randrange(2, 30), code=f"E{rng.randrange(100,999)}",
                               last4=rng.randrange(1000, 9999), region=rng.choice(["Germany", "Brazil", "Japan", "Canada"]),
                               device=rng.choice(["Pixel 8", "iPhone 15", "Galaxy S24"]))
            closing = rng.choice(ANGRY if angry else CALM)
            # 1.1.0 surface diversity (content stream so arms stay paired)
            opener = rng.choice(OPENERS)
            reg = rng.choice(list(REGISTER))
            body = REGISTER[reg](body)
            signoff = rng.choice(SIGNOFFS).format(name=rng.choice(NAMES))
            if rng.random() < 0.15:  # light typos in the body, never inside the order number
                chars = list(body)
                for _ in range(max(1, len(chars) // 60)):
                    j = rng.randrange(len(chars))
                    if chars[j].isalpha():
                        chars[j] = rng.choice("abcdefghijklmnopqrstuvwxyz")
                body = "".join(chars)
            age, open_t = rng.randrange(1, 60), rng.randrange(0, 4)
            ctx = f"Context: customer tier {tier}; account age {age} months; open tickets {open_t}."
            if unknowable:
                ctx = f"Context: account age {age} months; open tickets {open_t}."
            text = " ".join(x for x in (opener, body, closing, signoff) if x)
            state = f"Customer: \"{text}\"\n{ctx}"
            if distract:
                state += "\n" + ctl.choice(DISTRACTORS)
            state = pad_to_tokens(ctl, state, kb.target_tokens)

            priority = min(4, base + (1 if angry else 0) + (1 if tier in ("gold", "enterprise") else 0))
            queue_truth = queue
            options = [Option(key=k, description=d) for k, d in queues]
            if none_correct:
                # drop the true queue from the option list; truth becomes the abstain key
                options = [o for o in options if o.key != queue] or options
                queue_truth = "none_of_the_above"
            noise = False
            if kb.label_noise and noise_draw:
                noise = True
                queue_truth = ctl.choice([o.key for o in options if o.key != queue_truth] or [queue_truth])
                priority = ctl.choice([p for p in range(5) if p != priority])
                angry = not angry

            tid = f"tickets_{kb.seed}_{i:05d}"
            items.append(TaskItem(
                task_id=tid, tier="G", domain="support_triage", language=kb.language, state=state,
                state_tokens=estimate_tokens(state),
                questions={
                    "queue": Question(type="choice", instructions="Which queue should handle this ticket?", criteria=options,
                                      ground_truth=queue_truth, framing_group="tickets.queue", allow_abstain=none_correct),
                    "is_angry": Question(type="noul", instructions="Is the customer expressing anger or hostility?",
                                         ground_truth=angry, framing_group="tickets.is_angry"),
                    "priority": Question(type="score", instructions=PRIORITY_INSTRUCTIONS, criteria=PRIORITY_LEVELS,
                                         ground_truth=priority, framing_group="tickets.priority"),
                },
                label_provenance=LabelProvenance(source="generator", generator=self.generator_id, seed=kb.seed, noise_injected=noise),
                controls=Controls(unknowable=unknowable, none_correct=none_correct, distractor_density=kb.distractor_density),
                cost_matrix=self.cost_matrices(),
                canary=canary_for(tid),
                metadata={"template_queue": queue, "base_urgency": base, "tier": tier, "angry": angry, "register": reg},
            ))
        return items

    def regex_rules(self) -> dict:
        return {
            "tickets.queue": {
                "billing": [r"charged", r"invoice", r"autopay", r"late fee"],
                "shipping": [r"deliver", r"tracking", r"package", r"arrived"],
                "account": [r"locked out", r"two-factor", r"change the email", r"password"],
                "technical": [r"crash", r"error code", r"empty file", r"notifications stopped"],
                "cancellation": [r"cancel"],
                "sales": [r"discount", r"annual billing", r"seats"],
                "privacy": [r"GDPR", r"delete all my personal data"],
                "returns": [r"returned order", r"refund\?"],
                "fraud": [r"did not make", r"compromised"],
                "feedback": [r"no action needed", r"wanted to say"],
                "accessibility": [r"screen reader"],
                "partnership": [r"reseller", r"partnership"],
            },
            "tickets.is_angry": [r"unacceptable", r"furious", r"ridiculous", r"appalling", r"had it with", r"escalated"],
            "tickets.priority": {"4": [r"fraud|did not make|compromised"], "3": [r"crash|locked out|crushed"],
                                 "2": [r"charged .* twice|late fee|delivered .* nothing arrived|refund"],
                                 "1": [r"cancel|GDPR|export|screen reader"], "0": [r"no action needed|discount|partnership|invoice"]},
        }

    def cost_matrices(self) -> dict[str, dict[str, dict[str, float]]]:
        # asymmetric: routing fraud/billing wrongly is expensive; misrouting feedback is cheap
        base = {q[0]: 1.0 for q in QUEUES}
        expensive = {"fraud": 8.0, "billing": 3.0, "privacy": 4.0, "account": 3.0}
        queue = {t: {p: (0.0 if p == t else expensive.get(t, base[t])) for p in base} for t in base}
        priority = {str(t): {str(p): float(abs(t - p) * (2 if p < t else 1)) for p in range(5)} for t in range(5)}
        angry = {"true": {"true": 0.0, "false": 3.0}, "false": {"true": 0.5, "false": 0.0}}
        return {"queue": queue, "priority": priority, "is_angry": angry}
