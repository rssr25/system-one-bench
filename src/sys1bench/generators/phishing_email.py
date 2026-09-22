"""Phishing email generator (is_phishing: noul, attack_class: choice, urgency: score) with a built-in
5-way decomposition for Suite B (sender mismatch, credential request, urgency language, link/domain
mismatch, unexpected attachment) combined by a fixed rule.
"""

from __future__ import annotations

from ..schemas import Controls, LabelProvenance, Level, Option, Question, TaskItem
from .base import BaseGenerator, canary_for, pad_to_tokens, register_generator

BRANDS = ["Northwind Bank", "Contoso Cloud", "Fabrikam Shipping", "Tailspin Payroll", "Woodgrove HR"]
LEGIT_DOMAINS = {"Northwind Bank": "northwindbank.com", "Contoso Cloud": "contoso.com", "Fabrikam Shipping": "fabrikam.com",
                 "Tailspin Payroll": "tailspin.com", "Woodgrove HR": "woodgrove.com"}
LOOKALIKE = {"northwindbank.com": "northwind-bank-secure.com", "contoso.com": "contoso-cloud-login.net",
             "fabrikam.com": "fabrikam-delivery.co", "tailspin.com": "tailspin-payroll.info", "woodgrove.com": "woodgrove-hr.org"}

CLASSES = [
    ("credential_harvest", "Asks the recipient to log in or verify credentials via a link"),
    ("invoice_fraud", "Fake invoice or payment redirection"),
    ("malware_attachment", "Unexpected attachment the recipient is urged to open"),
    ("ceo_fraud", "Impersonates an executive requesting an urgent transfer or gift cards"),
    ("benign", "Legitimate message with no attack"),
]
URGENCY = [Level(level=0, description="No time pressure"), Level(level=1, description="Mild: 'when you get a chance'"),
           Level(level=2, description="Moderate: action requested within days"), Level(level=3, description="High: within 24 hours or account consequences")]

LEGIT_TEMPLATES = [
    ("Your {brand} statement is ready", "Hi {name}, your monthly statement is available in the app under Documents. No action is required.", 0, False),
    ("Delivery update for order {order}", "Hi {name}, your parcel is out for delivery today. Track it in your account.", 0, False),
    ("Team lunch Thursday", "Hi {name}, we're doing lunch Thursday at noon. Reply if you have dietary needs.", 1, False),
    ("Q3 timesheet reminder", "Hi {name}, timesheets are due Friday. Submit through the usual portal when you get a chance.", 1, False),
    ("Invoice {order} attached", "Hi {name}, attached is the invoice we discussed on the call. Let me know if anything looks off.", 1, True),
]
PHISH_TEMPLATES = [
    ("credential_harvest", "Urgent: verify your {brand} account", "Dear customer, unusual sign-in detected. Verify your identity within 24 hours at https://{fake}/verify or your account will be suspended.", 3, False),
    ("credential_harvest", "Password expires today", "Your {brand} password expires today. Reset now: https://{fake}/reset", 3, False),
    ("invoice_fraud", "Updated banking details for invoice {order}", "Hi {name}, please note our bank details have changed. Remit invoice {order} to the new account in the attached PDF by end of week.", 2, True),
    ("malware_attachment", "Scanned document", "Please see the attached scan (Document_{order}.zip). Open to review.", 1, True),
    ("ceo_fraud", "Quick favour", "{name}, I'm in a meeting and need you to buy 5 gift cards for a client today. Send the codes here. Keep it confidential. Sent from my iPhone", 3, False),
]
NAMES = ["Alex", "Priya", "Jordan", "Mei", "Sam", "Fatima", "Luca", "Noor"]


@register_generator("phishing_email")
class PhishingEmailGenerator(BaseGenerator):
    version = "1.0.0"

    def generate(self) -> list[TaskItem]:
        kb = self.knobs
        rng = self.rng("items")
        ctl = self.rng("controls")
        items: list[TaskItem] = []
        for i in range(kb.n):
            distract = ctl.random() < kb.distractor_density
            noise_draw = ctl.random() < kb.label_noise
            brand = rng.choice(BRANDS)
            legit_dom = LEGIT_DOMAINS[brand]
            name = rng.choice(NAMES)
            order = rng.randrange(10000, 99999)
            is_phish = rng.random() < 0.5
            if is_phish:
                cls, subj, body, urg, attach = rng.choice(PHISH_TEMPLATES)
                sender_dom = LOOKALIKE[legit_dom] if rng.random() < 0.8 else legit_dom  # 20% spoof the real domain
                fake = LOOKALIKE[legit_dom]
            else:
                subj, body, urg, attach = rng.choice(LEGIT_TEMPLATES)
                cls, sender_dom, fake = "benign", legit_dom, legit_dom
            body = body.format(brand=brand, name=name, order=order, fake=fake)
            subj = subj.format(brand=brand, order=order)
            state = {"from": f"{brand} <no-reply@{sender_dom}>", "to": f"{name.lower()}@example-corp.com", "subject": subj,
                     "body": body, "attachments": ([f"file_{order}.{'zip' if cls == 'malware_attachment' else 'pdf'}"] if attach else []),
                     "known_brand_domain": legit_dom}
            if distract:
                state["thread_history"] = "Earlier: 'Thanks for the quick turnaround last month!'"
            if kb.target_tokens:
                state["footer"] = pad_to_tokens(ctl, "", kb.target_tokens - int(len(body.split()) * 1.3))
            noise = False
            truth_phish, truth_cls, truth_urg = is_phish, cls, urg
            if kb.label_noise and noise_draw:
                noise = True
                truth_phish = not truth_phish
                truth_cls = ctl.choice([c[0] for c in CLASSES if c[0] != cls])
            tid = f"phish_{kb.seed}_{i:05d}"
            # decomposition sub-questions (Suite B): each is a noul with rule-derived truth
            sub = {
                "sender_mismatch": ("Does the sender domain differ from the known brand domain?", sender_dom != legit_dom),
                "asks_credentials": ("Does the message ask the recipient to log in, verify, or reset credentials via a link?", cls == "credential_harvest"),
                "urgent_language": ("Does the message pressure the recipient to act within 24 hours or threaten consequences?", urg >= 3),
                "payment_redirect": ("Does the message ask to change payment details or send money or gift cards?", cls in ("invoice_fraud", "ceo_fraud")),
                "unexpected_attachment": ("Does the message urge opening an attachment the recipient did not request?", cls == "malware_attachment"),
            }
            questions = {
                "is_phishing": Question(type="noul", instructions="Is this email a phishing or social-engineering attack?",
                                        ground_truth=truth_phish, framing_group="phish.is_phishing"),
                "attack_class": Question(type="choice", instructions="Classify the email.", criteria=[Option(key=k, description=d) for k, d in CLASSES],
                                         ground_truth=truth_cls, framing_group="phish.attack_class"),
                "urgency": Question(type="score", instructions="Rate the time pressure the message applies, 0 (none) to 3 (within 24 hours or threats).",
                                    criteria=URGENCY, ground_truth=truth_urg, framing_group="phish.urgency"),
            }
            for k, (instr, truth) in sub.items():
                questions[f"sub_{k}"] = Question(type="noul", instructions=instr, ground_truth=truth,
                                                 framing_group=f"phish.sub.{k}", decomposition_of="is_phishing")
            items.append(TaskItem(
                task_id=tid, tier="G", domain="email_security", language=kb.language, state=state,
                state_tokens=int(len(body.split()) * 1.3) + 40, questions=questions,
                label_provenance=LabelProvenance(source="generator", generator=self.generator_id, seed=kb.seed, noise_injected=noise),
                controls=Controls(distractor_density=kb.distractor_density), cost_matrix=self.cost_matrices(), canary=canary_for(tid),
                metadata={"class": cls, "sender_domain": sender_dom, "decomposition_rule": "phishing if sender_mismatch or any of the other four"},
            ))
        return items

    def regex_rules(self) -> dict:
        return {
            "phish.is_phishing": [r"verify your", r"suspended", r"expires today", r"gift cards", r"bank details have changed", r"\.zip", r"-secure\.com|-login\.net|-delivery\.co|-payroll\.info|-hr\.org"],
            "phish.attack_class": {"credential_harvest": [r"verify", r"reset now", r"password"], "invoice_fraud": [r"bank details", r"remit"],
                                   "malware_attachment": [r"\.zip", r"scanned document"], "ceo_fraud": [r"gift cards", r"confidential"],
                                   "benign": [r"no action is required", r"team lunch", r"timesheet", r"out for delivery"]},
            "phish.urgency": {"3": [r"24 hours|today|suspended"], "2": [r"end of week"], "1": [r"when you get a chance|let me know"], "0": [r"no action"]},
        }

    def cost_matrices(self) -> dict[str, dict[str, dict[str, float]]]:
        # missing a phish costs 20; false alarm costs 1
        return {"is_phishing": {"true": {"true": 0.0, "false": 20.0}, "false": {"true": 1.0, "false": 0.0}}}
