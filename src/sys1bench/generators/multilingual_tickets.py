"""Multilingual support tickets: the support_tickets task templated in German, Spanish, French and Hindi (Devanagari),
with English questions and criteria. Labels follow the same policy as the English generator. Used for Laya's language
router and for Jev's language behaviour; scripts other than Latin are the collapse case reported by Convai.
"""

from __future__ import annotations

from ..schemas import Controls, LabelProvenance, Option, Question, TaskItem
from .base import BaseGenerator, canary_for, register_generator
from .support_tickets import PRIORITY_INSTRUCTIONS, PRIORITY_LEVELS, QUEUES

# queue, base urgency, {lang: template}
T = [
    ("billing", 1, {"de": "Mir wurde am {day} zweimal {amt} € für Bestellung #{order} abgebucht. Es darf nur eine Abbuchung geben.",
                    "es": "Me cobraron dos veces {amt} € el {day} por el pedido #{order}. Solo debería haber un cargo.",
                    "fr": "J'ai été débité deux fois de {amt} € le {day} pour la commande #{order}. Il ne devrait y avoir qu'un seul débit.",
                    "hi": "ऑर्डर #{order} के लिए {day} को मुझसे दो बार ₹{amt} लिए गए। केवल एक ही शुल्क होना चाहिए।"}),
    ("shipping", 1, {"de": "Bestellung #{order} gilt seit {day} als zugestellt, aber nichts ist angekommen.",
                     "es": "El pedido #{order} figura como entregado el {day}, pero no ha llegado nada.",
                     "fr": "La commande #{order} est indiquée livrée le {day}, mais rien n'est arrivé.",
                     "hi": "ऑर्डर #{order} {day} को डिलीवर दिखाया गया है, लेकिन कुछ भी नहीं आया।"}),
    ("account", 2, {"de": "Ich bin ausgesperrt. Die Zwei-Faktor-Codes gehen an eine alte Nummer.",
                    "es": "No puedo entrar. Los códigos de dos factores van a un número antiguo.",
                    "fr": "Je suis bloqué. Les codes à deux facteurs partent vers un ancien numéro.",
                    "hi": "मैं लॉक आउट हो गया हूँ। टू-फैक्टर कोड पुराने नंबर पर जा रहे हैं।"}),
    ("technical", 2, {"de": "Die App stürzt seit dem Update am {day} beim Start ab. Fehlercode {code}.",
                      "es": "La aplicación se cierra al abrirla desde la actualización del {day}. Código de error {code}.",
                      "fr": "L'application plante au démarrage depuis la mise à jour du {day}. Code d'erreur {code}.",
                      "hi": "{day} के अपडेट के बाद से ऐप खोलते ही क्रैश हो जाता है। एरर कोड {code}।"}),
    ("cancellation", 1, {"de": "Bitte kündigen Sie mein Abo vor der Verlängerung am {day}.",
                         "es": "Por favor cancelen mi suscripción antes de la renovación del {day}.",
                         "fr": "Veuillez annuler mon abonnement avant le renouvellement du {day}.",
                         "hi": "कृपया {day} के नवीनीकरण से पहले मेरी सदस्यता रद्द करें।"}),
    ("sales", 0, {"de": "Gibt es einen Rabatt bei jährlicher Zahlung für das Team-Paket?",
                  "es": "¿Hay descuento por pago anual en el plan de equipo?",
                  "fr": "Y a-t-il une remise pour un paiement annuel sur l'offre équipe ?",
                  "hi": "टीम प्लान पर वार्षिक भुगतान के लिए कोई छूट है?"}),
]
ANGRY = {"de": "Das ist völlig inakzeptabel, ich bin wütend.", "es": "Esto es completamente inaceptable, estoy furioso.",
         "fr": "C'est totalement inacceptable, je suis furieux.", "hi": "यह पूरी तरह अस्वीकार्य है, मैं बहुत गुस्से में हूँ।"}
CALM = {"de": "Vielen Dank im Voraus.", "es": "Gracias de antemano.", "fr": "Merci d'avance.", "hi": "अग्रिम धन्यवाद।"}
CTX = {"de": "Kontext: Kundenstufe {tier}.", "es": "Contexto: nivel de cliente {tier}.", "fr": "Contexte : niveau client {tier}.", "hi": "संदर्भ: ग्राहक स्तर {tier}।"}
TIERS = ["free", "standard", "gold", "enterprise"]


@register_generator("multilingual_tickets")
class MultilingualTicketsGenerator(BaseGenerator):
    version = "1.0.0"

    def generate(self) -> list[TaskItem]:
        kb = self.knobs
        rng = self.rng("items")
        langs = [kb.language] if kb.language in ("de", "es", "fr", "hi") else ["de", "es", "fr", "hi"]
        queues = [q for q in QUEUES if q[0] in {t[0] for t in T}]
        items = []
        for i in range(kb.n):
            lang = langs[i % len(langs)]
            queue, base, tmpl = rng.choice(T)
            angry = rng.random() < 0.3
            tier = rng.choice(TIERS)
            body = tmpl[lang].format(day=rng.choice(["Montag", "3.", "gestern"]) if lang == "de" else rng.choice(["lunes", "3", "ayer"]) if lang == "es"
                                     else rng.choice(["lundi", "3", "hier"]) if lang == "fr" else rng.choice(["सोमवार", "3 तारीख", "कल"]),
                                     amt=rng.choice([19, 45, 89, 249]), order=rng.randrange(10000, 99999), code=f"E{rng.randrange(100, 999)}")
            state = f"{body} {(ANGRY if angry else CALM)[lang]}\n{CTX[lang].format(tier=tier)}"
            priority = min(4, base + (1 if angry else 0) + (1 if tier in ("gold", "enterprise") else 0))
            tid = f"mltickets_{kb.seed}_{lang}_{i:05d}"
            items.append(TaskItem(
                task_id=tid, tier="G", domain="support_triage", language=lang, state=state, state_tokens=int(len(state.split()) * 2.0),
                questions={
                    "queue": Question(type="choice", instructions="Which queue should handle this ticket?", criteria=[Option(key=k, description=d) for k, d in queues],
                                      ground_truth=queue, framing_group="tickets.queue"),
                    "is_angry": Question(type="noul", instructions="Is the customer expressing anger or hostility?", ground_truth=angry, framing_group="tickets.is_angry"),
                    "priority": Question(type="score", instructions=PRIORITY_INSTRUCTIONS, criteria=PRIORITY_LEVELS, ground_truth=priority, framing_group="tickets.priority"),
                },
                label_provenance=LabelProvenance(source="generator", generator=self.generator_id, seed=kb.seed), controls=Controls(),
                canary=canary_for(tid), metadata={"lang": lang, "tier": tier, "angry": angry, "base_urgency": base},
            ))
        return items

    def regex_rules(self) -> dict:
        return {"tickets.is_angry": [r"inakzeptabel|inaceptable|furieux|अस्वीकार्य"]}

    def cost_matrices(self):
        return {}
