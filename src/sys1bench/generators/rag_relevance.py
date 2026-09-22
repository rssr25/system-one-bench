"""RAG relevance generator: query plus K candidate passages.

Questions:
  best_passage  (choice over K passage ids, K up to 255)  -> the one passage that answers the query
  is_relevant   (noul)  -> does passage `focus` answer the query (focus is a random candidate; 50% positives)
  relevance     (score 0..3) -> grade of `focus`: 0 unrelated, 1 same topic, 2 partial, 3 answers it

Passages are built from a fact table: each fact has a topic, an entity, an attribute and a value. Distractors are
(a) same topic different entity (hard), (b) same entity different attribute (hard), (c) unrelated (easy). Labels are
exact by construction, so this is the cardinality sweep generator for Suite C (K in 2..255).
"""

from __future__ import annotations

import random

from ..schemas import Controls, LabelProvenance, Level, Option, Question, TaskItem
from .base import BaseGenerator, canary_for, pad_to_tokens, register_generator

TOPICS = {
    "planets": (["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
                {"moons": lambda r: str(r.randrange(0, 90)), "day_length_hours": lambda r: str(r.randrange(10, 5900)),
                 "discovery_year": lambda r: str(r.randrange(1600, 1950)), "mean_temp_c": lambda r: str(r.randrange(-220, 470))}),
    "companies": (["Northwind", "Contoso", "Fabrikam", "Tailspin", "Woodgrove", "Litware", "Adatum", "Proseware"],
                  {"headquarters": lambda r: r.choice(["Oslo", "Lisbon", "Austin", "Kyoto", "Nairobi", "Toronto"]),
                   "founded": lambda r: str(r.randrange(1950, 2022)), "employees": lambda r: str(r.randrange(20, 90000)),
                   "ceo": lambda r: r.choice(["A. Okafor", "M. Lindqvist", "R. Tanaka", "S. Mehta", "J. Alvarez"])}),
    "recipes": (["ratatouille", "bibimbap", "moussaka", "feijoada", "pho", "tagine", "paella", "goulash"],
                {"origin_country": lambda r: r.choice(["France", "Korea", "Greece", "Brazil", "Vietnam", "Morocco", "Spain", "Hungary"]),
                 "cook_time_minutes": lambda r: str(r.randrange(15, 240)), "main_protein": lambda r: r.choice(["beef", "pork", "chicken", "lamb", "none", "fish"]),
                 "servings": lambda r: str(r.randrange(2, 12))}),
    "software": (["Kestrel", "Nimbus", "Quartz", "Vellum", "Marlin", "Orchid", "Tundra", "Zephyr"],
                 {"latest_version": lambda r: f"{r.randrange(1, 9)}.{r.randrange(0, 20)}", "license": lambda r: r.choice(["MIT", "Apache-2.0", "GPL-3.0", "BSD-3", "proprietary"]),
                  "language": lambda r: r.choice(["Rust", "Go", "Python", "TypeScript", "C++"]), "first_release": lambda r: str(r.randrange(2005, 2026))}),
}
TEMPLATES = ["{entity}'s {attr_h} is {value}.", "According to the {topic} handbook, the {attr_h} of {entity} is {value}.",
             "Note: {entity} — {attr_h}: {value}.", "For {entity}, records list the {attr_h} as {value}."]
QUERY_TEMPLATES = ["What is the {attr_h} of {entity}?", "{entity}: {attr_h}?", "Tell me {entity}'s {attr_h}.", "Look up the {attr_h} for {entity}."]
REL_LEVELS = [Level(level=0, description="Unrelated to the query's topic or entity."),
              Level(level=1, description="Same topic but about a different entity or attribute; does not answer the query."),
              Level(level=2, description="About the right entity but a different attribute, or the right attribute for a different entity."),
              Level(level=3, description="Directly states the value the query asks for.")]


def _fact(rng: random.Random, topic: str, entity: str, attr: str) -> tuple[str, str]:
    ents, attrs = TOPICS[topic]
    value = attrs[attr](rng)
    text = rng.choice(TEMPLATES).format(entity=entity, attr_h=attr.replace("_", " "), value=value, topic=topic)
    return text, value


@register_generator("rag_relevance")
class RagRelevanceGenerator(BaseGenerator):
    version = "1.0.0"

    def generate(self) -> list[TaskItem]:
        kb = self.knobs
        rng = self.rng("items")
        K = max(2, min(kb.cardinality or 5, 255))
        items: list[TaskItem] = []
        for i in range(kb.n):
            topic = rng.choice(list(TOPICS))
            ents, attrs = TOPICS[topic]
            entity, attr = rng.choice(ents), rng.choice(list(attrs))
            query = rng.choice(QUERY_TEMPLATES).format(entity=entity, attr_h=attr.replace("_", " "))
            gold_text, _ = _fact(rng, topic, entity, attr)
            passages: list[tuple[str, int]] = [(gold_text, 3)]  # (text, relevance grade)
            # hard distractors
            while len(passages) < K:
                kind = rng.random()
                if kind < 0.35:  # same entity, other attribute
                    other_attr = rng.choice([a for a in attrs if a != attr])
                    passages.append((_fact(rng, topic, entity, other_attr)[0], 2))
                elif kind < 0.7:  # same attribute, other entity
                    other_ent = rng.choice([e for e in ents if e != entity])
                    passages.append((_fact(rng, topic, other_ent, attr)[0], 2))
                elif kind < 0.85:  # same topic, other entity and attribute
                    other_ent = rng.choice([e for e in ents if e != entity])
                    other_attr = rng.choice([a for a in attrs if a != attr])
                    passages.append((_fact(rng, topic, other_ent, other_attr)[0], 1))
                else:  # unrelated topic
                    ot = rng.choice([t for t in TOPICS if t != topic])
                    oe, oa = rng.choice(TOPICS[ot][0]), rng.choice(list(TOPICS[ot][1]))
                    passages.append((_fact(rng, ot, oe, oa)[0], 0))
            rng.shuffle(passages)
            ids = [f"p{j:03d}" for j in range(K)]
            gold_id = ids[[g for _, g in passages].index(3)]
            focus_idx = rng.randrange(K) if rng.random() < 0.5 else ids.index(gold_id)
            focus_id, focus_grade = ids[focus_idx], passages[focus_idx][1]
            state = {"query": query, "passages": {pid: txt for pid, (txt, _) in zip(ids, passages)}, "focus": focus_id}
            if kb.target_tokens:
                state["notes"] = pad_to_tokens(rng, "", kb.target_tokens)
            noise = False
            if kb.label_noise and rng.random() < kb.label_noise:
                noise = True
                gold_id = rng.choice([p for p in ids if p != gold_id])
            tid = f"rag_{kb.seed}_K{K}_{i:05d}"
            items.append(TaskItem(
                task_id=tid, tier="G", domain="rag_relevance", language=kb.language, state=state,
                state_tokens=int(sum(len(t.split()) for t, _ in passages) * 1.3) + 30,
                questions={
                    "best_passage": Question(type="choice", instructions="Which passage in `passages` directly answers `query`?",
                                             criteria=[Option(key=pid, description="") for pid in ids], ground_truth=gold_id, framing_group="rag.best_passage"),
                    "is_relevant": Question(type="noul", instructions="Does the passage with id `focus` directly answer `query`?",
                                            ground_truth=(focus_grade == 3), framing_group="rag.is_relevant"),
                    "relevance": Question(type="score", instructions="Grade how well the passage with id `focus` answers `query`.",
                                          criteria=REL_LEVELS, ground_truth=focus_grade, framing_group="rag.relevance"),
                },
                label_provenance=LabelProvenance(source="generator", generator=self.generator_id, seed=kb.seed, noise_injected=noise),
                controls=Controls(distractor_density=kb.distractor_density), cost_matrix=self.cost_matrices(), canary=canary_for(tid),
                metadata={"topic": topic, "entity": entity, "attribute": attr, "K": K, "focus_grade": focus_grade},
            ))
        return items

    def regex_rules(self) -> dict:
        return {"rag.is_relevant": [r"(?!)"]}  # no sensible regex baseline: passages are ids; kept so the rule file exists

    def cost_matrices(self) -> dict[str, dict[str, dict[str, float]]]:
        return {"is_relevant": {"true": {"true": 0.0, "false": 5.0}, "false": {"true": 1.0, "false": 0.0}}}
