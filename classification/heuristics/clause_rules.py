"""Individual heuristic rules applied after model prediction + calibration.

Each rule is a pure function: (text, current_confidence, ner_entities) -> new_confidence.
Rules only ever *raise* confidence — they never lower it — since they encode
high-precision domain knowledge that should override statistical under-scoring,
not compete with the model's own uncertainty.
"""
from __future__ import annotations

import re

JURISDICTION_RE = re.compile(
    r"\b(Delaware|New York|California|England and Wales|"
    r"Texas|Illinois|Washington|Singapore|Hong Kong)\b",
    re.IGNORECASE,
)

RENEWAL_RE = re.compile(
    r"\b(automatically renew|auto.renew|unless terminated|"
    r"evergreen|successive term|rolling renewal)\b",
    re.IGNORECASE,
)

CURRENCY_RE = re.compile(
    r"(\$[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|thousand))?|"
    r"USD\s?[\d,]+|EUR\s?[\d,]+)",
    re.IGNORECASE,
)

TERMINATION_PHRASE_A_RE = re.compile(r"either party may terminate", re.IGNORECASE)
TERMINATION_PHRASE_B_RE = re.compile(r"(without cause|for any reason|for convenience|for its convenience)", re.IGNORECASE)

LIABILITY_CAP_RE = re.compile(r"shall not exceed", re.IGNORECASE)


def rule_governing_law(text: str, confidence: float, ner_entities: list | None = None) -> float:
    if confidence >= 0.40 and JURISDICTION_RE.search(text):
        return max(confidence, 0.85)
    return confidence


def rule_auto_renewal(text: str, confidence: float, ner_entities: list | None = None) -> float:
    if confidence >= 0.35 and RENEWAL_RE.search(text):
        return max(confidence, 0.80)
    return confidence


def rule_expiration_date(text: str, confidence: float, ner_entities: list | None = None) -> float:
    if confidence < 0.30:
        return confidence
    entities = ner_entities or []
    has_date = any(getattr(e, "label", getattr(e, "get", lambda *_: None)("label", None)) == "DATE"
                   if not hasattr(e, "label") else e.label == "DATE" for e in entities)
    if has_date:
        return max(confidence, 0.75)
    return confidence


def rule_termination_for_convenience(text: str, confidence: float, ner_entities: list | None = None) -> float:
    match_a = TERMINATION_PHRASE_A_RE.search(text)
    match_b = TERMINATION_PHRASE_B_RE.search(text)
    if match_a and match_b and abs(match_a.start() - match_b.start()) <= 100:
        return max(confidence, 0.90)
    return confidence


def rule_limitation_of_liability(text: str, confidence: float, ner_entities: list | None = None) -> float:
    if CURRENCY_RE.search(text) and LIABILITY_CAP_RE.search(text):
        return max(confidence, 0.88)
    return confidence


# Rule registry — applied in this exact order by post_processor.apply_heuristics.
# Maps clause label name (must match classification/config/clause_labels.json) -> rule fn.
RULES = [
    ("GOVERNING_LAW", rule_governing_law),
    ("RENEWAL_TERM", rule_auto_renewal),
    ("EXPIRATION_DATE", rule_expiration_date),
    ("TERMINATION_FOR_CONVENIENCE", rule_termination_for_convenience),
    ("CAP_ON_LIABILITY", rule_limitation_of_liability),
]
