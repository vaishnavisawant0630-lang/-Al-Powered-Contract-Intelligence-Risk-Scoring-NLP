"""Orchestrates all heuristic rules, in a fixed order, over a ClauseResult list."""
from __future__ import annotations

import logging

from .clause_rules import RULES

logger = logging.getLogger(__name__)


def apply_heuristics(text: str, clause_results: list, ner_entities: list | None = None) -> list:
    """Apply all heuristic rules in RULES order.

    Parameters
    ----------
    text : str
        The contract text span being classified.
    clause_results : list[ClauseResult]
        Model + calibration output, one entry per clause type.
    ner_entities : list | None
        Entities from Phase 1 NERModel.extract_entities(), used by NER-grounded rules.

    Returns
    -------
    list[ClauseResult]
        Updated results with adjusted confidence values (dataclasses are frozen,
        so this returns new instances rather than mutating in place).
    """
    by_label = {r.clause_type: r for r in clause_results}
    ner_entities = ner_entities or []

    for label_name, rule_fn in RULES:
        result = by_label.get(label_name)
        if result is None:
            continue
        original = result.confidence
        new_conf = rule_fn(text, original, ner_entities)
        if new_conf != original:
            logger.debug(
                "Heuristic boost: rule=%s label=%s original=%.3f new=%.3f",
                rule_fn.__name__, label_name, original, new_conf,
            )
            by_label[label_name] = _with_confidence(result, new_conf)

    # Preserve original ordering of clause_results.
    return [by_label[r.clause_type] for r in clause_results]


def _with_confidence(result, new_confidence: float):
    """Return a copy of a frozen ClauseResult dataclass with confidence replaced."""
    import dataclasses
    present = new_confidence >= 0.5
    return dataclasses.replace(result, confidence=new_confidence, present=present)
