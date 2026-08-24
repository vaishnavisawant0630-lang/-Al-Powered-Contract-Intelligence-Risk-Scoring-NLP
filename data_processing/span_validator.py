"""
data_processing/span_validator.py
===================================
Validates and resolves overlapping or misaligned annotation spans.

Three checks in order:
  1. Bounds check   — remove spans where end > len(text) or start < 0 or start >= end
  2. Token alignment — shift ±1 char to align to whitespace boundaries
  3. Overlap resolution — keep longer span (spaCy forbids overlapping NER spans)

Every discarded span is recorded as a SpanConflict for the audit trail.

Pure function design — SpanValidator.validate() has no I/O and no state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A single annotation span."""
    start:      int
    end:        int
    label:      str
    text:       str = ""     # surface text (filled during validation)
    confidence: float = 1.0

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SpanConflict:
    """Audit record for every discarded span."""
    doc_id:    str
    reason:    str        # "out_of_bounds" | "overlap_shorter" | "empty_span" | "inverted"
    kept:      Entity | None     # None when the discarded span had no replacement
    discarded: Entity


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class SpanValidator:
    """
    Validates and resolves entity span issues.

    All methods are @staticmethod — no instantiation needed.

    Usage
    -----
        cleaned, conflicts = SpanValidator.validate(
            text=context_text,
            entities=raw_entities,
            doc_id="cuad_contract_001",
        )
    """

    @staticmethod
    def validate(
        text:     str,
        entities: list[Entity],
        doc_id:   str = "",
    ) -> tuple[list[Entity], list[SpanConflict]]:
        """
        Validate and clean entity spans.

        Pipeline:
          bounds_check → token_alignment → overlap_resolution

        Parameters
        ----------
        text : str
            Source contract text.
        entities : list[Entity]
            Raw (possibly invalid) entity spans.
        doc_id : str
            Document identifier for audit records.

        Returns
        -------
        tuple[list[Entity], list[SpanConflict]]
            (clean entities, conflict audit records)
        """
        if not entities:
            return [], []

        if not text and entities:
            logger.warning("validate called with empty text but non-empty entities — discarding all")
            conflicts = [
                SpanConflict(doc_id=doc_id, reason="empty_text", kept=None, discarded=e)
                for e in entities
            ]
            return [], conflicts

        all_conflicts: list[SpanConflict] = []

        # Step 1 — Bounds check
        valid, bounds_conflicts = SpanValidator._check_bounds(
            entities, len(text), doc_id
        )
        all_conflicts.extend(bounds_conflicts)

        # Step 2 — Token alignment (shift ±1 to whitespace boundary)
        aligned = SpanValidator._align_to_tokens(valid, text)

        # Step 3 — Overlap resolution
        clean, overlap_conflicts = SpanValidator._resolve_overlaps(aligned, doc_id)
        all_conflicts.extend(overlap_conflicts)

        # Fill surface text on clean entities
        for ent in clean:
            if not ent.text:
                ent.text = text[ent.start: ent.end]

        return clean, all_conflicts

    # ── Step 1: Bounds check ──────────────────────────────────────────────

    @staticmethod
    def _check_bounds(
        entities:    list[Entity],
        text_length: int,
        doc_id:      str,
    ) -> tuple[list[Entity], list[SpanConflict]]:
        """Remove entities with invalid offsets."""
        valid:     list[Entity]       = []
        conflicts: list[SpanConflict] = []

        for ent in entities:
            # Empty or inverted span
            if ent.start >= ent.end:
                conflicts.append(SpanConflict(
                    doc_id=doc_id, reason="empty_span",
                    kept=None, discarded=ent,
                ))
                continue

            # Negative start
            if ent.start < 0:
                conflicts.append(SpanConflict(
                    doc_id=doc_id, reason="out_of_bounds",
                    kept=None, discarded=ent,
                ))
                continue

            # End exceeds text
            if ent.end > text_length:
                conflicts.append(SpanConflict(
                    doc_id=doc_id, reason="out_of_bounds",
                    kept=None, discarded=ent,
                ))
                continue

            valid.append(ent)

        return valid, conflicts

    # ── Step 2: Token alignment ───────────────────────────────────────────

    @staticmethod
    def _align_to_tokens(entities: list[Entity], text: str) -> list[Entity]:
        """
        Shift span boundaries by ±1 character to align to whitespace boundaries.

        CUAD annotations are sometimes off-by-one due to tokenisation in the
        original annotation tool. This prevents spaCy from complaining about
        spans that start/end mid-token.

        Rule:
          - If text[start] is whitespace → move start forward by 1
          - If text[end-1] is whitespace → move end backward by 1
          - Clamp to [0, len(text)] after adjustment
        """
        aligned = []
        n = len(text)

        for ent in entities:
            start = ent.start
            end   = ent.end

            # Trim leading whitespace
            while start < end and text[start].isspace():
                start += 1

            # Trim trailing whitespace
            while end > start and text[end - 1].isspace():
                end -= 1

            # Clamp
            start = max(0, min(start, n))
            end   = max(0, min(end, n))

            if start < end:
                aligned.append(Entity(
                    start=start, end=end,
                    label=ent.label, confidence=ent.confidence,
                ))

        return aligned

    # ── Step 3: Overlap resolution ────────────────────────────────────────

    @staticmethod
    def _resolve_overlaps(
        entities: list[Entity],
        doc_id:   str,
    ) -> tuple[list[Entity], list[SpanConflict]]:
        """
        Resolve overlapping spans using interval sweep.

        Strategy: keep the LONGER span, discard the shorter one.

        Algorithm
        ---------
        1. Sort by (start ASC, length DESC) — longer spans come first on ties
        2. Maintain accepted list + last_end pointer
        3. For each span:
              if span.start >= last_end → accept
              else (overlap) → compare lengths:
                  if current longer → swap (remove last accepted, accept current)
                  else             → discard current
        """
        if not entities:
            return [], []

        # Sort: start ASC, length DESC (so longer spans win on ties)
        sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))

        accepted:  list[Entity]       = []
        conflicts: list[SpanConflict] = []
        last_end   = -1

        for ent in sorted_ents:
            if ent.start >= last_end:
                # No overlap — accept
                accepted.append(ent)
                last_end = ent.end
            else:
                # Overlap with last accepted span
                prev = accepted[-1]

                if ent.length > prev.length:
                    # Current span is longer → replace previous
                    conflicts.append(SpanConflict(
                        doc_id=doc_id,
                        reason="overlap_shorter",
                        kept=ent,
                        discarded=prev,
                    ))
                    accepted.pop()
                    accepted.append(ent)
                    last_end = ent.end
                else:
                    # Previous span is longer or equal → discard current
                    conflicts.append(SpanConflict(
                        doc_id=doc_id,
                        reason="overlap_shorter",
                        kept=prev,
                        discarded=ent,
                    ))

        return accepted, conflicts

    # ── Convenience class methods ─────────────────────────────────────────

    @classmethod
    def from_cuad_answers(
        cls,
        answers:     dict,
        label:       str,
        text:        str,
        doc_id:      str = "",
    ) -> tuple[list[Entity], list[SpanConflict]]:
        """
        Build and validate entities from a raw CUAD answers dict.

        Parameters
        ----------
        answers : dict
            {"text": [str], "answer_start": [int]}
        label : str
            NER label to assign (e.g. "ORG", "DATE").
        text : str
            Source contract text.
        doc_id : str
        """
        raw: list[Entity] = []
        for span_text, start in zip(
            answers.get("text", []),
            answers.get("answer_start", []),
        ):
            if span_text:
                raw.append(Entity(
                    start=start,
                    end=start + len(span_text),
                    label=label,
                    text=span_text,
                ))
        return cls.validate(text=text, entities=raw, doc_id=doc_id)
