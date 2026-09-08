"""
tests/data_processing/test_span_validator.py
=============================================
Unit tests for data_processing/span_validator.py — SpanValidator.

WHAT IS TESTED
--------------
SpanValidator.validate() is a pure function — all tests use synthetic
Entity objects and string inputs. Zero I/O, zero mocking needed.

1. _check_bounds()
   - Keeps entities with valid start/end
   - Discards entity where start < 0
   - Discards entity where end > len(text)
   - Discards entity where start >= end (zero-length or inverted)
   - Returns SpanConflict for each discarded entity

2. _resolve_overlaps()
   - Non-overlapping entities: all kept, no conflicts
   - Two overlapping entities: longer one kept, shorter discarded
   - Three spans where A and B overlap, B and C overlap: resolved correctly
   - Exact duplicate spans: one kept, one discarded (first wins by convention)
   - Spans of equal length: first span kept

3. validate() — integration
   - Returns (cleaned_entities, conflicts) tuple
   - Cleaned entities are sorted by start position
   - No two returned entities overlap (invariant test)
   - SpanConflict.reason == "longer span wins" for overlap resolutions

4. Edge cases
   - Empty entity list → returns ([], [])
   - Single entity → returns ([entity], [])
   - Raises SpanValidationError if text is empty but entities are non-empty

PROPERTY TEST (optional — use hypothesis if installed)
------------------------------------------------------
For any random list of entities with valid offsets:
    cleaned, _ = SpanValidator.validate(text, entities, "test")
    assert all non-overlapping pairs in cleaned
"""

from __future__ import annotations

import pytest

# TODO (implementation): from data_processing.span_validator import SpanValidator
# from core.types import Entity, SpanConflict

# Helpers for tests
def make_entity(label, start, end, text="x" * 100):
    """Create a synthetic Entity for testing."""
    # from core.types import Entity
    # return Entity(label=label, text=text[start:end], start=start, end=end)
    pass


class TestBoundsCheck:
    """Tests for SpanValidator._check_bounds()."""

    def test_valid_entities_kept(self):
        """Entities within bounds are kept without change."""
        pass

    def test_negative_start_discarded(self):
        """Entity with start < 0 is discarded and recorded as SpanConflict."""
        pass

    def test_end_beyond_text_length_discarded(self):
        """Entity with end > len(text) is discarded."""
        pass

    def test_zero_length_span_discarded(self):
        """Entity with start == end is discarded (zero-length)."""
        pass

    def test_inverted_span_discarded(self):
        """Entity with start > end is discarded."""
        pass

    def test_returns_conflict_for_each_discarded(self):
        """One SpanConflict is returned for each discarded entity."""
        pass


class TestResolveOverlaps:
    """Tests for SpanValidator._resolve_overlaps()."""

    def test_non_overlapping_all_kept(self):
        """Non-overlapping entities: all returned, no conflicts."""
        pass

    def test_overlap_longer_wins(self):
        """Shorter of two overlapping spans is discarded."""
        pass

    def test_equal_length_first_wins(self):
        """When two overlapping spans are equal length, first is kept."""
        pass

    def test_three_way_overlap_resolved(self):
        """Three mutually overlapping spans resolve to one winner."""
        pass

    def test_empty_input_returns_empty(self):
        """Empty entity list → ([], [])."""
        pass

    def test_single_entity_no_conflicts(self):
        """Single entity → returned unchanged, no conflicts."""
        pass


class TestValidateIntegration:
    """Integration tests for SpanValidator.validate()."""

    def test_output_entities_are_sorted_by_start(self):
        """Returned entities are sorted by start position."""
        pass

    def test_output_entities_are_non_overlapping(self):
        """No two returned entities overlap (invariant)."""
        pass

    def test_conflict_reason_is_longer_span_wins(self):
        """SpanConflict.reason == 'longer span wins' for overlaps."""
        pass

    def test_raises_if_text_empty_but_entities_nonempty(self):
        """SpanValidationError raised when text is empty but entities exist."""
        pass
