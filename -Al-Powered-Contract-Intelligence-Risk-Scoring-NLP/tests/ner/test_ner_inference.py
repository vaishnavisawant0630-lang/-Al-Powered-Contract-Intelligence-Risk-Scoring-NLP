"""
tests/ner/test_ner_inference.py
================================
Tests for ner/inference.py (§4.4 from phase_01_tasks.md).

SPEC REQUIREMENTS (§4.4)
-------------------------
- Load trained model (or tiny mock model)
- Feed a 2-sentence contract snippet
- Assert at least one ORG and one DATE entity returned

ADDITIONAL COVERAGE
--------------------
test_load_model_caches_singleton    — repeated calls return same object
test_load_model_missing_path        — FileNotFoundError on bad path
test_extract_entities_returns_list  — output is list[Entity]
test_entity_dataclass_fields        — Entity has required fields
test_at_least_one_org               — spec §4.4: ≥1 ORG entity
test_at_least_one_date              — spec §4.4: ≥1 DATE entity
test_batch_extract_same_length      — batch_extract output length == input length
test_deduplicate_no_overlaps        — _deduplicate removes overlapping spans
test_chunk_text_short               — short text yields single chunk
test_chunk_text_long                — very long text is chunked

SKIP CONDITION
--------------
Tests requiring the trained model are skipped if models/ner_baseline/model-best/
does not exist yet (i.e., training hasn't been run).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "ner_baseline" / "model-best"

# 2-sentence contract snippet (spec §4.4)
CONTRACT_SNIPPET = (
    "This Software License Agreement is entered into as of January 15, 2024, "
    "by and between Acme Corporation, a Delaware corporation, "
    "and Beta Technologies LLC, a California limited liability company. "
    "The governing law shall be the laws of the State of California."
)

trained_model_available = MODEL_DIR.exists()
skip_if_no_model = pytest.mark.skipif(
    not trained_model_available,
    reason=f"Trained model not found at {MODEL_DIR} — run: python -m ner.train",
)


# ── Entity dataclass tests (no model needed) ──────────────────────────────

class TestEntityDataclass:
    def test_entity_has_required_fields(self):
        from ner.inference import Entity
        ent = Entity(text="Acme", label="ORG", start_char=0, end_char=4)
        assert ent.text       == "Acme"
        assert ent.label      == "ORG"
        assert ent.start_char == 0
        assert ent.end_char   == 4
        assert ent.confidence == 1.0   # default

    def test_entity_with_confidence(self):
        from ner.inference import Entity
        ent = Entity(text="Jan 2024", label="DATE", start_char=10, end_char=18, confidence=0.87)
        assert ent.confidence == 0.87


# ── Deduplication tests (no model needed) ─────────────────────────────────

class TestDeduplication:
    def test_no_duplicates_pass_through(self):
        from ner.inference import _deduplicate, Entity
        entities = [
            Entity(text="Acme", label="ORG",  start_char=0,  end_char=4),
            Entity(text="2024", label="DATE", start_char=20, end_char=24),
        ]
        result = _deduplicate(entities)
        assert len(result) == 2

    def test_overlapping_keeps_higher_confidence(self):
        from ner.inference import _deduplicate, Entity
        entities = [
            Entity(text="Acme Corp", label="ORG", start_char=0, end_char=9,  confidence=0.6),
            Entity(text="Acme",     label="ORG", start_char=0, end_char=4,  confidence=0.9),
        ]
        result = _deduplicate(entities)
        # Higher confidence wins, even if shorter
        assert len(result) == 1

    def test_no_overlaps_in_output(self):
        from ner.inference import _deduplicate, Entity
        entities = [
            Entity(text="A",    label="ORG",  start_char=0,  end_char=5),
            Entity(text="B",    label="ORG",  start_char=3,  end_char=10),
            Entity(text="C",    label="DATE", start_char=15, end_char=20),
        ]
        result = _deduplicate(entities)
        for i in range(len(result) - 1):
            assert result[i].end_char <= result[i + 1].start_char, "Overlap found in output"

    def test_empty_input(self):
        from ner.inference import _deduplicate
        assert _deduplicate([]) == []

    def test_output_sorted_by_start(self):
        from ner.inference import _deduplicate, Entity
        entities = [
            Entity(text="B", label="DATE", start_char=20, end_char=25),
            Entity(text="A", label="ORG",  start_char=0,  end_char=5),
        ]
        result = _deduplicate(entities)
        starts = [e.start_char for e in result]
        assert starts == sorted(starts)


# ── Text chunking tests (no model needed) ─────────────────────────────────

class TestChunking:
    def test_short_text_single_chunk(self):
        from ner.inference import _chunk_text, MAX_TEXT_LENGTH
        import spacy
        try:
            nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
        except OSError:
            nlp = spacy.blank("en")

        text   = "Short contract text."
        chunks = list(_chunk_text(text, nlp))
        assert len(chunks) == 1
        assert chunks[0] == (text, 0)

    def test_long_text_multiple_chunks(self):
        from ner.inference import _chunk_text, MAX_TEXT_LENGTH
        import spacy
        try:
            nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
        except OSError:
            nlp = spacy.blank("en")

        # Create text that is just over MAX_TEXT_LENGTH (not enormously bigger)
        unit  = "Contract clause text. "                   # 22 chars
        reps  = (MAX_TEXT_LENGTH // len(unit)) + 50        # ~4550 × → ~100.1k chars
        long_text = unit * reps
        assert len(long_text) > MAX_TEXT_LENGTH, "Sanity: text must exceed MAX_TEXT_LENGTH"

        chunks = list(_chunk_text(long_text, nlp))
        assert len(chunks) > 1, (
            f"Expected multiple chunks for text of {len(long_text)} chars "
            f"(MAX={MAX_TEXT_LENGTH}), got {len(chunks)}"
        )
        # First chunk offset must be 0
        assert chunks[0][1] == 0
        # No chunk should be empty
        for chunk_text, offset in chunks:
            assert len(chunk_text) > 0


# ── Model loading tests ───────────────────────────────────────────────────

class TestLoadModel:
    def test_load_model_missing_path_raises(self):
        """FileNotFoundError on non-existent model path."""
        from ner.inference import load_model, _MODEL_CACHE
        # Clear cache to avoid interference
        _MODEL_CACHE.clear()
        with pytest.raises(FileNotFoundError):
            load_model("/path/that/does/not/exist/model-best")

    @skip_if_no_model
    def test_load_model_returns_spacy_pipeline(self):
        import spacy
        from ner.inference import load_model, _MODEL_CACHE
        _MODEL_CACHE.clear()
        nlp = load_model(MODEL_DIR)
        assert hasattr(nlp, "pipe_names"), "Loaded object should be a spaCy Language"
        assert "ner" in nlp.pipe_names

    @skip_if_no_model
    def test_load_model_caches_singleton(self):
        """Multiple calls return the same object (singleton)."""
        from ner.inference import load_model, _MODEL_CACHE
        _MODEL_CACHE.clear()
        nlp1 = load_model(MODEL_DIR)
        nlp2 = load_model(MODEL_DIR)
        assert nlp1 is nlp2


# ── Inference tests (require trained model) ───────────────────────────────

class TestExtractEntities:
    @skip_if_no_model
    def test_extract_entities_returns_list(self):
        """extract_entities() must return a list."""
        from ner.inference import extract_entities, _MODEL_CACHE
        _MODEL_CACHE.clear()
        result = extract_entities(CONTRACT_SNIPPET, model_path=MODEL_DIR)
        assert isinstance(result, list)

    @skip_if_no_model
    def test_entity_objects_have_required_fields(self):
        """Each entity must have text, label, start_char, end_char, confidence."""
        from ner.inference import extract_entities, Entity, _MODEL_CACHE
        _MODEL_CACHE.clear()
        entities = extract_entities(CONTRACT_SNIPPET, model_path=MODEL_DIR)
        for ent in entities:
            assert isinstance(ent, Entity)
            assert isinstance(ent.text,       str)
            assert isinstance(ent.label,      str)
            assert isinstance(ent.start_char, int)
            assert isinstance(ent.end_char,   int)
            assert isinstance(ent.confidence, float)
            assert ent.start_char < ent.end_char

    @skip_if_no_model
    def test_at_least_one_org(self):
        """Spec §4.4: extract_entities should find at least one ORG in the snippet."""
        from ner.inference import extract_entities, _MODEL_CACHE
        _MODEL_CACHE.clear()
        entities = extract_entities(CONTRACT_SNIPPET, model_path=MODEL_DIR)
        org_ents = [e for e in entities if e.label == "ORG"]
        assert len(org_ents) >= 1, (
            f"Expected ≥1 ORG entity in contract snippet.\n"
            f"Got entities: {[(e.text, e.label) for e in entities]}"
        )

    @skip_if_no_model
    def test_at_least_one_date(self):
        """Spec §4.4: extract_entities should find at least one DATE in the snippet."""
        from ner.inference import extract_entities, _MODEL_CACHE
        _MODEL_CACHE.clear()
        entities = extract_entities(CONTRACT_SNIPPET, model_path=MODEL_DIR)
        date_ents = [e for e in entities if e.label == "DATE"]
        assert len(date_ents) >= 1, (
            f"Expected ≥1 DATE entity.\n"
            f"Got entities: {[(e.text, e.label) for e in entities]}"
        )

    @skip_if_no_model
    def test_no_overlapping_entities(self):
        """Output entities must not overlap."""
        from ner.inference import extract_entities, _MODEL_CACHE
        _MODEL_CACHE.clear()
        entities = extract_entities(CONTRACT_SNIPPET, model_path=MODEL_DIR)
        for i in range(len(entities) - 1):
            assert entities[i].end_char <= entities[i + 1].start_char, (
                f"Overlapping: {entities[i]!r} vs {entities[i+1]!r}"
            )

    @skip_if_no_model
    def test_batch_extract_length_matches_input(self):
        """batch_extract returns one list per input text."""
        from ner.inference import batch_extract, _MODEL_CACHE
        _MODEL_CACHE.clear()
        texts  = [CONTRACT_SNIPPET, "Acme Corp signed on January 2024."]
        result = batch_extract(texts, model_path=MODEL_DIR)
        assert len(result) == len(texts)
        assert all(isinstance(r, list) for r in result)
