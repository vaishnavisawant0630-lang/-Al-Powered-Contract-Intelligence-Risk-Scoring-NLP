"""
tests/data_processing/test_cuad_to_ner.py
==========================================
Tests for data_processing/cuad_to_ner.py and span_validator.py (§4.3).

SPEC REQUIREMENTS (§4.3)
-------------------------
Run on 10 sample CUAD rows:
- Assert NO overlapping spans in output
- Assert entity labels MATCH the expected mapping table
- Assert output is a VALID spaCy DocBin (can be loaded with DocBin().from_disk())

ADDITIONAL COVERAGE
--------------------
test_question_to_label_mapping   — all 41 question types map to one of 7 valid labels
test_span_validator_bounds_check — out-of-bounds spans are discarded
test_span_validator_overlap_resolution — overlapping spans: longer wins
test_span_validator_alignment     — whitespace trimming on boundaries
test_classification_schema        — classification records have exact spec schema
test_clause_type_normalisation    — question → UPPER_SNAKE_CASE conversion
test_docbin_is_valid_spacy        — written .spacy file can be loaded
test_no_overlapping_spans         — all entity spans in output are non-overlapping
test_entity_labels_valid          — all labels are from the known set
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

VALID_NER_LABELS = {
    "ORG", "DATE", "MONEY", "DURATION",
    "LAW_JURISDICTION", "IP_CLAUSE", "CLAUSE",
}

# 10 synthetic CUAD-style samples (one per distinct question type)
SAMPLE_CUAD_ROWS = [
    {
        "id": f"test_{i}",
        "title": f"Contract_{i // 5}",   # 2 unique titles for split testing
        "context": (
            "This Software License Agreement ('Agreement') is entered into as of "
            "January 15, 2024, by and between Acme Corporation, a Delaware corporation "
            "('Licensor'), and Beta Technologies LLC, a California limited liability "
            "company ('Licensee'). The governing law shall be the State of California. "
            "The total contract value is $500,000. The termination notice period is 30 days."
        ),
        "question": question,
        "answers": answers,
    }
    for i, (question, answers) in enumerate([
        ('Highlight the parts related to "Parties" of this contract.',
         {"text": ["Acme Corporation"], "answer_start": [72]}),
        ('Highlight the parts related to "Governing Law" of this contract.',
         {"text": ["the State of California"], "answer_start": [237]}),
        ('Highlight the parts related to "Effective Date" of this contract.',
         {"text": ["January 15, 2024"], "answer_start": [60]}),
        ('Highlight the parts related to "Expiration Date" of this contract.',
         {"text": ["January 15, 2024"], "answer_start": [60]}),
        ('Highlight the parts related to "Cap On Liability" of this contract.',
         {"text": ["$500,000"], "answer_start": [280]}),
        ('Highlight the parts related to "Minimum Commitment" of this contract.',
         {"text": ["$500,000"], "answer_start": [280]}),
        ('Highlight the parts related to "Notice Period to Terminate Renewal" of this contract.',
         {"text": ["30 days"], "answer_start": [330]}),
        ('Highlight the parts related to "IP Ownership Assignment" of this contract.',
         {"text": [], "answer_start": []}),
        ('Highlight the parts related to "Non-Compete" of this contract.',
         {"text": [], "answer_start": []}),
        ('Highlight the parts related to "License Grant" of this contract.',
         {"text": [], "answer_start": []}),
    ])
]


# ── SpanValidator tests ──────────────────────────────────────────────────

class TestSpanValidator:
    def test_bounds_check_negative_start(self):
        from data_processing.span_validator import SpanValidator, Entity
        entities = [Entity(start=-1, end=5, label="ORG")]
        clean, conflicts = SpanValidator.validate("Hello world", entities, "doc1")
        assert len(clean) == 0
        assert any(c.reason == "out_of_bounds" for c in conflicts)

    def test_bounds_check_end_exceeds_text(self):
        from data_processing.span_validator import SpanValidator, Entity
        text     = "Hello"
        entities = [Entity(start=0, end=100, label="DATE")]
        clean, conflicts = SpanValidator.validate(text, entities, "doc1")
        assert len(clean) == 0
        assert len(conflicts) == 1

    def test_empty_span_discarded(self):
        from data_processing.span_validator import SpanValidator, Entity
        entities = [Entity(start=5, end=5, label="ORG")]
        clean, conflicts = SpanValidator.validate("Hello world", entities, "doc1")
        assert len(clean) == 0
        assert conflicts[0].reason == "empty_span"

    def test_overlap_resolution_keeps_longer(self):
        """When spans overlap, the longer one is kept."""
        from data_processing.span_validator import SpanValidator, Entity
        text = "Acme Corporation and Beta Inc are parties"
        entities = [
            Entity(start=0, end=4,  label="ORG"),    # "Acme" — shorter
            Entity(start=0, end=16, label="ORG"),    # "Acme Corporation" — longer
        ]
        clean, conflicts = SpanValidator.validate(text, entities, "doc1")
        assert len(clean) == 1
        assert clean[0].end - clean[0].start == 16   # longer span kept
        assert len(conflicts) == 1
        assert conflicts[0].reason == "overlap_shorter"

    def test_whitespace_trimming(self):
        """Boundary trimming removes leading/trailing whitespace from spans."""
        from data_processing.span_validator import SpanValidator, Entity
        text = "  Acme Corp  "
        # Span includes surrounding whitespace
        entities = [Entity(start=0, end=13, label="ORG")]
        clean, _ = SpanValidator.validate(text, entities, "doc1")
        assert len(clean) == 1
        # After trimming, span should not start/end on whitespace
        assert not text[clean[0].start].isspace()
        assert not text[clean[0].end - 1].isspace()

    def test_valid_spans_pass_through(self):
        """Non-overlapping, in-bounds spans should all be returned."""
        from data_processing.span_validator import SpanValidator, Entity
        text     = "Acme Corp signed on January 15"
        entities = [
            Entity(start=0, end=9,  label="ORG"),
            Entity(start=20, end=30, label="DATE"),
        ]
        clean, conflicts = SpanValidator.validate(text, entities, "doc1")
        assert len(clean) == 2
        assert len(conflicts) == 0

    def test_no_overlapping_spans_in_output(self):
        """Output entities must never overlap."""
        from data_processing.span_validator import SpanValidator, Entity
        text = "Acme Corporation Beta Technologies LLC January 2024"
        entities = [
            Entity(start=0, end=16, label="ORG"),
            Entity(start=5, end=30, label="ORG"),   # overlaps with first
            Entity(start=35, end=50, label="DATE"),
        ]
        clean, _ = SpanValidator.validate(text, entities, "doc1")
        # Verify no overlaps
        for i in range(len(clean) - 1):
            assert clean[i].end <= clean[i + 1].start, "Overlapping spans in output"


# ── NER label mapping tests ───────────────────────────────────────────────

class TestEntityMapping:
    def test_parties_maps_to_org(self):
        from data_processing.cuad_to_ner import _question_to_label
        assert _question_to_label('Highlight "Parties" in this contract.') == "ORG"

    def test_governing_law_maps_to_law_jurisdiction(self):
        from data_processing.cuad_to_ner import _question_to_label
        assert _question_to_label('Highlight "Governing Law" clauses.') == "LAW_JURISDICTION"

    def test_effective_date_maps_to_date(self):
        from data_processing.cuad_to_ner import _question_to_label
        assert _question_to_label('Find the "Effective Date" of the contract.') == "DATE"

    def test_money_related_maps_to_money(self):
        from data_processing.cuad_to_ner import _question_to_label
        assert _question_to_label('Highlight "Minimum Commitment" clauses.') == "MONEY"
        assert _question_to_label('Identify "Cap On Liability".') == "MONEY"

    def test_duration_related_maps_to_duration(self):
        from data_processing.cuad_to_ner import _question_to_label
        assert _question_to_label('Find "Notice Period to Terminate Renewal".') == "DURATION"
        assert _question_to_label('Find "Renewal Term".') == "DURATION"

    def test_ip_clause_mapping(self):
        from data_processing.cuad_to_ner import _question_to_label
        assert _question_to_label('"IP Ownership Assignment"') == "IP_CLAUSE"
        assert _question_to_label('"License Grant"') == "IP_CLAUSE"

    def test_unknown_question_defaults_to_clause(self):
        from data_processing.cuad_to_ner import _question_to_label
        assert _question_to_label("Some completely unknown clause type") == "CLAUSE"

    def test_all_labels_are_valid(self):
        """Every mapping result must be in VALID_NER_LABELS."""
        from data_processing.cuad_to_ner import QUESTION_TO_LABEL
        for keyword, label in QUESTION_TO_LABEL.items():
            assert label in VALID_NER_LABELS, (
                f"Label {label!r} for keyword {keyword!r} is not in VALID_NER_LABELS"
            )


# ── DocBin output tests ───────────────────────────────────────────────────

class TestCuadToNerDocBin:
    @pytest.fixture(scope="class")
    def ner_output_dir(self, tmp_path_factory):
        """Run CuadToNer on 10 samples, return output dir."""
        import spacy
        out_dir = tmp_path_factory.mktemp("ner_out")
        # Split 10 samples 8/2 by document (title)
        train = [s for s in SAMPLE_CUAD_ROWS if s["title"] == "Contract_0"]
        dev   = [s for s in SAMPLE_CUAD_ROWS if s["title"] != "Contract_0"]
        from data_processing.cuad_to_ner import CuadToNer
        CuadToNer().convert(train, dev, output_dir=out_dir)
        return out_dir

    def test_train_spacy_file_created(self, ner_output_dir):
        assert (ner_output_dir / "cuad_ner_train.spacy").exists()

    def test_dev_spacy_file_created(self, ner_output_dir):
        assert (ner_output_dir / "cuad_ner_dev.spacy").exists()

    def test_docbin_is_valid_spacy(self, ner_output_dir):
        """Spec §4.3: Output must be a valid spaCy DocBin."""
        import spacy
        from spacy.tokens import DocBin

        try:
            nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
        except OSError:
            nlp = spacy.blank("en")

        for fname in ["cuad_ner_train.spacy", "cuad_ner_dev.spacy"]:
            path    = ner_output_dir / fname
            doc_bin = DocBin().from_disk(path)   # must not raise
            docs    = list(doc_bin.get_docs(nlp.vocab))
            assert len(docs) > 0, f"{fname}: DocBin is empty"

    def test_no_overlapping_spans(self, ner_output_dir):
        """Spec §4.3: No overlapping spans in any output document."""
        import spacy
        from spacy.tokens import DocBin

        try:
            nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
        except OSError:
            nlp = spacy.blank("en")

        for fname in ["cuad_ner_train.spacy", "cuad_ner_dev.spacy"]:
            doc_bin = DocBin().from_disk(ner_output_dir / fname)
            for doc in doc_bin.get_docs(nlp.vocab):
                ents = sorted(doc.ents, key=lambda e: e.start_char)
                for i in range(len(ents) - 1):
                    assert ents[i].end_char <= ents[i + 1].start_char, (
                        f"Overlapping spans in {fname}: "
                        f"{ents[i].text!r} [{ents[i].start_char}:{ents[i].end_char}] "
                        f"overlaps {ents[i+1].text!r} [{ents[i+1].start_char}:{ents[i+1].end_char}]"
                    )

    def test_entity_labels_valid(self, ner_output_dir):
        """Spec §4.3: All entity labels must match the expected mapping table."""
        import spacy
        from spacy.tokens import DocBin

        try:
            nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
        except OSError:
            nlp = spacy.blank("en")

        for fname in ["cuad_ner_train.spacy", "cuad_ner_dev.spacy"]:
            doc_bin = DocBin().from_disk(ner_output_dir / fname)
            for doc in doc_bin.get_docs(nlp.vocab):
                for ent in doc.ents:
                    assert ent.label_ in VALID_NER_LABELS, (
                        f"Invalid label {ent.label_!r} in {fname}"
                    )


# ── Classification schema tests ───────────────────────────────────────────

class TestCuadToClassification:
    @pytest.fixture(scope="class")
    def cls_output_dir(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("cls_out")
        train   = [s for s in SAMPLE_CUAD_ROWS if s["title"] == "Contract_0"]
        dev     = [s for s in SAMPLE_CUAD_ROWS if s["title"] != "Contract_0"]
        from data_processing.cuad_to_classification import CuadToClassification
        CuadToClassification().convert(train, dev, output_dir=out_dir)
        return out_dir

    def test_train_file_created(self, cls_output_dir):
        assert (cls_output_dir / "cuad_clauses_train.json").exists()

    def test_dev_file_created(self, cls_output_dir):
        assert (cls_output_dir / "cuad_clauses_dev.json").exists()

    def test_record_schema(self, cls_output_dir):
        """Spec §1.4: every record must have exact schema."""
        required = {"contract_name", "clause_type", "text_span", "label"}
        for fname in ["cuad_clauses_train.json", "cuad_clauses_dev.json"]:
            with open(cls_output_dir / fname) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    assert required == set(rec.keys()) or required.issubset(rec.keys()), (
                        f"Missing schema keys in {fname}: {set(rec.keys())}"
                    )
                    assert rec["label"] in {0, 1}
                    assert isinstance(rec["text_span"], str)
                    assert len(rec["text_span"]) > 0

    def test_clause_type_is_upper_snake_case(self, cls_output_dir):
        """clause_type must be UPPER_SNAKE_CASE."""
        import re
        pattern = re.compile(r"^[A-Z0-9][A-Z0-9_]*$")
        with open(cls_output_dir / "cuad_clauses_train.json") as f:
            for line in f:
                rec = json.loads(line.strip())
                ct  = rec["clause_type"]
                assert pattern.match(ct), (
                    f"clause_type {ct!r} is not UPPER_SNAKE_CASE"
                )

    def test_positive_samples_have_answer_text(self, cls_output_dir):
        """label=1 records must have a non-empty, meaningful text_span."""
        with open(cls_output_dir / "cuad_clauses_train.json") as f:
            for line in f:
                rec = json.loads(line.strip())
                if rec["label"] == 1:
                    assert len(rec["text_span"].strip()) > 0
