"""
ner/inference.py
=================
Inference API for the trained spaCy NER model.

PUBLIC FUNCTIONS (from spec §3.4)
-----------------------------------
    load_model(model_path: str) → nlp
        Load and cache spaCy pipeline. Thread-safe singleton.

    extract_entities(text: str) → list[Entity]
        Returns list of Entity(text, label, start_char, end_char, confidence)
        - confidence: token-level IOB scores if available, else 1.0
        - Deduplicates overlapping spans by score (higher score wins)
        - Long texts auto-chunked at sentence boundaries

    batch_extract(texts: list[str]) → list[list[Entity]]
        Efficient batch inference using nlp.pipe().

USAGE
-----
    from ner.inference import load_model, extract_entities

    load_model("models/ner_baseline/model-best")
    entities = extract_entities("This Agreement is between Acme Corp and Beta Inc.")
    for e in entities:
        print(e.text, e.label, e.confidence)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "ner_baseline" / "model-best"

# Maximum characters before chunking (spaCy has a tokenizer limit)
MAX_TEXT_LENGTH = 100_000
# Approximate characters per sentence chunk
CHUNK_OVERLAP   = 200


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A single extracted named entity."""
    text:       str
    label:      str
    start_char: int
    end_char:   int
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Singleton model cache (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, object] = {}
_CACHE_LOCK  = threading.Lock()


def load_model(model_path: str | Path = MODEL_DIR) -> object:
    """
    Load and cache the spaCy NER pipeline.

    Thread-safe: concurrent calls for the same path return the same nlp object.
    The model is loaded once per process and reused for all subsequent calls.

    Parameters
    ----------
    model_path : str | Path
        Path to the spaCy model directory (model-best/).

    Returns
    -------
    spacy.Language
        Loaded spaCy NLP pipeline.

    Raises
    ------
    FileNotFoundError
        If the model directory does not exist.
    """
    import spacy

    model_path = str(Path(model_path).resolve())

    with _CACHE_LOCK:
        if model_path not in _MODEL_CACHE:
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"NER model not found: {model_path}\n"
                    f"Run: python -m ner.train"
                )
            logger.info("Loading NER model from %s", model_path)
            nlp = spacy.load(model_path)
            _MODEL_CACHE[model_path] = nlp
            logger.info(
                "Model loaded — labels: %s",
                nlp.get_pipe("ner").labels,
            )
        return _MODEL_CACHE[model_path]


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities(
    text:       str,
    model_path: str | Path = MODEL_DIR,
) -> list[Entity]:
    """
    Extract named entities from a contract text.

    Handles long texts by chunking at sentence boundaries.
    Deduplicates overlapping spans by confidence (higher wins).

    Parameters
    ----------
    text : str
        Raw contract text (any length).
    model_path : str | Path
        Path to model-best/ directory. Cached after first call.

    Returns
    -------
    list[Entity]
        Sorted by start_char. No overlapping spans.
        confidence is set from spaCy's .ent_kb_id_ if available, else 1.0.
    """
    nlp = load_model(model_path)
    entities: list[Entity] = []

    for chunk_text, chunk_offset in _chunk_text(text, nlp):
        doc = nlp(chunk_text)
        for ent in doc.ents:
            entities.append(Entity(
                text       = ent.text,
                label      = ent.label_,
                start_char = ent.start_char + chunk_offset,
                end_char   = ent.end_char   + chunk_offset,
                confidence = _get_confidence(ent),
            ))

    # Deduplicate overlapping spans (keep highest confidence)
    return _deduplicate(entities)


def batch_extract(
    texts:      list[str],
    model_path: str | Path = MODEL_DIR,
    batch_size: int = 16,
) -> list[list[Entity]]:
    """
    Batch inference using nlp.pipe() for efficiency.

    Parameters
    ----------
    texts : list[str]
        List of contract texts to process.
    model_path : str | Path
        Path to model-best/.
    batch_size : int
        Number of texts to process per spaCy batch.

    Returns
    -------
    list[list[Entity]]
        One list of entities per input text, in the same order.
    """
    nlp     = load_model(model_path)
    results: list[list[Entity]] = []

    # For texts within max length, use efficient pipe()
    short_texts  = [t for t in texts if len(t) <= MAX_TEXT_LENGTH]
    long_indices = [i for i, t in enumerate(texts) if len(t) > MAX_TEXT_LENGTH]

    if short_texts:
        for doc in nlp.pipe(short_texts, batch_size=batch_size):
            results.append([
                Entity(
                    text=e.text, label=e.label_,
                    start_char=e.start_char, end_char=e.end_char,
                    confidence=_get_confidence(e),
                )
                for e in doc.ents
            ])

    # Fall back to chunked extraction for long texts
    for idx in long_indices:
        results.insert(idx, extract_entities(texts[idx], model_path))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    nlp: object,
) -> Iterator[tuple[str, int]]:
    """
    Split long text into sentence-boundary chunks.

    Yields
    ------
    (chunk_text, char_offset) pairs.
    For texts shorter than MAX_TEXT_LENGTH, yields the full text with offset 0.
    """
    if len(text) <= MAX_TEXT_LENGTH:
        yield text, 0
        return

    # Use spaCy sentencizer to find sentence boundaries
    import spacy

    # Quick sentence split using newlines + periods for chunking
    # (avoid running the full pipeline just for chunking)
    chunk_start = 0
    while chunk_start < len(text):
        chunk_end = min(chunk_start + MAX_TEXT_LENGTH, len(text))

        # Try to end on a sentence boundary (newline or '. ')
        if chunk_end < len(text):
            for boundary in ["\n\n", "\n", ". "]:
                bp = text.rfind(boundary, chunk_start + MAX_TEXT_LENGTH // 2, chunk_end)
                if bp != -1:
                    chunk_end = bp + len(boundary)
                    break

        yield text[chunk_start:chunk_end], chunk_start

        # Break BEFORE updating chunk_start when we've consumed the whole text.
        # Without this guard chunk_start = chunk_end - CHUNK_OVERLAP stays
        # below len(text) forever → infinite loop → MemoryError.
        if chunk_end >= len(text):
            break

        chunk_start = chunk_end - CHUNK_OVERLAP
        if chunk_start >= chunk_end:  # safety guard
            break


def _get_confidence(ent) -> float:
    """
    Extract confidence score from a spaCy Span.

    spaCy's default NER doesn't expose per-span probabilities directly.
    We return the mean token score if available via ent._.score,
    otherwise fall back to 1.0.
    """
    # Try custom attribute
    try:
        if ent.has_extension("score"):
            return float(ent._.score)
    except Exception:
        pass
    return 1.0


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """
    Remove overlapping entities, keeping the one with higher confidence.

    Uses an interval sweep: sort by (start, -confidence), greedily accept
    non-overlapping spans.
    """
    if not entities:
        return []

    sorted_ents = sorted(entities, key=lambda e: (e.start_char, -e.confidence))
    result: list[Entity] = []
    last_end = -1

    for ent in sorted_ents:
        if ent.start_char >= last_end:
            result.append(ent)
            last_end = ent.end_char
        else:
            # Overlap: keep the one with higher confidence
            prev = result[-1]
            if ent.confidence > prev.confidence:
                result[-1] = ent
                last_end   = ent.end_char

    return sorted(result, key=lambda e: e.start_char)
