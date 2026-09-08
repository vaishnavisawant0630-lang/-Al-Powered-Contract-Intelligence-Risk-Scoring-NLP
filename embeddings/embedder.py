"""
embeddings/embedder.py
========================
Thin wrapper around a sentence-transformers model, loaded once and cached
(same singleton pattern as ner/inference.py's load_model()).
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from api.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def load_embedder():
    """Load and cache the sentence-transformers model (thread-safe singleton)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string. Truncates very long contracts to the
    first ~6000 chars (document-level embedding, not full re-indexing of
    every clause)."""
    model = load_embedder()
    snippet = text[:6000]
    vec = model.encode(snippet, normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")
