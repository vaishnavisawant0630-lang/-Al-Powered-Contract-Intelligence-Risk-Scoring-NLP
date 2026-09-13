"""
embeddings/vector_store.py
============================
Local FAISS index standing in for Pinecone/Milvus (phase_03_tasks.md).
Cosine-similarity search via IndexFlatIP on normalised vectors.

Persists to disk as two files under api.config.FAISS_DIR:
    index.faiss   — the FAISS index itself
    ids.json      — ordered list[str] mapping FAISS row -> contract_id
                    (FAISS IndexFlatIP has no native id->metadata store)
"""
from __future__ import annotations

import json
import logging
import threading

import numpy as np

from api.config import EMBEDDING_DIM, FAISS_DIR

logger = logging.getLogger(__name__)

_INDEX_PATH = FAISS_DIR / "index.faiss"
_IDS_PATH   = FAISS_DIR / "ids.json"

_index = None
_ids: list[str] = []
_lock = threading.Lock()


def _load() -> None:
    global _index, _ids
    import faiss
    if _index is not None:
        return
    with _lock:
        if _index is not None:
            return
        if _INDEX_PATH.exists() and _IDS_PATH.exists():
            logger.info("Loading FAISS index from %s", _INDEX_PATH)
            _index = faiss.read_index(str(_INDEX_PATH))
            _ids = json.loads(_IDS_PATH.read_text())
        else:
            logger.info("Creating new FAISS index (dim=%d)", EMBEDDING_DIM)
            _index = faiss.IndexFlatIP(EMBEDDING_DIM)
            _ids = []


def _save() -> None:
    import faiss
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_index, str(_INDEX_PATH))
    _IDS_PATH.write_text(json.dumps(_ids))


def add(contract_id: str, vector: np.ndarray) -> None:
    """Add (or re-add) a contract's embedding to the index."""
    _load()
    with _lock:
        _index.add(vector.reshape(1, -1))
        _ids.append(contract_id)
        _save()


def search(query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    """Returns [(contract_id, cosine_score), ...] sorted by descending score."""
    _load()
    if _index.ntotal == 0:
        return []
    with _lock:
        scores, idxs = _index.search(query_vector.reshape(1, -1), min(top_k, _index.ntotal))
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append((_ids[idx], float(score)))
    return results
