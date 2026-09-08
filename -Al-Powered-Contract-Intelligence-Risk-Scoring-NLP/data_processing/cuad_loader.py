"""
data_processing/cuad_loader.py
================================
CUAD dataset loader — single point of contact for loading the dataset.

Supports two modes:
  1. Local CUADv1.json  (primary — we already have this in data/raw/)
  2. HuggingFace Hub   (fallback: load_dataset("theatticusproject/cuad"))

Returns plain Python dicts with a normalised schema — no dependency
on the `datasets` library anywhere else in the codebase.

NORMALISED SCHEMA (one dict per QA pair)
-----------------------------------------
{
  "id":            str,   # unique QA pair identifier
  "title":         str,   # contract filename (source)
  "context":       str,   # full contract text
  "question":      str,   # clause-type question template
  "answers":       dict,  # {"text": [str], "answer_start": [int]}
                          # empty lists → clause absent in this contract
}

TRAIN / DEV SPLIT
------------------
Document-level split (no contract straddles train and dev):
  - train_split = 0.85  (default) → ~18,982 rows
  - dev_split   = 0.15  (default) → ~3,350 rows
  - random_state = 42
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
_DEFAULT_CUAD_JSON = (
    Path(__file__).parent.parent
    / "data" / "raw" / "DATA RAW" / "data" / "CUADv1.json"
)

# Split config
TRAIN_RATIO   = 0.85
RANDOM_SEED   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def load_cuad(
    local_path: str | Path | None = None,
    train_ratio: float = TRAIN_RATIO,
    random_seed: int   = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Load CUAD and return (train_samples, dev_samples).

    Each sample is a flat dict:
        {id, title, context, question, answers}

    Parameters
    ----------
    local_path : str | Path | None
        Path to CUADv1.json. If None, uses bundled default path,
        then falls back to HuggingFace Hub download.
    train_ratio : float
        Fraction of DOCUMENTS (not rows) to use for training.
    random_seed : int
        Random seed for deterministic split.
    """
    loader = CuadLoader(local_path=local_path, train_ratio=train_ratio,
                        random_seed=random_seed)
    return loader.load()


# ─────────────────────────────────────────────────────────────────────────────
# Main Loader class
# ─────────────────────────────────────────────────────────────────────────────

class CuadLoader:
    """
    Loads and splits the CUAD dataset.

    Usage
    -----
        loader = CuadLoader()
        train_samples, dev_samples = loader.load()
        print(train_samples[0].keys())
        # dict_keys(['id', 'title', 'context', 'question', 'answers'])
    """

    def __init__(
        self,
        local_path: str | Path | None = None,
        train_ratio: float = TRAIN_RATIO,
        random_seed: int   = RANDOM_SEED,
    ) -> None:
        self.local_path  = Path(local_path) if local_path else _DEFAULT_CUAD_JSON
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self._schema_logged = False

    # ── Public API ────────────────────────────────────────────────────────

    def load(self) -> tuple[list[dict], list[dict]]:
        """
        Load CUAD and return (train_samples, dev_samples).

        Tries local CUADv1.json first. Falls back to HuggingFace Hub
        if file not found.

        Returns
        -------
        tuple[list[dict], list[dict]]
            (train_samples, dev_samples) — each sample is a plain dict
        """
        raw_samples = self._load_from_source()

        # Filter out samples with empty or very short contexts
        filtered = [s for s in raw_samples if len(s.get("context", "")) >= 50]
        discarded = len(raw_samples) - len(filtered)
        if discarded:
            logger.warning(f"Discarded {discarded} samples with context < 50 chars")

        logger.info(
            f"Loaded {len(filtered)} CUAD samples "
            f"({len(set(s['title'] for s in filtered))} contracts)"
        )

        train, dev = self._split(filtered)
        logger.info(f"Split → train={len(train)}  dev={len(dev)}")
        return train, dev

    def schema_info(self) -> dict:
        """Return schema metadata for inspection / logging."""
        train, dev = self.load()
        sample = train[0] if train else {}
        return {
            "features": list(sample.keys()),
            "num_rows":  {"train": len(train), "dev": len(dev)},
            "total_contracts": len(set(s["title"] for s in train + dev)),
        }

    # ── Loading strategy ──────────────────────────────────────────────────

    def _load_from_source(self) -> list[dict]:
        """Load from local JSON first, then HuggingFace Hub."""
        if self.local_path.exists():
            logger.info(f"Loading CUAD from local file: {self.local_path}")
            return self._load_local_json(self.local_path)
        else:
            logger.info("Local CUADv1.json not found — downloading from HuggingFace Hub")
            return self._load_from_huggingface()

    def _load_local_json(self, path: Path) -> list[dict]:
        """
        Load from CUADv1.json (SQuAD-style format).

        CUADv1.json structure:
            {"data": [{"title": str, "paragraphs": [{"context": str, "qas": [...]}]}]}

        Each QA pair becomes one sample row.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples: list[dict] = []
        for entry in raw["data"]:
            title = entry.get("title", "unknown")
            for para in entry.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    samples.append({
                        "id":       qa.get("id", ""),
                        "title":    title,
                        "context":  context,
                        "question": qa.get("question", ""),
                        "answers":  {
                            "text":         [a["text"]         for a in qa.get("answers", [])],
                            "answer_start": [a["answer_start"] for a in qa.get("answers", [])],
                        },
                    })

        self._log_schema_info(samples)
        return samples

    def _load_from_huggingface(self) -> list[dict]:
        """
        Download CUAD from HuggingFace Hub.
        Identifier: "theatticusproject/cuad"
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError(
                "The `datasets` library is required to download CUAD from HuggingFace.\n"
                "Install with: pip install datasets\n"
                f"Or place CUADv1.json at: {_DEFAULT_CUAD_JSON}"
            )

        logger.info("Downloading CUAD from HuggingFace Hub (theatticusproject/cuad)...")
        dataset = load_dataset("theatticusproject/cuad", trust_remote_code=True)
        train_split = dataset["train"]

        # Convert to normalised dicts
        samples: list[dict] = []
        for row in train_split:
            samples.append({
                "id":       row.get("id", ""),
                "title":    row.get("title", ""),
                "context":  row.get("context", ""),
                "question": row.get("question", ""),
                "answers":  row.get("answers", {"text": [], "answer_start": []}),
            })

        self._log_schema_info(samples)
        return samples

    # ── Splitting ─────────────────────────────────────────────────────────

    def _split(
        self, samples: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Deterministic document-level split.

        Stratifies by contract title to guarantee no contract appears
        in both train and dev (prevents data leakage).

        Algorithm:
        1. Collect unique titles
        2. Shuffle with fixed seed
        3. First train_ratio → train_titles, rest → dev_titles
        4. Partition samples by title
        """
        titles = sorted(set(s["title"] for s in samples))

        rng = random.Random(self.random_seed)
        rng.shuffle(titles)

        n_train = max(1, int(len(titles) * self.train_ratio))
        train_titles = set(titles[:n_train])
        dev_titles   = set(titles[n_train:])

        train = [s for s in samples if s["title"] in train_titles]
        dev   = [s for s in samples if s["title"] in dev_titles]

        logger.info(
            f"Document split — "
            f"train_docs={len(train_titles)}  dev_docs={len(dev_titles)}  "
            f"train_rows={len(train)}  dev_rows={len(dev)}"
        )
        return train, dev

    # ── Internal helpers ──────────────────────────────────────────────────

    def _log_schema_info(self, samples: list[dict]) -> None:
        if self._schema_logged or not samples:
            return
        sample = samples[0]
        logger.info(
            "CUAD schema: features=%s  total_rows=%d  contracts=%d",
            list(sample.keys()),
            len(samples),
            len(set(s["title"] for s in samples)),
        )
        self._schema_logged = True
