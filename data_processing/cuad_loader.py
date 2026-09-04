"""
data_processing/cuad_loader.py

CUAD dataset loader.

Primary source:
    data/raw/cuad_v1.json

Also checks:
    data/raw/CUADv1.json
    data/raw/DATA RAW/data/CUADv1.json

Returns normalized Python dictionaries.

No Hugging Face datasets dependency is required.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Possible CUAD locations
POSSIBLE_CUAD_PATHS = [
    DATA_RAW_DIR / "cuad_v1.json",
    DATA_RAW_DIR / "CUADv1.json",
    DATA_RAW_DIR / "DATA RAW" / "data" / "CUADv1.json",
]

TRAIN_RATIO = 0.85
RANDOM_SEED = 42


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def load_cuad(
    local_path: str | Path | None = None,
    train_ratio: float = TRAIN_RATIO,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:

    loader = CuadLoader(
        local_path=local_path,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    return loader.load()


# ============================================================
# CUAD LOADER
# ============================================================

class CuadLoader:

    def __init__(
        self,
        local_path: str | Path | None = None,
        train_ratio: float = TRAIN_RATIO,
        random_seed: int = RANDOM_SEED,
    ) -> None:

        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self._schema_logged = False

        # ----------------------------------------------------
        # Determine local CUAD path
        # ----------------------------------------------------

        if local_path:
            self.local_path = Path(local_path)
        else:
            self.local_path = self._find_cuad_file()

    # ========================================================
    # FIND CUAD FILE
    # ========================================================

    def _find_cuad_file(self) -> Path:

        logger.info("Searching for local CUADv1.json...")

        for path in POSSIBLE_CUAD_PATHS:

            logger.info("Checking: %s", path)

            if path.exists():
                logger.info(
                    "Found CUAD file: %s",
                    path
                )
                return path

        # ----------------------------------------------------
        # Recursive search as final local fallback
        # ----------------------------------------------------

        logger.info(
            "CUAD not found in standard locations. "
            "Searching data/raw recursively..."
        )

        if DATA_RAW_DIR.exists():

            for path in DATA_RAW_DIR.rglob("*.json"):

                filename = path.name.lower()

                if filename in {
                    "cuadv1.json",
                    "cuad_v1.json",
                    "cuad.json",
                }:

                    logger.info(
                        "Found CUAD file: %s",
                        path
                    )

                    return path

        raise FileNotFoundError(
            "\nCUADv1.json was not found.\n\n"
            "Please place the CUAD file at:\n\n"
            f"  {DATA_RAW_DIR / 'cuad_v1.json'}\n\n"
            "Example project structure:\n\n"
            "data/\n"
            "└── raw/\n"
            "    └── cuad_v1.json\n"
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self) -> tuple[list[dict], list[dict]]:

        logger.info("=" * 60)
        logger.info("LOADING CUAD DATASET")
        logger.info("=" * 60)

        raw_samples = self._load_local_json(self.local_path)

        # ----------------------------------------------------
        # Filter short contexts
        # ----------------------------------------------------

        filtered = [
            sample
            for sample in raw_samples
            if len(sample.get("context", "")) >= 50
        ]

        discarded = len(raw_samples) - len(filtered)

        if discarded:
            logger.warning(
                "Discarded %d samples with context < 50 characters",
                discarded,
            )

        if not filtered:
            raise ValueError(
                "No valid CUAD samples were found."
            )

        logger.info(
            "Loaded %d CUAD samples",
            len(filtered)
        )

        logger.info(
            "Contracts: %d",
            len(set(sample["title"] for sample in filtered))
        )

        # ----------------------------------------------------
        # Train/dev split
        # ----------------------------------------------------

        train, dev = self._split(filtered)

        logger.info(
            "Split complete: train=%d dev=%d",
            len(train),
            len(dev)
        )

        return train, dev

    # ========================================================
    # LOAD LOCAL JSON
    # ========================================================

    def _load_local_json(self, path: Path) -> list[dict]:

        if not path.exists():

            raise FileNotFoundError(
                f"CUAD file does not exist:\n{path}"
            )

        logger.info(
            "Loading CUAD from local file:\n%s",
            path
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                raw = json.load(file)

        except json.JSONDecodeError as exc:

            raise ValueError(
                f"Invalid JSON file:\n{path}\n\n"
                f"JSON error: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Validate CUAD structure
        # ----------------------------------------------------

        if "data" not in raw:

            raise ValueError(
                "Invalid CUADv1.json format.\n"
                "Expected a top-level 'data' field."
            )

        samples: list[dict] = []

        # ----------------------------------------------------
        # SQuAD-style CUAD format
        # ----------------------------------------------------

        for entry in raw["data"]:

            title = entry.get(
                "title",
                "unknown"
            )

            paragraphs = entry.get(
                "paragraphs",
                []
            )

            for paragraph in paragraphs:

                context = paragraph.get(
                    "context",
                    ""
                )

                qas = paragraph.get(
                    "qas",
                    []
                )

                for qa in qas:

                    answers = qa.get(
                        "answers",
                        []
                    )

                    answer_texts = []
                    answer_starts = []

                    for answer in answers:

                        text = answer.get(
                            "text",
                            ""
                        )

                        start = answer.get(
                            "answer_start",
                            0
                        )

                        if text:

                            answer_texts.append(text)
                            answer_starts.append(start)

                    sample = {
                        "id": str(
                            qa.get(
                                "id",
                                ""
                            )
                        ),

                        "title": str(title),

                        "context": str(context),

                        "question": str(
                            qa.get(
                                "question",
                                ""
                            )
                        ),

                        "answers": {
                            "text": answer_texts,
                            "answer_start": answer_starts,
                        },
                    }

                    samples.append(sample)

        if not samples:

            raise ValueError(
                "CUAD JSON was loaded successfully, "
                "but it contains zero QA samples."
            )

        self._log_schema_info(samples)

        return samples

    # ========================================================
    # TRAIN / DEV SPLIT
    # ========================================================

    def _split(
        self,
        samples: list[dict],
    ) -> tuple[list[dict], list[dict]]:

        titles = sorted(
            set(
                sample["title"]
                for sample in samples
            )
        )

        if len(titles) < 2:

            raise ValueError(
                "CUAD must contain at least 2 contracts "
                "to create train/dev splits."
            )

        rng = random.Random(
            self.random_seed
        )

        rng.shuffle(titles)

        n_train = max(
            1,
            int(
                len(titles)
                * self.train_ratio
            )
        )

        # Make sure dev has at least one contract
        if n_train >= len(titles):

            n_train = len(titles) - 1

        train_titles = set(
            titles[:n_train]
        )

        dev_titles = set(
            titles[n_train:]
        )

        train = [
            sample
            for sample in samples
            if sample["title"] in train_titles
        ]

        dev = [
            sample
            for sample in samples
            if sample["title"] in dev_titles
        ]

        logger.info(
            "Document-level split:"
        )

        logger.info(
            "  Train contracts: %d",
            len(train_titles)
        )

        logger.info(
            "  Dev contracts: %d",
            len(dev_titles)
        )

        logger.info(
            "  Train samples: %d",
            len(train)
        )

        logger.info(
            "  Dev samples: %d",
            len(dev)
        )

        return train, dev

    # ========================================================
    # SCHEMA INFORMATION
    # ========================================================

    def _log_schema_info(
        self,
        samples: list[dict],
    ) -> None:

        if self._schema_logged or not samples:
            return

        sample = samples[0]

        logger.info(
            "CUAD schema: %s",
            list(sample.keys())
        )

        logger.info(
            "Total QA rows: %d",
            len(samples)
        )

        logger.info(
            "Total contracts: %d",
            len(
                set(
                    sample["title"]
                    for sample in samples
                )
            )
        )

        self._schema_logged = True

    # ========================================================
    # SCHEMA INFO PUBLIC METHOD
    # ========================================================

    def schema_info(self) -> dict:

        train, dev = self.load()

        sample = train[0] if train else {}

        return {
            "features": list(sample.keys()),

            "num_rows": {
                "train": len(train),
                "dev": len(dev),
            },

            "total_contracts": len(
                set(
                    sample["title"]
                    for sample in train + dev
                )
            ),
        }