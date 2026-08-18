"""
data_processing/dataset_stats.py
==================================
Prints a comprehensive dataset statistics report after CUAD processing.

REPORT SECTIONS (from spec §1.5)
----------------------------------
1. Entity type → count table       (from cuad_ner_train.spacy)
2. Clause type → positive/negative ratio table
3. Summary: total samples, conflicts resolved, avg text length

Can be run standalone or imported and called from other scripts.

Usage
-----
    # From project root:
    python -m data_processing.dataset_stats

    # Or import:
    from data_processing.dataset_stats import DatasetStats
    DatasetStats.print_report()
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths (relative to project root)
PROCESSED_DIR    = Path("data/processed")
NER_TRAIN_SPACY  = PROCESSED_DIR / "cuad_ner_train.spacy"
CLAUSES_TRAIN    = PROCESSED_DIR / "cuad_clauses_train.json"
CLAUSES_DEV      = PROCESSED_DIR / "cuad_clauses_dev.json"
CUAD_JSON_PATH   = (
    Path("data") / "raw" / "DATA RAW" / "data" / "CUADv1.json"
)


class DatasetStats:
    """
    Computes and prints dataset statistics after CUAD processing.

    Usage
    -----
        DatasetStats.print_report()
    """

    @classmethod
    def print_report(
        cls,
        processed_dir: str | Path = PROCESSED_DIR,
        cuad_json_path: str | Path = CUAD_JSON_PATH,
    ) -> None:
        """Print the full statistics report to stdout."""
        processed_dir   = Path(processed_dir)
        cuad_json_path  = Path(cuad_json_path)

        print()
        print("=" * 65)
        print("  CUAD DATASET STATISTICS REPORT")
        print("=" * 65)

        # ── Section 1: NER entity type counts ────────────────────────────
        cls._print_ner_stats(processed_dir)

        # ── Section 2: Clause type distribution ──────────────────────────
        cls._print_clause_stats(processed_dir)

        # ── Section 3: Summary ────────────────────────────────────────────
        cls._print_summary(processed_dir, cuad_json_path)

        print("=" * 65)
        print()

    # ── Section 1: NER entity counts ─────────────────────────────────────

    @classmethod
    def _print_ner_stats(cls, processed_dir: Path) -> None:
        """Print entity type → count table from spaCy DocBin."""
        spacy_path = processed_dir / "cuad_ner_train.spacy"

        print()
        print("  SECTION 1 — NER Entity Type Distribution")
        print("  " + "-" * 50)

        if not spacy_path.exists():
            print("  [!] cuad_ner_train.spacy not found — run cuad_to_ner.py first")
            return

        try:
            import spacy
            from spacy.tokens import DocBin

            try:
                nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
            except OSError:
                nlp = spacy.blank("en")

            doc_bin = DocBin().from_disk(spacy_path)
            docs    = list(doc_bin.get_docs(nlp.vocab))

            label_counts: dict[str, int] = defaultdict(int)
            total_ents   = 0
            for doc in docs:
                for ent in doc.ents:
                    label_counts[ent.label_] += 1
                    total_ents += 1

            print(f"  {'Label':<22}  {'Count':>8}  {'%':>6}")
            print("  " + "-" * 42)
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                pct = (count / total_ents * 100) if total_ents else 0
                print(f"  {label:<22}  {count:>8,}  {pct:>5.1f}%")
            print("  " + "-" * 42)
            print(f"  {'TOTAL':<22}  {total_ents:>8,}  {'100.0%':>6}")
            print(f"\n  Source: {spacy_path}  ({len(docs)} documents)")

        except ImportError:
            print("  [!] spaCy not installed — cannot read .spacy files")
            print("      Install: pip install spacy && python -m spacy download en_core_web_lg")

    # ── Section 2: Clause distribution ───────────────────────────────────

    @classmethod
    def _print_clause_stats(cls, processed_dir: Path) -> None:
        """Print clause type → positive/negative ratio table."""
        train_path = processed_dir / "cuad_clauses_train.json"
        dev_path   = processed_dir / "cuad_clauses_dev.json"

        print()
        print("  SECTION 2 — Clause Type Distribution (pos/neg ratio)")
        print("  " + "-" * 62)

        if not train_path.exists():
            print("  [!] cuad_clauses_train.json not found — run cuad_to_classification.py first")
            return

        clause_pos: dict[str, int] = defaultdict(int)
        clause_neg: dict[str, int] = defaultdict(int)

        for path in [train_path, dev_path]:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        ct     = record["clause_type"]
                        if record["label"] == 1:
                            clause_pos[ct] += 1
                        else:
                            clause_neg[ct] += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

        all_types = sorted(set(list(clause_pos.keys()) + list(clause_neg.keys())))

        print(f"  {'Clause Type':<38}  {'Pos':>5}  {'Neg':>5}  {'Rate':>6}")
        print("  " + "-" * 62)
        for ct in sorted(all_types, key=lambda c: -clause_pos.get(c, 0)):
            pos  = clause_pos.get(ct, 0)
            neg  = clause_neg.get(ct, 0)
            tot  = pos + neg
            rate = f"{pos/tot:.1%}" if tot > 0 else "  N/A"
            print(f"  {ct:<38}  {pos:>5,}  {neg:>5,}  {rate:>6}")

        total_pos = sum(clause_pos.values())
        total_neg = sum(clause_neg.values())
        total     = total_pos + total_neg
        print("  " + "-" * 62)
        print(f"  {'TOTAL':<38}  {total_pos:>5,}  {total_neg:>5,}  {total_pos/total:.1%}")

    # ── Section 3: Summary ────────────────────────────────────────────────

    @classmethod
    def _print_summary(cls, processed_dir: Path, cuad_json_path: Path) -> None:
        """Print summary statistics."""
        print()
        print("  SECTION 3 — Dataset Summary")
        print("  " + "-" * 50)

        # Count from classification files
        train_path = processed_dir / "cuad_clauses_train.json"
        dev_path   = processed_dir / "cuad_clauses_dev.json"

        total_samples    = 0
        context_lengths  = []
        contracts_seen   = set()

        for path in [train_path, dev_path]:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        total_samples += 1
                        context_lengths.append(len(r.get("text_span", "")))
                        contracts_seen.add(r.get("contract_name", ""))
                    except json.JSONDecodeError:
                        continue

        # Read span conflicts from NER stats or CUAD JSON
        conflicts_resolved = cls._count_conflicts_from_stats(processed_dir)

        avg_len = (
            sum(context_lengths) / len(context_lengths)
            if context_lengths else 0
        )

        print(f"  Total classification samples:  {total_samples:>10,}")
        print(f"  Unique contracts:              {len(contracts_seen):>10,}")
        print(f"  Avg text_span length (chars):  {avg_len:>10,.0f}")
        print(f"  Span conflicts resolved:       {conflicts_resolved:>10,}")

        # Check output files
        print()
        print("  Output files:")
        output_files = [
            ("cuad_ner_train.spacy",      "spaCy DocBin — NER train"),
            ("cuad_ner_dev.spacy",        "spaCy DocBin — NER dev"),
            ("cuad_clauses_train.json",   "JSON Lines   — Clause train"),
            ("cuad_clauses_dev.json",     "JSON Lines   — Clause dev"),
        ]
        for fname, desc in output_files:
            fpath = processed_dir / fname
            if fpath.exists():
                size_kb = fpath.stat().st_size / 1024
                print(f"    [OK] {fname:<32}  {size_kb:>8.0f} KB  ({desc})")
            else:
                print(f"    [!!] {fname:<32}  NOT FOUND  ({desc})")

    @staticmethod
    def _count_conflicts_from_stats(processed_dir: Path) -> int:
        """Try to read conflict count from dataset_statistics.json if available."""
        stats_path = processed_dir / "dataset_statistics.json"
        if stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("span_conflicts", 0)
            except Exception:
                pass
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    DatasetStats.print_report()
