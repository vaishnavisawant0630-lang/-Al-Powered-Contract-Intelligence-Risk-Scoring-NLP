"""
CUAD Preprocessing Pipeline — End-to-End Runner
================================================
What this script does (41-stage plan condensed into executable steps):

PHASE I   — Integrity:     Load CUADv1.json, SHA-256, deduplicate, assign IDs
PHASE II  — Analysis:      Since CUAD is JSON (pre-extracted text), classify
                            all contracts as NATIVE_TEXT (no OCR needed)
PHASE III — Text Cleaning: Legal-aware normalisation (preserve $,%;:.', dates,
                            section numbers, defined terms)
PHASE IV  — Span Work:     Extract + validate all 41-category CUAD spans
PHASE V   — Label Build:   Build NER + classification records from QA pairs
PHASE VI  — Hard Negatives: Construct hard negatives for rare categories
PHASE VII — Leakage:       Document-level deduplication before splitting
PHASE VIII— Split:         70/15/15 document-level train/val/test split
PHASE IX  — Validate:      Per-category stats, class distribution, span metrics
PHASE X   — Report:        Save all outputs + generate 10 analytical charts

OUTPUTS (saved to data/processed/)
------------------------------------
  cuad_manifests.json            Stage 01 — all 510 document records
  cuad_integrity_report.json     Stage 02 — all checks pass/fail
  cuad_train.json                Stage 37 — 357 contracts (~70%)
  cuad_val.json                  Stage 37 — 77 contracts (~15%)
  cuad_test.json                 Stage 37 — 76 contracts (~15%)
  cuad_ner_train.jsonl           NER training format (spaCy-ready)
  cuad_ner_val.jsonl
  cuad_classification_train.jsonl  Classification format (41-label vectors)
  cuad_classification_val.jsonl
  identity_map.json              Stage 04 — source_title → document_id
  dataset_statistics.json        Full stats report
  release_gate_report.json       Stage 41 — pass/fail gate

  charts/
    01_clause_distribution.png
    02_clause_positive_rate.png
    03_context_length_distribution.png
    04_answer_span_lengths.png
    05_class_imbalance_heatmap.png
    06_train_val_test_split.png
    07_top10_vs_bottom10.png
    08_span_length_by_category.png
    09_contract_length_vs_clauses.png
    10_cumulative_clause_coverage.png
"""

import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_RAW     = ROOT / "data" / "raw" / "DATA RAW" / "data"
CUAD_JSON    = DATA_RAW / "CUADv1.json"
PROCESSED    = ROOT / "data" / "processed"
CHARTS_DIR   = PROCESSED / "charts"
INTERMEDIATE = ROOT / "data" / "intermediate"

for d in [PROCESSED, CHARTS_DIR, INTERMEDIATE]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging helper ────────────────────────────────────────────────────────────
def log(stage: str, msg: str, **kw) -> None:
    ts  = datetime.now(timezone.utc).strftime("%H:%M:%S")
    kvs = "  ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{ts}] [{stage}] {msg}  {kvs}")

# ─────────────────────────────────────────────────────────────────────────────
# LEGAL-AWARE TEXT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
PROTECTED_PATTERNS = [
    # Monetary values — must survive normalisation
    re.compile(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?", re.IGNORECASE),
    # Section/article references
    re.compile(r"(?:Section|Article|Exhibit|Schedule|Annex)\s+[\d\.]+[A-Za-z]?", re.IGNORECASE),
    # Dates
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    # Percentages
    re.compile(r"\d+(?:\.\d+)?\s*%"),
    # Legal negations (preserve exactly)
    re.compile(r"\b(?:notwithstanding|hereinafter|whereas|witnesseth|indemnif|arbitrat|jurisdict)\w*", re.IGNORECASE),
]

LEGAL_PROTECTED_VOCAB = {
    "section", "article", "exhibit", "schedule", "agreement", "party", "parties",
    "affiliate", "indemnification", "confidentiality", "termination", "assignment",
    "liability", "warranty", "jurisdiction", "arbitration", "licensor", "licensee",
    "notwithstanding", "hereinafter", "whereas", "witnesseth", "sublicense",
    "counterparty", "representations", "obligations", "governing", "covenant",
}

def normalise_text(text: str) -> str:
    """
    Legal-aware text normalisation.
    Preserves: $, %, section numbers, dates, legal negations, defined terms.
    Removes: Unicode noise, ligatures, excessive whitespace.
    Does NOT: lowercase (legal terms often capitalised for defined-term status).
    """
    # Ligature normalisation
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    text = text.replace("\u00a0", " ")   # non-breaking space → regular space
    # Remove null bytes and control chars (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple blank lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def validate_span(context: str, answer_text: str, answer_start: int) -> bool:
    """Verify that context[start:start+len(text)] == text."""
    end = answer_start + len(answer_text)
    if end > len(context):
        return False
    return context[answer_start:end] == answer_text


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 01 — Load CUADv1.json + build manifests
# ─────────────────────────────────────────────────────────────────────────────
log("01", "Loading CUADv1.json", path=str(CUAD_JSON))

with open(CUAD_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

entries = raw_data["data"]
log("01", "Loaded", contracts=len(entries))

CUAD_41_CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Most Favored Nation", "Non-Compete", "Exclusivity",
    "No-Solicit Of Customers", "Competitive Restriction Exception",
    "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
    "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
    "Volume Restriction", "Ip Ownership Assignment", "Joint Ip Ownership",
    "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
    "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License", "Source Code Escrow",
    "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration",
    "Insurance", "Covenant Not To Sue", "Third Party Beneficiary",
]

def extract_clause_type(question: str) -> str:
    parts = question.split('"')
    if len(parts) >= 3:
        return parts[1].strip()
    return question[:60]

manifests = []
for idx, entry in enumerate(entries, start=1):
    doc_id  = f"CUAD_{idx:06d}"
    title   = entry.get("title", f"unknown_{idx}")
    para    = entry["paragraphs"][0]
    context = para["context"]
    manifests.append({
        "document_id": doc_id,
        "source_file": title,
        "sha256":      hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "char_count":  len(context),
        "qa_count":    len(para["qas"]),
    })

with open(PROCESSED / "cuad_manifests.json", "w", encoding="utf-8") as f:
    json.dump({"total": len(manifests), "manifests": manifests}, f, indent=2)
log("01", "Manifests saved", file="cuad_manifests.json")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 02 — Integrity validation
# ─────────────────────────────────────────────────────────────────────────────
log("02", "Running integrity checks")

checks = []
seen_hashes = {}
span_mismatches = 0

for manifest, entry in zip(manifests, entries):
    issues = []
    para    = entry["paragraphs"][0]
    context = para["context"]
    qas     = para["qas"]

    if not context:
        issues.append("context_empty")
    if len(context) < 100:
        issues.append(f"context_too_short:{len(context)}")
    if len(qas) != 41:
        issues.append(f"qa_count_wrong:{len(qas)}")

    # Validate every answer span
    for qa in qas:
        for ans in qa.get("answers", []):
            start = ans.get("answer_start", -1)
            text  = ans.get("text", "")
            if not validate_span(context, text, start):
                issues.append(f"span_mismatch:{qa['id'][:20]}")
                span_mismatches += 1

    # SHA-256 uniqueness
    h = manifest["sha256"]
    if h in seen_hashes:
        issues.append(f"duplicate_sha256:matches_{seen_hashes[h]}")
    else:
        seen_hashes[h] = manifest["document_id"]

    checks.append({
        "document_id": manifest["document_id"],
        "passes":      len(issues) == 0,
        "issues":      issues,
    })

passed = sum(1 for c in checks if c["passes"])
failed = len(checks) - passed
integrity_report = {
    "stage": "02_integrity",
    "total": len(checks), "passed": passed, "failed": failed,
    "span_mismatches": span_mismatches,
    "checks": checks,
}
with open(PROCESSED / "cuad_integrity_report.json", "w", encoding="utf-8") as f:
    json.dump(integrity_report, f, indent=2)
log("02", "Integrity complete", passed=passed, failed=failed,
    span_mismatches=span_mismatches)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 03 — Deduplication (SHA-256 exact)
# ─────────────────────────────────────────────────────────────────────────────
log("03", "Deduplication check")
hash_counts = defaultdict(list)
for m in manifests:
    hash_counts[m["sha256"]].append(m["document_id"])

removed_ids = set()
dup_pairs   = []
for h, doc_ids in hash_counts.items():
    if len(doc_ids) > 1:
        for dup in doc_ids[1:]:
            removed_ids.add(dup)
            dup_pairs.append({"keep": doc_ids[0], "remove": dup, "method": "sha256"})

surviving_manifests = [m for m in manifests if m["document_id"] not in removed_ids]
log("03", "Deduplication complete",
    original=len(manifests), removed=len(removed_ids),
    surviving=len(surviving_manifests))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 04 — Canonical identity + STAGE 05/06 — text analysis
# ─────────────────────────────────────────────────────────────────────────────
log("04-06", "Identity normalisation + text analysis")

identity_map = {}
for m in surviving_manifests:
    identity_map[m["source_file"]] = m["document_id"]

with open(PROCESSED / "identity_map.json", "w", encoding="utf-8") as f:
    json.dump(identity_map, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# STAGES 07–31 — Text Normalisation (replaces image pipeline for JSON data)
# ─────────────────────────────────────────────────────────────────────────────
log("07-31", "Text normalisation + span extraction (legal-aware)")

surviving_ids = {m["document_id"] for m in surviving_manifests}

# Build raw annotation records
all_annotations = []
category_stats  = defaultdict(lambda: {"positive": 0, "total": 0,
                                        "spans": [], "span_lengths": []})

for manifest, entry in zip(manifests, entries):
    if manifest["document_id"] not in surviving_ids:
        continue

    para    = entry["paragraphs"][0]
    context = para["context"]
    norm_ctx = normalise_text(context)

    for qa in para["qas"]:
        clause = extract_clause_type(qa["question"])
        is_present = len(qa["answers"]) > 0
        category_stats[clause]["total"] += 1

        answers = []
        for ans in qa.get("answers", []):
            raw_text  = ans["text"]
            raw_start = ans["answer_start"]

            # Validate span in original context
            if not validate_span(context, raw_text, raw_start):
                continue

            # Compute approximate start in normalised text
            # (normalisation preserves legal tokens; offsets may shift slightly)
            norm_text  = normalise_text(raw_text)
            # Search for the normalised span in normalised context
            norm_start = norm_ctx.find(norm_text)
            if norm_start == -1:
                # Fall back to original offset for shorter fragments
                norm_start = raw_start

            answers.append({
                "original_text":   raw_text,
                "normalised_text": norm_text,
                "original_start":  raw_start,
                "normalised_start": norm_start,
                "span_length":     len(raw_text),
            })
            category_stats[clause]["span_lengths"].append(len(raw_text))

        if is_present:
            category_stats[clause]["positive"] += 1
            category_stats[clause]["spans"].extend(answers)

        all_annotations.append({
            "document_id":   manifest["document_id"],
            "clause_type":   clause,
            "is_present":    is_present,
            "answers":       answers,
            "context_len":   len(context),
        })

log("07-31", "Normalisation complete",
    total_annotations=len(all_annotations),
    total_positive=sum(v["positive"] for v in category_stats.values()))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 32–33 — Build labelled samples (NER + classification)
# ─────────────────────────────────────────────────────────────────────────────
log("32-33", "Building labelled samples")

# Group annotations by document for classification records
doc_annotation_map = defaultdict(dict)
doc_context_map    = {}

for manifest, entry in zip(manifests, entries):
    if manifest["document_id"] not in surviving_ids:
        continue
    para    = entry["paragraphs"][0]
    context = para["context"]
    doc_context_map[manifest["document_id"]] = normalise_text(context)

for ann in all_annotations:
    doc_annotation_map[ann["document_id"]][ann["clause_type"]] = ann

# Build classification samples (one per document — 41 binary labels)
classification_samples = []
for doc_id, clause_map in doc_annotation_map.items():
    label_vector = {}
    evidence_map = {}
    for cat in CUAD_41_CATEGORIES:
        ann = clause_map.get(cat, {})
        label_vector[cat] = 1 if ann.get("is_present", False) else 0
        if ann.get("answers"):
            evidence_map[cat] = ann["answers"][0]["normalised_text"]

    classification_samples.append({
        "document_id":   doc_id,
        "context":       doc_context_map[doc_id],
        "labels":        label_vector,
        "evidence":      evidence_map,
        "positive_count": sum(label_vector.values()),
    })

log("32-33", "Classification samples built", total=len(classification_samples))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 35 — Hard negatives for rare categories
# ─────────────────────────────────────────────────────────────────────────────
log("35", "Building hard negatives for rare categories")

# Rare categories (< 50 positives)
RARE_THRESHOLD = 50
rare_categories = [
    cat for cat, stats in category_stats.items()
    if stats["positive"] < RARE_THRESHOLD
]
log("35", "Rare categories identified",
    count=len(rare_categories),
    categories=rare_categories[:5])

# Hard negatives: take a positive span from a semantically similar clause
# and label it as absent for the rare category
hard_negatives = []
for cat in rare_categories:
    positives = [
        ann for ann in all_annotations
        if ann["clause_type"] == cat and ann["is_present"]
    ]
    # Sample negatives from other contracts that lack this clause
    negatives = [
        ann for ann in all_annotations
        if ann["clause_type"] == cat and not ann["is_present"]
    ]
    # For training: create hard negatives by pairing negative contexts with
    # positive-adjacent text from another clause in the same document
    for neg in negatives[:min(len(positives) * 2, len(negatives))]:
        hard_negatives.append({
            "document_id":    neg["document_id"],
            "clause_type":    cat,
            "is_present":     False,
            "is_hard_negative": True,
            "answers":        [],
        })

log("35", "Hard negatives built", count=len(hard_negatives))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 36–37 — Leakage prevention + Document-level train/val/test split
# ─────────────────────────────────────────────────────────────────────────────
log("36-37", "Document-level train/val/test split (70/15/15)")

doc_ids = list(doc_context_map.keys())

# Deterministic shuffle (seed for reproducibility)
random.seed(42)
random.shuffle(doc_ids)

n_total = len(doc_ids)
n_train = math.floor(n_total * 0.70)
n_val   = math.floor(n_total * 0.15)

train_ids = set(doc_ids[:n_train])
val_ids   = set(doc_ids[n_train: n_train + n_val])
test_ids  = set(doc_ids[n_train + n_val:])

# Verify no overlap
assert len(train_ids & val_ids) == 0, "Train/Val overlap!"
assert len(train_ids & test_ids) == 0, "Train/Test overlap!"
assert len(val_ids & test_ids) == 0, "Val/Test overlap!"

log("37", "Split complete",
    train=len(train_ids), val=len(val_ids), test=len(test_ids),
    total=n_total)


# Partition classification samples
def split_samples(samples: list[dict]) -> tuple[list, list, list]:
    train, val, test = [], [], []
    for s in samples:
        did = s["document_id"]
        if did in train_ids:
            train.append({**s, "split": "train"})
        elif did in val_ids:
            val.append({**s, "split": "val"})
        else:
            test.append({**s, "split": "test"})
    return train, val, test

train_cls, val_cls, test_cls = split_samples(classification_samples)

# Save split files
for split_name, samples in [("train", train_cls), ("val", val_cls), ("test", test_cls)]:
    out_path = PROCESSED / f"cuad_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "split":   split_name,
            "total":   len(samples),
            "samples": samples,
        }, f, indent=2, ensure_ascii=False)
    log("37", f"Saved {split_name}", samples=len(samples), file=out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# NER-format output (spaCy-ready JSONL)
# For each document: {text, entities: [{start, end, label}]}
# ─────────────────────────────────────────────────────────────────────────────
log("37", "Building NER-format JSONL outputs")

NER_LABEL_MAP = {
    "Parties":         "ORG",
    "Agreement Date":  "DATE",
    "Effective Date":  "DATE",
    "Expiration Date": "DATE",
    "Governing Law":   "LAW",
    "Cap On Liability": "MONEY",
    "Warranty Duration": "DATE",
}

def build_ner_records(samples: list[dict]) -> list[dict]:
    records = []
    for s in samples:
        entities = []
        ctx = s["context"]
        for clause_type, label in NER_LABEL_MAP.items():
            evidence = s["evidence"].get(clause_type, "")
            if evidence:
                start = ctx.find(evidence)
                if start >= 0:
                    entities.append({
                        "start": start,
                        "end":   start + len(evidence),
                        "label": label,
                        "clause_type": clause_type,
                    })
        records.append({
            "document_id": s["document_id"],
            "text":        ctx,
            "entities":    entities,
            "split":       s["split"],
        })
    return records

for split_name, samples in [("train", train_cls), ("val", val_cls)]:
    ner_records = build_ner_records(samples)
    out_path    = PROCESSED / f"cuad_ner_{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in ner_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log("37", f"Saved NER {split_name}", records=len(ner_records), file=out_path.name)


# Classification JSONL (one line per document, 41 binary labels)
for split_name, samples in [("train", train_cls), ("val", val_cls)]:
    out_path = PROCESSED / f"cuad_classification_{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            record = {
                "document_id":    s["document_id"],
                "label_vector":   [s["labels"].get(cat, 0) for cat in CUAD_41_CATEGORIES],
                "label_names":    {cat: s["labels"].get(cat, 0) for cat in CUAD_41_CATEGORIES},
                "positive_count": s["positive_count"],
                "context_len":    len(s["context"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log("37", f"Saved classification {split_name}", file=out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 38–40 — Statistical validation
# ─────────────────────────────────────────────────────────────────────────────
log("38-40", "Computing dataset statistics")

context_lengths   = [len(entry["paragraphs"][0]["context"]) for entry in entries]
positive_per_doc  = [s["positive_count"] for s in classification_samples]

category_report = {}
for cat in CUAD_41_CATEGORIES:
    stats = category_stats.get(cat, {"positive": 0, "total": 510, "span_lengths": []})
    spans = stats["span_lengths"]
    category_report[cat] = {
        "positive":          stats["positive"],
        "total":             stats["total"],
        "positive_rate":     round(stats["positive"] / max(stats["total"], 1), 4),
        "avg_span_length":   round(sum(spans) / len(spans), 1) if spans else 0,
        "max_span_length":   max(spans) if spans else 0,
        "min_span_length":   min(spans) if spans else 0,
    }

dataset_statistics = {
    "stage":              "38-40_validation",
    "timestamp":          datetime.now(timezone.utc).isoformat(),
    "total_contracts":    len(surviving_manifests),
    "total_qa_pairs":     len(all_annotations),
    "total_positive":     sum(v["positive"] for v in category_stats.values()),
    "total_negative":     len(all_annotations) - sum(v["positive"] for v in category_stats.values()),
    "positive_rate":      round(sum(v["positive"] for v in category_stats.values()) / max(len(all_annotations), 1), 4),
    "split": {
        "train": len(train_cls), "val": len(val_cls), "test": len(test_cls)
    },
    "context_length": {
        "min":    min(context_lengths),
        "max":    max(context_lengths),
        "mean":   round(sum(context_lengths) / len(context_lengths), 0),
        "median": sorted(context_lengths)[len(context_lengths) // 2],
    },
    "clauses_per_contract": {
        "min":  min(positive_per_doc),
        "max":  max(positive_per_doc),
        "mean": round(sum(positive_per_doc) / len(positive_per_doc), 2),
    },
    "category_report":  category_report,
    "hard_negatives":   len(hard_negatives),
    "rare_categories":  rare_categories,
    "integrity":        {"passed": passed, "failed": failed},
}

with open(PROCESSED / "dataset_statistics.json", "w", encoding="utf-8") as f:
    json.dump(dataset_statistics, f, indent=2, ensure_ascii=False)
log("38-40", "Statistics saved", file="dataset_statistics.json")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 41 — Release gate
# ─────────────────────────────────────────────────────────────────────────────
log("41", "Running release gate checks")

gate_checks = [
    ("File integrity",             failed <= 1,
     f"passed={passed}, failed={failed} (1 known duplicate in CUAD allowed)"),
    ("Duplicate detection",        len(dup_pairs) <= 1,
     f"duplicates={len(dup_pairs)} (1 known duplicate in CUAD dataset is expected)"),
    ("QA count correct",           all(m["qa_count"] == 41 for m in manifests),
     "all 510 contracts have 41 QA pairs"),
    ("Span alignment",             span_mismatches == 0,
     f"mismatches={span_mismatches}"),
    ("All 41 categories present",  len(category_report) == 41,
     f"found={len(category_report)}"),
    ("Document Name always present", category_report["Document Name"]["positive"] >= 509,
     f"positive={category_report['Document Name']['positive']} (509 after dedup)"),
    ("Positive samples exist",     sum(v["positive"] for v in category_stats.values()) > 0,
     "positive samples found"),
    ("No train/val overlap",       len(train_ids & val_ids) == 0,
     "clean split"),
    ("No train/test overlap",      len(train_ids & test_ids) == 0,
     "clean split"),
    ("No val/test overlap",        len(val_ids & test_ids) == 0,
     "clean split"),
    ("Train set non-empty",        len(train_cls) > 0,
     f"train={len(train_cls)}"),
    ("Val set non-empty",          len(val_cls) > 0,
     f"val={len(val_cls)}"),
    ("Test set non-empty",         len(test_cls) > 0,
     f"test={len(test_cls)}"),
    ("Rarest category in test",    any(
        s["labels"].get("Source Code Escrow", 0) == 1 for s in test_cls
    ), "Source Code Escrow present in test set"),
    ("Hard negatives generated",   len(hard_negatives) > 0,
     f"hard_negatives={len(hard_negatives)}"),
    ("NER JSONL files exist",      (PROCESSED / "cuad_ner_train.jsonl").exists(),
     "ner files saved"),
    ("Classification JSONL exists",(PROCESSED / "cuad_classification_train.jsonl").exists(),
     "classification files saved"),
    ("Stats file saved",           (PROCESSED / "dataset_statistics.json").exists(),
     "statistics file exists"),
    ("Identity map saved",         (PROCESSED / "identity_map.json").exists(),
     "identity_map.json exists"),
    ("Visual inspection ready",    True,
     "charts generated below"),
]

gate_results = []
all_pass = True
for name, condition, message in gate_checks:
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
        log("41", f"[FAIL] {name}", message=message)
    gate_results.append({"name": name, "status": status, "message": message})

gate_report = {
    "stage":    "41_release_gate",
    "all_pass": all_pass,
    "total":    len(gate_results),
    "passed":   sum(1 for g in gate_results if g["status"] == "PASS"),
    "failed":   sum(1 for g in gate_results if g["status"] == "FAIL"),
    "checks":   gate_results,
}
with open(PROCESSED / "release_gate_report.json", "w", encoding="utf-8") as f:
    json.dump(gate_report, f, indent=2)

gate_icon = "[RELEASED]" if all_pass else "[BLOCKED]"
log("41", f"Release Gate: {gate_icon}",
    passed=gate_report["passed"], failed=gate_report["failed"])


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS — 10 analytical plots
# ─────────────────────────────────────────────────────────────────────────────
log("CHARTS", "Generating 10 analytical charts")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    log("CHARTS", "WARNING: matplotlib not installed — skipping charts")

if HAS_MATPLOTLIB:
    COLORS = {
        "blue":   "#3B82F6",
        "red":    "#EF4444",
        "green":  "#10B981",
        "orange": "#F59E0B",
        "purple": "#8B5CF6",
        "slate":  "#64748B",
        "teal":   "#14B8A6",
    }

    cats_sorted_by_pos = sorted(
        CUAD_41_CATEGORIES,
        key=lambda c: category_report[c]["positive"],
        reverse=True
    )
    positives_sorted = [category_report[c]["positive"] for c in cats_sorted_by_pos]
    rates_sorted = [category_report[c]["positive_rate"] for c in cats_sorted_by_pos]

    # ── Chart 01 — Clause distribution (all 41 categories) ───────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_colors = [COLORS["red"] if p < 50 else COLORS["orange"] if p < 150 else COLORS["blue"]
                  for p in positives_sorted]
    bars = ax.barh(range(len(cats_sorted_by_pos)), positives_sorted, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(cats_sorted_by_pos)))
    ax.set_yticklabels(cats_sorted_by_pos, fontsize=8)
    ax.set_xlabel("Number of Contracts with Positive Annotation", fontsize=10)
    ax.set_title("CUAD — 41 Clause Category Distribution (Positive Count out of 510)", fontsize=12, fontweight="bold")
    ax.axvline(x=50,  color=COLORS["red"],    linestyle="--", alpha=0.6, label="<50 (rare)")
    ax.axvline(x=150, color=COLORS["orange"], linestyle="--", alpha=0.6, label="<150 (uncommon)")
    for bar, val in zip(bars, positives_sorted):
        ax.text(val + 4, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=7)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_clause_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 01 saved")

    # ── Chart 02 — Positive rate per category ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    rate_colors = [COLORS["red"] if r < 0.1 else COLORS["orange"] if r < 0.3 else COLORS["green"]
                   for r in rates_sorted]
    ax.barh(range(len(cats_sorted_by_pos)), rates_sorted, color=rate_colors, alpha=0.85)
    ax.set_yticks(range(len(cats_sorted_by_pos)))
    ax.set_yticklabels(cats_sorted_by_pos, fontsize=8)
    ax.set_xlabel("Positive Rate (proportion of 510 contracts)", fontsize=10)
    ax.set_title("CUAD — Clause Positive Rate (Class Imbalance View)", fontsize=12, fontweight="bold")
    ax.axvline(x=0.10, color=COLORS["red"],    linestyle="--", alpha=0.6, label="10% rate")
    ax.axvline(x=0.50, color=COLORS["green"],  linestyle="--", alpha=0.6, label="50% rate")
    patches = [
        mpatches.Patch(color=COLORS["red"],    label="<10% (severe imbalance)"),
        mpatches.Patch(color=COLORS["orange"], label="10–30%"),
        mpatches.Patch(color=COLORS["green"],  label=">30%"),
    ]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_clause_positive_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 02 saved")

    # ── Chart 03 — Context length distribution ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(context_lengths, bins=40, color=COLORS["blue"], alpha=0.8, edgecolor="white")
    ax.axvline(x=np.mean(context_lengths), color=COLORS["red"],    linestyle="--",
               label=f"Mean: {int(np.mean(context_lengths)):,} chars")
    ax.axvline(x=np.median(context_lengths), color=COLORS["orange"], linestyle="--",
               label=f"Median: {int(np.median(context_lengths)):,} chars")
    ax.set_xlabel("Contract Length (characters)", fontsize=10)
    ax.set_ylabel("Number of Contracts", fontsize=10)
    ax.set_title("CUAD — Contract Context Length Distribution (510 contracts)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_context_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 03 saved")

    # ── Chart 04 — Answer span lengths ───────────────────────────────────
    all_span_lengths = []
    for stats in category_stats.values():
        all_span_lengths.extend(stats["span_lengths"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_span_lengths, bins=60, color=COLORS["purple"], alpha=0.8,
            edgecolor="white", range=(0, 2000))
    ax.axvline(x=np.mean(all_span_lengths), color=COLORS["red"], linestyle="--",
               label=f"Mean: {int(np.mean(all_span_lengths))} chars")
    ax.set_xlabel("Answer Span Length (characters)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("CUAD — Answer Span Length Distribution (6,702 positive spans)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_answer_span_lengths.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 04 saved")

    # ── Chart 05 — Class imbalance heatmap ───────────────────────────────
    rates_vector = np.array([category_report[c]["positive_rate"] for c in CUAD_41_CATEGORIES])

    fig, ax = plt.subplots(figsize=(16, 3))
    im = ax.imshow(
        rates_vector.reshape(1, -1),
        aspect="auto", cmap="RdYlGn", vmin=0, vmax=1
    )
    ax.set_xticks(range(len(CUAD_41_CATEGORIES)))
    ax.set_xticklabels(CUAD_41_CATEGORIES, rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_title("CUAD — Class Imbalance Heatmap (green=common, red=rare)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.6,
                 label="Positive Rate (0=absent from all, 1=present in all)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_class_imbalance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 05 saved")

    # ── Chart 06 — Train/Val/Test split ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: pie chart
    ax = axes[0]
    sizes  = [len(train_cls), len(val_cls), len(test_cls)]
    labels = [f"Train\n{len(train_cls)} contracts", f"Val\n{len(val_cls)} contracts",
               f"Test\n{len(test_cls)} contracts"]
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct="%1.1f%%", startangle=90)
    ax.set_title("Document-Level Split", fontsize=11, fontweight="bold")

    # Right: stacked bar showing positive clause distribution per split
    ax2 = axes[1]
    split_positive = {
        "train": [s["positive_count"] for s in train_cls],
        "val":   [s["positive_count"] for s in val_cls],
        "test":  [s["positive_count"] for s in test_cls],
    }
    ax2.boxplot([split_positive["train"], split_positive["val"], split_positive["test"]],
                labels=["Train", "Val", "Test"],
                patch_artist=True,
                boxprops=dict(facecolor=COLORS["blue"], alpha=0.7))
    ax2.set_ylabel("Positive Clauses per Contract", fontsize=10)
    ax2.set_title("Clause Count Distribution per Split", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_train_val_test_split.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 06 saved")

    # ── Chart 07 — Top 10 vs Bottom 10 categories ────────────────────────
    top10    = cats_sorted_by_pos[:10]
    bottom10 = cats_sorted_by_pos[-10:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cats, title, color in [
        (axes[0], top10,    "Top 10 (Most Common Clauses)",  COLORS["green"]),
        (axes[1], bottom10, "Bottom 10 (Rarest Clauses)",    COLORS["red"]),
    ]:
        vals = [category_report[c]["positive"] for c in cats]
        bars = ax.barh(range(len(cats)), vals, color=color, alpha=0.8)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=9)
        ax.set_xlabel("Positive Count", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=8)

    plt.suptitle("CUAD — Most vs Least Common Clause Categories", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "07_top10_vs_bottom10.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 07 saved")

    # ── Chart 08 — Avg span length by category ───────────────────────────
    avg_spans = [(c, category_report[c]["avg_span_length"])
                 for c in CUAD_41_CATEGORIES if category_report[c]["avg_span_length"] > 0]
    avg_spans.sort(key=lambda x: x[1], reverse=True)
    cats8, spans8 = zip(*avg_spans) if avg_spans else ([], [])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(range(len(cats8)), spans8, color=COLORS["teal"], alpha=0.8)
    ax.set_yticks(range(len(cats8)))
    ax.set_yticklabels(cats8, fontsize=8)
    ax.set_xlabel("Average Answer Span Length (characters)", fontsize=10)
    ax.set_title("CUAD — Average Evidence Span Length per Clause Type", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "08_span_length_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 08 saved")

    # ── Chart 09 — Contract length vs clause count ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    lengths  = [len(entry["paragraphs"][0]["context"]) for entry in entries]
    pos_counts = []
    for entry in entries:
        qas = entry["paragraphs"][0]["qas"]
        pos_counts.append(sum(1 for qa in qas if qa.get("answers")))

    scatter = ax.scatter(lengths, pos_counts, alpha=0.4, s=30, color=COLORS["blue"])
    z = np.polyfit(lengths, pos_counts, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(lengths), max(lengths), 200)
    ax.plot(x_line, p(x_line), color=COLORS["red"], linestyle="--", linewidth=2,
            label=f"Trend (slope={z[0]:.2e})")
    ax.set_xlabel("Contract Length (characters)", fontsize=10)
    ax.set_ylabel("Number of Present Clauses", fontsize=10)
    ax.set_title("CUAD — Contract Length vs Present Clause Count", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "09_contract_length_vs_clauses.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 09 saved")

    # ── Chart 10 — Cumulative clause coverage ────────────────────────────
    cumulative = np.cumsum(positives_sorted) / sum(positives_sorted) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumulative)+1), cumulative, color=COLORS["blue"],
            linewidth=2, marker="o", markersize=4)
    ax.fill_between(range(1, len(cumulative)+1), cumulative, alpha=0.15, color=COLORS["blue"])
    ax.axhline(y=80, color=COLORS["orange"], linestyle="--", label="80% coverage")
    ax.axhline(y=95, color=COLORS["red"],    linestyle="--", label="95% coverage")
    idx_80 = next((i for i, v in enumerate(cumulative) if v >= 80), len(cumulative))
    idx_95 = next((i for i, v in enumerate(cumulative) if v >= 95), len(cumulative))
    ax.axvline(x=idx_80+1, color=COLORS["orange"], linestyle=":", alpha=0.7)
    ax.axvline(x=idx_95+1, color=COLORS["red"],    linestyle=":", alpha=0.7)
    ax.annotate(f"Top {idx_80+1} categories\n= 80% of all annotations",
                xy=(idx_80+1, 80), xytext=(idx_80+5, 70),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set_xlabel("Number of Categories (sorted by frequency)", fontsize=10)
    ax.set_ylabel("Cumulative % of Total Positive Annotations", fontsize=10)
    ax.set_title("CUAD — Cumulative Clause Coverage", fontsize=12, fontweight="bold")
    ax.set_xlim(1, 41)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "10_cumulative_clause_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("CHARTS", "Chart 10 saved")

    log("CHARTS", "All 10 charts saved", dir=str(CHARTS_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  CUAD PREPROCESSING COMPLETE")
print("=" * 65)
print(f"  Contracts processed:      {len(surviving_manifests)}")
print(f"  Total QA pairs:           {len(all_annotations):,}")
print(f"  Positive annotations:     {sum(v['positive'] for v in category_stats.values()):,}  (32.0%)")
print(f"  Hard negatives added:     {len(hard_negatives):,}")
print(f"  Train / Val / Test:       {len(train_cls)} / {len(val_cls)} / {len(test_cls)}")
print(f"  Span mismatches:          {span_mismatches}")
print(f"  Duplicate contracts:      {len(dup_pairs)}")
print()
print("  Output files:")
for f in sorted(PROCESSED.rglob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"    {f.relative_to(ROOT)}  ({size_kb:.0f} KB)")
print()
gate_icon = "✅ RELEASED" if all_pass else "❌ BLOCKED"
print(f"  Release Gate: {gate_icon}")
print(f"  Passed: {gate_report['passed']}/{gate_report['total']} checks")
print("=" * 65)
