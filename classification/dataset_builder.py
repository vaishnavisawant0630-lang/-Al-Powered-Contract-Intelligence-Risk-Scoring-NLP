"""Build a HuggingFace Dataset with 41-dim multi-label vectors from Phase 1 JSON output."""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import torch
from datasets import Dataset

from .config import TrainingConfig

logger = logging.getLogger(__name__)


def _read_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_by_span(records: list[dict], labels: list[str]) -> list[dict]:
    """Group records sharing the same (contract_name, text_span) into one
    example with a 41-dim binary label vector."""
    label_to_idx = {name: i for i, name in enumerate(labels)}
    grouped: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * len(labels))

    for r in records:
        key = (r["contract_name"], r["text_span"])
        idx = label_to_idx.get(r["clause_type"])
        if idx is None:
            continue
        if r.get("label", 0):
            grouped[key][idx] = 1

    examples = []
    for (contract_name, text_span), vec in grouped.items():
        examples.append({
            "text": text_span,
            "contract_name": contract_name,
            "label_vector": vec,
        })
    return examples


def compute_pos_weight(examples: list[dict], num_labels: int) -> torch.Tensor:
    """pos_weight[i] = (N - P_i) / P_i, clipped to avoid div-by-zero for
    labels with zero positives in this split."""
    n = len(examples)
    pos_counts = [0] * num_labels
    for ex in examples:
        for i, v in enumerate(ex["label_vector"]):
            pos_counts[i] += v

    weights = []
    for p in pos_counts:
        if p == 0:
            weights.append(1.0)  # no positive signal to weight — neutral
        else:
            weights.append(max((n - p) / p, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def build_datasets(config: TrainingConfig, tokenizer) -> tuple[Dataset, Dataset, torch.Tensor]:
    labels = config.load_labels()
    assert len(labels) == config.num_labels, (
        f"clause_labels.json has {len(labels)} labels, "
        f"but config.num_labels={config.num_labels}"
    )

    train_records = _read_records(config.train_path)
    dev_records = _read_records(config.dev_path)

    train_examples = _group_by_span(train_records, labels)
    dev_examples = _group_by_span(dev_records, labels)

    logger.info(
        "Built %d train spans, %d dev spans (from %d / %d raw records)",
        len(train_examples), len(dev_examples), len(train_records), len(dev_records),
    )

    pos_weight = compute_pos_weight(train_examples, config.num_labels)

    train_ds = Dataset.from_list(train_examples)
    dev_ds = Dataset.from_list(dev_examples)

    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )
        enc["labels"] = [[float(v) for v in vec] for vec in batch["label_vector"]]
        return enc

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "contract_name", "label_vector"])
    dev_ds = dev_ds.map(tokenize_fn, batched=True, remove_columns=["text", "contract_name", "label_vector"])

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, dev_ds, pos_weight


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from transformers import AutoTokenizer

    cfg = TrainingConfig()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tr, dv, pw = build_datasets(cfg, tok)
    print(f"train={len(tr)} dev={len(dv)} pos_weight_shape={pw.shape}")
