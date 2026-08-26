import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from classification.config import TrainingConfig  # noqa: E402
from classification.dataset_builder import build_datasets, compute_pos_weight  # noqa: E402


def test_pos_weight_shape_and_values():
    examples = [
        {"label_vector": [1, 0, 0]},
        {"label_vector": [1, 0, 0]},
        {"label_vector": [0, 1, 0]},
        {"label_vector": [0, 0, 0]},
    ]
    pw = compute_pos_weight(examples, num_labels=3)
    assert pw.shape == (3,)
    assert pw[0] == pytest.approx((4 - 2) / 2)  # label 0: 2 positives out of 4
    assert pw[2] == 1.0  # no positives — neutral fallback, not div-by-zero


@pytest.mark.integration
def test_build_datasets_shapes():
    cfg = TrainingConfig(model_name="roberta-base")  # small/fast for CI
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, dev_ds, pos_weight = build_datasets(cfg, tokenizer)

    assert pos_weight.shape == (cfg.num_labels,)
    ex = train_ds[0]
    assert ex["input_ids"].shape[0] == cfg.max_length
    assert ex["labels"].shape[0] == cfg.num_labels
    assert not torch.isnan(ex["labels"]).any()
