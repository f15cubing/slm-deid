"""Day 2 — data-gen orchestration (mock teacher, no network)."""

from src.common import tags
from src.datagen.generate import DatagenConfig, build_dataset, write_splits
from src.datagen.teacher import TeacherGenerator


def _mock_teacher():
    # Deterministic teacher: always returns the same well-formed tagged passage, and a verifier
    # that agrees. build_dataset should gate it in and split it.
    passage = f"{tags.wrap('Grace')} lent me her notes, but she showed grace under pressure."
    return TeacherGenerator(gen=lambda s, u: passage, verify=lambda s, u: passage)


def test_build_dataset_gates_and_splits():
    cfg = DatagenConfig(
        category_counts={"person_vs_common": 8},
        negatives=6,
        seed=0,
        val_frac=0.25,
    )
    train, val, drops = build_dataset(cfg, _mock_teacher())
    total = len(train) + len(val)
    assert total == 14  # 8 teacher + 6 negatives, all pass the gate + agree
    assert len(val) == int(total * 0.25)
    for ex in train + val:
        ex.validate()  # everything written is schema-valid


def test_build_dataset_drops_verifier_disagreement():
    passage = f"{tags.wrap('Grace')} lent me notes, but she showed grace under pressure."
    disagree = (
        f"{tags.wrap('Grace')} lent me notes, but she showed "
        f"{tags.wrap('grace')} under pressure."
    )
    teacher = TeacherGenerator(gen=lambda s, u: passage, verify=lambda s, u: disagree)
    cfg = DatagenConfig(category_counts={"person_vs_common": 5}, negatives=0, seed=1)
    train, val, drops = build_dataset(cfg, teacher)
    assert len(train) + len(val) == 0
    assert drops.get("verifier_disagreement") == 5


def test_write_splits(tmp_path):
    cfg = DatagenConfig(category_counts={"easy": 4}, negatives=0, seed=2, val_frac=0.5)
    train, val, _ = build_dataset(cfg, _mock_teacher())
    counts = write_splits(train, val, out_dir=str(tmp_path))
    assert (tmp_path / "splits" / "train.jsonl").exists()
    assert counts["train"] == len(train) and counts["val"] == len(val)
