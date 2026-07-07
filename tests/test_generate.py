"""Day 2 — data-gen orchestration (mock teacher, no network)."""

from src.common import tags
from src.common.schema import Example, Span, dumps
from src.datagen.generate import DatagenConfig, build_dataset, deleak_and_split, write_splits
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


def test_deleak_and_split_drops_eval_matches(tmp_path):
    # One example duplicates a quarantined eval input -> must be dropped.
    (tmp_path / "eval").mkdir()
    raw_eval = "Chelsea helped me revise my thesis."
    tgt_eval = f"{tags.wrap('Chelsea')} helped me revise my thesis."
    ev = Example(id="e", input=raw_eval, target=tgt_eval, spans=[Span(0, 7, "Chelsea", True)],
                 quarantine=True).validate()
    (tmp_path / "eval" / "hc.jsonl").write_text(dumps(ev) + "\n", encoding="utf-8")

    leak = Example(id="leak", input=raw_eval, target=tgt_eval,
                   spans=[Span(0, 7, "Chelsea", True)]).validate()
    clean = Example(id="ok", input="Ada coded.", target=f"{tags.wrap('Ada')} coded.",
                    spans=[Span(0, 3, "Ada", True)]).validate()

    train, val, n_leak = deleak_and_split(
        [leak, clean], eval_dir=str(tmp_path / "eval"), val_frac=0.0, seed=0
    )
    assert n_leak == 1
    ids = {e.id for e in train + val}
    assert ids == {"ok"}


def test_write_splits(tmp_path):
    cfg = DatagenConfig(category_counts={"easy": 4}, negatives=0, seed=2, val_frac=0.5)
    train, val, _ = build_dataset(cfg, _mock_teacher())
    counts = write_splits(train, val, out_dir=str(tmp_path))
    assert (tmp_path / "splits" / "train.jsonl").exists()
    assert counts["train"] == len(train) and counts["val"] == len(val)
