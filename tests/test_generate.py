"""Day 2 — data-gen orchestration (mock teacher, no network).

Day 4 additions: token-level eval-leakage guard + minimal-pair sub-recipe wiring.
"""

import re

from src.common import tags
from src.common.schema import Example, Span, dumps
from src.datagen.generate import (
    DatagenConfig,
    build_dataset,
    deleak_and_split,
    drop_eval_token_overlap,
    write_splits,
)
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


def test_token_leak_guard_drops_eval_token_overlap(tmp_path):
    # TASK 4: an example whose ambiguous_token is an eval token ("Chelsea") must be dropped even
    # when its PASSAGE text matches no eval passage (the passage-level de-leak would miss it).
    (tmp_path / "eval").mkdir()
    ev = Example(
        id="e", input="Chelsea helped me revise my thesis.",
        target=f"{tags.wrap('Chelsea')} helped me revise my thesis.",
        spans=[Span(0, 7, "Chelsea", True)], quarantine=True,
    ).validate()
    (tmp_path / "eval" / "hc.jsonl").write_text(dumps(ev) + "\n", encoding="utf-8")

    leak = Example(
        id="leak", input="Chelsea explained recursion to the study group tonight.",
        target=f"{tags.wrap('Chelsea')} explained recursion to the study group tonight.",
        spans=[Span(0, 7, "Chelsea", True)], category="person_vs_place",
        source="synthetic_teacher", ambiguous_token="Chelsea",
    ).validate()
    clean = Example(
        id="ok", input="Austin explained recursion to the study group tonight.",
        target=f"{tags.wrap('Austin')} explained recursion to the study group tonight.",
        spans=[Span(0, 6, "Austin", True)], category="person_vs_place",
        source="synthetic_teacher", ambiguous_token="Austin",
    ).validate()

    kept, dropped = drop_eval_token_overlap([leak, clean], eval_dir=str(tmp_path / "eval"))
    assert dropped == 1
    assert [e.id for e in kept] == ["ok"]


def _pair_mock_teacher():
    """Mock teacher that answers minimal-pair prompts by reading the token/sense/category."""

    def gen(system, user):
        token = re.search(r'"([^"]+)"', user).group(1)
        sense = "person" if "SENSE=person" in user else "nonperson"
        category = re.search(r"Category: (\w+)\.", user).group(1)
        if category == "possessive":
            if sense == "person":
                return f"{tags.wrap(token)}'s essay improved after peer review."
            return f"{token}'s law explains the observed result."
        if sense == "person":
            return f"{tags.wrap(token)} explained the topic clearly after class."
        return f"We discussed {token} together during the seminar."

    return TeacherGenerator(gen=gen, verify=None)


def test_build_dataset_minimal_pairs(tmp_path):
    # TASK 5: minimal_pairs config produces 2 examples per pair; drop counts include token-leak.
    cfg = DatagenConfig(
        minimal_pairs={"person_vs_place": 3, "possessive": 2},
        category_counts={},
        negatives=0,
        seed=0,
        val_frac=0.0,
        eval_dir=str(tmp_path / "eval"),  # no eval dir -> no leak drops
    )
    train, val, drops = build_dataset(cfg, _pair_mock_teacher())
    got = train + val
    assert len(got) == (3 + 2) * 2  # each pair -> person + non-person
    assert drops.get("eval_token_leak") == 0
    for ex in got:
        ex.validate()
    # every minimal-pair example carries its ambiguous token and passes semantics
    assert all(ex.ambiguous_token for ex in got)
    persons = [ex for ex in got if ex.name_spans()]
    nonpersons = [ex for ex in got if not ex.name_spans()]
    assert len(persons) == 5 and len(nonpersons) == 5  # one tagged + one untagged per pair


def test_build_dataset_scale_knob_multiplies_counts(tmp_path):
    cfg = DatagenConfig(
        minimal_pairs={"person_vs_place": 2},
        negatives=0,
        seed=0,
        val_frac=0.0,
        scale=2.0,
        eval_dir=str(tmp_path / "eval"),
    )
    train, val, _ = build_dataset(cfg, _pair_mock_teacher())
    assert len(train) + len(val) == 2 * 2 * 2  # 2 pairs * scale 2 * 2 examples


def test_write_splits(tmp_path):
    cfg = DatagenConfig(category_counts={"easy": 4}, negatives=0, seed=2, val_frac=0.5)
    train, val, _ = build_dataset(cfg, _mock_teacher())
    counts = write_splits(train, val, out_dir=str(tmp_path))
    assert (tmp_path / "splits" / "train.jsonl").exists()
    assert counts["train"] == len(train) and counts["val"] == len(val)
