"""Merge of dataset batches (+ CRAPII) into deduped, leakage-guarded splits."""

import json

from src.common import tags
from src.common.schema import Example, Span, write_jsonl
from src.datagen.merge import _dedup, merge


def _ex(id_: str, name: str) -> Example:
    inp = f"{name} explained the topic clearly after class."
    return Example(
        id=id_,
        input=inp,
        target=f"{tags.wrap(name)} explained the topic clearly after class.",
        spans=[Span(0, len(name), name, True)],
        category="person_vs_common",
        source="synthetic_teacher",
    ).validate()


def test_dedup_by_normalized_input():
    out = _dedup([_ex("a", "Joy"), _ex("b", "Joy"), _ex("c", "Iris")])
    assert len(out) == 2


def test_merge_combines_dedups_and_folds_crapii(tmp_path):
    s1 = tmp_path / "s1.jsonl"
    s2 = tmp_path / "s2.jsonl"
    write_jsonl(s1, [_ex("a", "Joy"), _ex("b", "Iris")])
    write_jsonl(s2, [_ex("c", "Joy"), _ex("d", "Pearl")])  # "Joy" duplicates s1

    rec = {
        "document": 1,
        "full_text": "Kwame wrote an essay.",
        "tokens": ["Kwame", "wrote", "an", "essay", "."],
        "trailing_whitespace": [True, True, True, False, False],
        "labels": ["B-NAME", "O", "O", "O", "O"],
    }
    crapii = tmp_path / "crapii.jsonl"
    crapii.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    train, val, stats = merge(
        [str(s1), str(s2)],
        crapii_path=str(crapii),
        crapii_limit=5,
        eval_dir=str(tmp_path / "noeval"),
        val_frac=0.0,
        seed=0,
    )
    got = train + val
    assert len({e.input for e in got}) == len(got)  # no duplicate inputs
    assert any(e.source == "real_crapii" for e in got)
    assert stats["dupes_dropped"] >= 1
    for e in got:
        e.validate()
