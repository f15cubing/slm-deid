"""Offline tests for the API-benchmark generator (scripts/build_api_bench.py).

The live generation is exercised by running the script with a key; here we lock the pure bits: the
recipe shape and the quarantine stamping (so a generated benchmark can never be mistaken for
training data).
"""

from scripts.build_api_bench import build_config, quarantine
from src.common.schema import Example


def test_config_is_balanced_and_single_split():
    cfg = build_config(scale=1.0, seed=7)
    # val_frac=0 → one split; quarantined benchmark, not a train/val pair.
    assert cfg.val_frac == 0.0
    # Person-vs-* pairs dominate (the hard discrimination cases).
    assert set(cfg.minimal_pairs) == {
        "person_vs_common",
        "person_vs_place",
        "possessive",
        "person_vs_eponym",
    }
    assert set(cfg.category_counts) == {"negative_trap", "first_name_only", "third_party"}
    assert cfg.eval_dir == "eval"


def test_quarantine_stamps_every_row():
    rows = [
        Example(id="a", input="x", target="x", category="easy", source="synthetic_teacher"),
        Example(id="b", input="y", target="y", category="easy", source="synthetic_teacher"),
    ]
    assert all(not r.quarantine for r in rows)  # default off
    out = quarantine(rows)
    assert all(r.quarantine for r in out)
    # Does not mutate the inputs (dataclasses.replace returns new objects).
    assert all(not r.quarantine for r in rows)
