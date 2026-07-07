"""Day 2, spec S2.7 — the mechanical eval-leakage guard (a hard ceiling).

Fails if any quarantined eval input also appears in a training split. Also asserts the eval set
itself is schema-valid and flagged quarantine=true. This is the automated enforcement of the
AGENTS.md ceiling "never leak the eval set into data generation or training".
"""

import re
from pathlib import Path

import pytest

from src.common.schema import read_jsonl

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
TRAIN_GLOBS = ["data/splits/*.jsonl", "data/generated/*.jsonl"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _eval_files() -> list[Path]:
    return sorted(EVAL_DIR.rglob("*.jsonl"))


def test_eval_set_exists_and_is_valid_and_quarantined():
    files = _eval_files()
    assert files, "no eval/**/*.jsonl found — build it with scripts/build_hardcases.py"
    total = 0
    for f in files:
        for ex in read_jsonl(f):
            ex.validate()
            assert ex.quarantine, f"{f.name}:{ex.id} must have quarantine=true"
            assert ex.source == "handbuilt" or ex.source.startswith("real_"), (
                f"{f.name}:{ex.id} eval items must be handbuilt or real, not synthetic"
            )
            total += 1
    assert total > 0


def test_no_eval_input_appears_in_training_splits():
    eval_inputs = {
        _norm(ex.input) for f in _eval_files() for ex in read_jsonl(f)
    }
    train_files = [p for g in TRAIN_GLOBS for p in REPO.glob(g)]
    if not train_files:
        pytest.skip("no training splits yet — guard is a no-op until Day 3 data exists")

    offenders = []
    for f in train_files:
        for ex in read_jsonl(f):
            if _norm(ex.input) in eval_inputs:
                offenders.append(f"{f.name}:{ex.id}")
    assert not offenders, f"EVAL LEAKAGE — these training rows duplicate eval inputs: {offenders}"
