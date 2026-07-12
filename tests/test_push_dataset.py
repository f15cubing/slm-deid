"""Guard tests for scripts/push_dataset.py — the hard-ceiling publication check.

The publish path must refuse to upload if a split (a) carries a quarantine=true row, or (b) shares
any input with the quarantined eval sets. These tests prove both refusals fire, and that the real
in-repo splits pass the guard.
"""

import importlib.util
from pathlib import Path

import pytest

from src.common.schema import Example, write_jsonl

REPO = Path(__file__).resolve().parents[1]

# Load the script module by path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location("push_dataset", REPO / "scripts" / "push_dataset.py")
push_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(push_dataset)


def _ex(id_: str, input_: str, quarantine: bool = False) -> Example:
    # target = input with no tags is byte-identical → schema-valid (unwrap == input).
    return Example(
        id=id_,
        input=input_,
        target=input_,
        register="exposition",
        category="person_vs_common",
        spans=[],
        source="synthetic_teacher",
        quarantine=quarantine,
    )


def test_real_splits_pass_the_guard():
    files = sorted((REPO / "data" / "splits").glob("*.jsonl"))
    assert files, "expected data/splits/*.jsonl to exist"
    rows, _ = push_dataset.check_splits(files)
    assert rows > 0


def test_quarantine_row_is_refused(tmp_path):
    f = tmp_path / "train.jsonl"
    write_jsonl(
        f,
        [
            _ex("ok-1", "A clean training sentence with no overlap zzqq."),
            _ex("bad-1", "A quarantined row zzqq.", quarantine=True),
        ],
    )
    with pytest.raises(SystemExit, match="quarantine=true"):
        push_dataset.check_splits([f])


def test_eval_overlap_is_refused(tmp_path):
    # Pull a real eval input and drop it into a "training" split → must be caught.
    eval_files = sorted((REPO / "eval").rglob("*.jsonl"))
    from src.common.schema import read_jsonl

    leaked_input = read_jsonl(eval_files[0])[0].input
    f = tmp_path / "train.jsonl"
    write_jsonl(f, [_ex("leak-1", leaked_input)])
    with pytest.raises(SystemExit, match="EVAL LEAKAGE"):
        push_dataset.check_splits([f])


def test_empty_dir_is_refused():
    with pytest.raises(SystemExit, match="nothing to publish"):
        push_dataset.check_splits([])
