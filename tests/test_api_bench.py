"""Guard tests for the API benchmark (benchmarks/api_bench/) — enforce the no-leak hard ceiling.

api_bench lives OUTSIDE eval/ (it intentionally reuses in-vocab surfaces, which would violate the
eval-vocab-disjointness invariant in test_vocab.py, and its source is synthetic_teacher, not the
handbuilt/real that eval items require). So it is not covered by test_no_eval_leakage.py — these
tests give it the same protection: quarantined, schema-valid, and disjoint from both the training
splits and the quarantined eval sets.
"""

import re
from pathlib import Path

import pytest

from src.common.schema import read_jsonl

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "api_bench" / "api_bench.jsonl"
TRAIN_GLOBS = ["data/splits/*.jsonl", "data/generated/*.jsonl"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


@pytest.mark.skipif(not BENCH.exists(), reason="api_bench not generated in this checkout")
def test_api_bench_is_valid_and_quarantined():
    rows = read_jsonl(BENCH)
    assert rows, "api_bench is empty"
    for ex in rows:
        ex.validate()
        assert ex.quarantine, f"{ex.id}: api_bench rows must be quarantine=true"


@pytest.mark.skipif(not BENCH.exists(), reason="api_bench not generated in this checkout")
def test_api_bench_does_not_leak_into_training():
    bench = {_norm(ex.input) for ex in read_jsonl(BENCH)}
    train_files = [p for g in TRAIN_GLOBS for p in REPO.glob(g)]
    if not train_files:
        pytest.skip("no training splits in this checkout")
    offenders = [ex.id for f in train_files for ex in read_jsonl(f) if _norm(ex.input) in bench]
    assert not offenders, f"BENCHMARK LEAKAGE — training dupes api_bench inputs: {offenders}"


@pytest.mark.skipif(not BENCH.exists(), reason="api_bench not generated in this checkout")
def test_api_bench_is_disjoint_from_quarantined_eval():
    bench = {_norm(ex.input) for ex in read_jsonl(BENCH)}
    eval_inputs = {
        _norm(ex.input) for f in (REPO / "eval").rglob("*.jsonl") for ex in read_jsonl(f)
    }
    overlap = bench & eval_inputs
    assert not overlap, f"api_bench overlaps the eval sets ({len(overlap)} rows)"
