"""Day 2, spec S2.9 — Faker pattern-type negatives."""

from src.common import tags
from src.datagen.negatives import generate_negatives
from src.datagen.quality_gate import gate


def test_generates_requested_count_and_all_valid():
    exs = generate_negatives(n=15, seed=1)
    assert len(exs) == 15
    for ex in exs:
        assert gate(ex).ok  # every generated negative passes the quality gate


def test_deterministic_with_seed():
    a = generate_negatives(n=10, seed=7)
    b = generate_negatives(n=10, seed=7)
    assert [e.input for e in a] == [e.input for e in b]


def test_pure_negatives_tag_nothing():
    exs = generate_negatives(n=30, seed=2)
    pure = [e for e in exs if not e.name_spans()]
    assert pure, "expected some pure pattern-only negatives"
    for ex in pure:
        assert ex.target == ex.input  # nothing tagged
        assert tags.tagged_spans(ex.target) == []


def test_mixed_examples_tag_only_the_name():
    exs = generate_negatives(n=30, seed=3)
    mixed = [e for e in exs if e.name_spans()]
    assert mixed, "expected some mixed name+pii examples"
    for ex in mixed:
        # exactly the name is tagged; the email/phone/id/url stays untagged
        assert len(tags.tagged_spans(ex.target)) == 1
        assert tags.unwrap(ex.target) == ex.input
