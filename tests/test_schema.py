"""Day 2 foundation — schema + integrity validation."""

import pytest

from src.common import tags
from src.common.schema import Example, SchemaError, Span, dumps, loads


def make_valid() -> Example:
    raw = "Chelsea helped me, but I visited Chelsea in London."
    target = f"{tags.wrap('Chelsea')} helped me, but I visited Chelsea in London."
    return Example(
        id="ex-1",
        input=raw,
        target=target,
        register="dialogue",
        category="person_vs_place",
        spans=[
            Span(0, 7, "Chelsea", True),    # the person
            Span(33, 40, "Chelsea", False),  # the place
        ],
        source="handbuilt",
        paraphrase_group="pg-1",
        quarantine=True,
    )


def test_valid_example_passes():
    make_valid().validate()


def test_roundtrip_jsonl():
    ex = make_valid()
    again = loads(dumps(ex))
    assert again == ex
    again.validate()


def test_integrity_violation_rejected():
    ex = make_valid()
    ex.target = ex.target + " EXTRA"  # unwrap(target) != input
    with pytest.raises(SchemaError, match="integrity"):
        ex.validate()


def test_malformed_target_rejected():
    ex = make_valid()
    ex.target = ex.target.replace(tags.NAME_CLOSE, "", 1)  # unbalanced
    with pytest.raises(SchemaError):
        ex.validate()


def test_span_text_mismatch_rejected():
    ex = make_valid()
    ex.spans = [Span(0, 7, "ChelseX", True), Span(33, 40, "Chelsea", False)]
    with pytest.raises(SchemaError, match="span text"):
        ex.validate()


def test_tagged_must_equal_name_spans():
    # target tags "Chelsea" (person) but gold marks NO name -> mismatch.
    ex = make_valid()
    ex.spans = [Span(0, 7, "Chelsea", False), Span(33, 40, "Chelsea", False)]
    with pytest.raises(SchemaError, match="!= gold name spans"):
        ex.validate()


def test_bad_enum_values_rejected():
    ex = make_valid()
    ex.category = "not_a_category"
    with pytest.raises(SchemaError, match="category"):
        ex.validate()


def test_untagged_example_with_no_names():
    raw = "We applied the Newton method to approximate the root."
    ex = Example(
        id="ex-2",
        input=raw,
        target=raw,  # nothing tagged
        register="essay",
        category="person_vs_eponym",
        spans=[Span(15, 21, "Newton", False)],
        source="handbuilt",
    )
    ex.validate()
    assert ex.name_spans() == []
