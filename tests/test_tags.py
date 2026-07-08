"""Day 1, spec S1.2 — the tag syntax contract.

These are pure-Python and run anywhere (no GPU, no model). They lock the behavior of
``src/common/tags.py`` so the rest of the pipeline can rely on it.
"""

from src.common import tags
from src.common.tags import NAME_CLOSE, NAME_OPEN


def test_wrap_uses_locked_markers():
    assert tags.wrap("Sarah") == f"{NAME_OPEN}Sarah{NAME_CLOSE}"


def test_unwrap_inverts_wrap():
    for x in ["Sarah", "", "two words", "O'Brien-Núñez", "⟨not a real tag⟩"]:
        assert tags.unwrap(tags.wrap(x)) == x


def test_unwrap_strips_only_markers_never_other_text():
    tagged = f"Thanks {NAME_OPEN}Sam{NAME_CLOSE}, said {NAME_OPEN}Dr. Rao{NAME_CLOSE}."
    assert tags.unwrap(tagged) == "Thanks Sam, said Dr. Rao."


def test_integrity_invariant_holds_for_multi_span():
    raw = "Newton met Chelsea in Florence."
    tagged = f"{tags.wrap('Newton')} met {tags.wrap('Chelsea')} in Florence."
    assert tags.unwrap(tagged) == raw


def test_ascii_angle_brackets_are_not_markers():
    # A student typing <NAME> or code like a<b must survive untouched.
    text = "if a < b and List<Name> then print(<NAME>)"
    assert tags.unwrap(text) == text
    assert tags.tagged_spans(text) == []


class TestWellFormed:
    def test_balanced_ok(self):
        assert tags.is_well_formed(f"a {NAME_OPEN}x{NAME_CLOSE} b")

    def test_empty_span_is_structurally_ok(self):
        assert tags.is_well_formed(f"{NAME_OPEN}{NAME_CLOSE}")

    def test_unclosed_open_rejected(self):
        assert not tags.is_well_formed(f"a {NAME_OPEN}x b")

    def test_close_before_open_rejected(self):
        assert not tags.is_well_formed(f"a {NAME_CLOSE}x{NAME_OPEN} b")

    def test_nesting_rejected(self):
        assert not tags.is_well_formed(f"{NAME_OPEN}a{NAME_OPEN}b{NAME_CLOSE}{NAME_CLOSE}")


class TestTaggedSpans:
    def test_single_span_offsets_index_into_unwrapped(self):
        tagged = f"Hi {NAME_OPEN}Sam{NAME_CLOSE}!"
        raw = tags.unwrap(tagged)
        spans = tags.tagged_spans(tagged)
        assert len(spans) == 1
        s = spans[0]
        assert (s.start, s.end, s.text) == (3, 6, "Sam")
        assert raw[s.start : s.end] == "Sam"

    def test_multiple_spans(self):
        tagged = f"{NAME_OPEN}Ada{NAME_CLOSE} and {NAME_OPEN}Alan{NAME_CLOSE} coded."
        raw = tags.unwrap(tagged)
        spans = tags.tagged_spans(tagged)
        assert [s.text for s in spans] == ["Ada", "Alan"]
        for s in spans:
            assert raw[s.start : s.end] == s.text

    def test_no_spans_when_untagged(self):
        assert tags.tagged_spans("just plain text, no names tagged") == []
