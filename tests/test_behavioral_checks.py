"""Day 2, spec S2.2 — deterministic behavioral checks.

Required coverage: at least one leakage case, one over-tag case, one integrity-violation case.
"""

from src.common import tags
from src.common.schema import Example, Span
from src.eval import behavioral_checks as bc


def _ex(input_text: str, target: str, spans: list[Span]) -> Example:
    return Example(id="t", input=input_text, target=target, spans=spans).validate()


# "Chelsea helped me, but I visited Chelsea in London." — first is a person, second a place.
RAW = "Chelsea helped me, but I visited Chelsea in London."
GOLD_TARGET = f"{tags.wrap('Chelsea')} helped me, but I visited Chelsea in London."
SPANS = [Span(0, 7, "Chelsea", True), Span(33, 40, "Chelsea", False)]


def test_perfect_output_passes():
    ex = _ex(RAW, GOLD_TARGET, SPANS)
    r = bc.check(ex, GOLD_TARGET)
    assert r.passed
    assert (r.tp, r.fp, r.fn) == (1, 0, 0)
    assert not r.leaked and not r.over_tagged


def test_leakage_when_name_left_untagged():
    ex = _ex(RAW, GOLD_TARGET, SPANS)
    out = RAW  # nothing tagged -> the person "Chelsea" is leaked
    r = bc.check(ex, out)
    assert r.integrity_ok  # text unchanged, just under-tagged
    assert r.leaked and not r.over_tagged
    assert (r.tp, r.fp, r.fn) == (0, 0, 1)
    assert (0, 7, "Chelsea") in bc.leakage(ex, out)


def test_over_tag_when_place_tagged():
    ex = _ex(RAW, GOLD_TARGET, SPANS)
    # Tag BOTH Chelseas -> the place (second) is a false positive.
    out = f"{tags.wrap('Chelsea')} helped me, but I visited {tags.wrap('Chelsea')} in London."
    r = bc.check(ex, out)
    assert r.integrity_ok
    assert r.over_tagged and not r.leaked
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)
    assert (33, 40, "Chelsea") in bc.over_tag(ex, out)


def test_integrity_violation_is_automatic_fail():
    ex = _ex(RAW, GOLD_TARGET, SPANS)
    # Model dropped trailing text (like the Day-1 base did) -> integrity fail.
    out = f"{tags.wrap('Chelsea')} helped me"
    r = bc.check(ex, out)
    assert not r.integrity_ok
    assert not r.passed
    # All gold counts as missed on an integrity failure.
    assert r.fn == 1


def test_malformed_tags_fail_and_dont_crash():
    ex = _ex(RAW, GOLD_TARGET, SPANS)
    out = f"{tags.NAME_OPEN}Chelsea helped me"  # unbalanced
    r = bc.check(ex, out)
    assert not r.well_formed
    assert not r.passed


def test_example_with_no_names_passes_when_untagged():
    raw = "We applied the Newton method to approximate the root."
    ex = _ex(raw, raw, [Span(15, 21, "Newton", False)])
    r = bc.check(ex, raw)
    assert r.passed
    assert (r.tp, r.fp, r.fn) == (0, 0, 0)


def test_example_with_no_names_over_tags_if_eponym_tagged():
    raw = "We applied the Newton method to approximate the root."
    ex = _ex(raw, raw, [Span(15, 21, "Newton", False)])
    out = f"We applied the {tags.wrap('Newton')} method to approximate the root."
    r = bc.check(ex, out)
    assert r.over_tagged
    assert (15, 21, "Newton") in bc.over_tag(ex, out)
