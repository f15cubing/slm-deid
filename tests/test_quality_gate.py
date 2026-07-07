"""Day 2, spec S2.10 — the data-gen quality gate."""

from src.common import tags
from src.common.schema import Example, Span
from src.datagen import quality_gate as qg


def _ex(raw, target, spans, id_="g"):
    return Example(id=id_, input=raw, target=target, spans=spans,
                   source="synthetic_teacher", category="person_vs_place")


GOOD_RAW = "Chelsea helped me, but I visited Chelsea in London."
GOOD_TGT = f"{tags.wrap('Chelsea')} helped me, but I visited Chelsea in London."
GOOD_SPANS = [Span(0, 7, "Chelsea", True), Span(33, 40, "Chelsea", False)]


def test_good_example_passes():
    assert qg.gate(_ex(GOOD_RAW, GOOD_TGT, GOOD_SPANS)).ok


def test_integrity_failure_rejected():
    bad = _ex(GOOD_RAW, GOOD_TGT + " oops", GOOD_SPANS)
    r = qg.gate(bad)
    assert not r.ok and r.reason == "integrity"


def test_malformed_tags_rejected():
    bad = _ex(GOOD_RAW, f"{tags.NAME_OPEN}Chelsea helped me...", [])
    r = qg.gate(bad)
    assert not r.ok and r.reason == "malformed_tags"


def test_schema_mismatch_rejected():
    # target tags Chelsea-person, but gold marks it not-a-name -> schema mismatch.
    bad = _ex(GOOD_RAW, GOOD_TGT, [Span(0, 7, "Chelsea", False), Span(33, 40, "Chelsea", False)])
    r = qg.gate(bad)
    assert not r.ok and r.reason.startswith("schema")


def test_verifier_disagreement_dropped():
    ex = _ex(GOOD_RAW, GOOD_TGT, GOOD_SPANS)
    # Verifier tagged the SECOND Chelsea (the place) instead -> disagreement.
    verifier = f"{tags.wrap('Chelsea')} helped me, but I visited {tags.wrap('Chelsea')} in London."
    r = qg.gate(ex, verifier_target=verifier)
    assert not r.ok and r.reason == "verifier_disagreement"


def test_verifier_altered_text_dropped():
    ex = _ex(GOOD_RAW, GOOD_TGT, GOOD_SPANS)
    verifier = f"{tags.wrap('Chelsea')} helped me."  # different underlying text
    r = qg.gate(ex, verifier_target=verifier)
    assert not r.ok and r.reason == "verifier_altered_text"


def test_filter_counts_drops_by_reason():
    good = _ex(GOOD_RAW, GOOD_TGT, GOOD_SPANS, id_="ok")
    bad_integrity = _ex(GOOD_RAW, GOOD_TGT + "x", GOOD_SPANS, id_="bad")
    kept, drops = qg.filter_examples([good, bad_integrity])
    assert [e.id for e in kept] == ["ok"]
    assert drops == {"integrity": 1}
