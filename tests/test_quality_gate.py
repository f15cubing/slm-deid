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


# --- Day-4 TASK 3: category-semantics gate (labels must be trustworthy) ------------------
def _sem_ex(raw, target, spans, category, token=None, id_="s"):
    return Example(id=id_, input=raw, target=target, spans=spans,
                   source="synthetic_teacher", category=category, ambiguous_token=token)


def test_negative_trap_that_tags_a_name_is_dropped():
    # The exact Day-3 mislabel: a negative_trap row that tags real names (Freud/Jung/Skinner).
    raw = "Skinner shaped the whole field of study."
    tgt = f"{tags.wrap('Skinner')} shaped the whole field of study."
    ex = _sem_ex(raw, tgt, [Span(0, 7, "Skinner", True)], "negative_trap")
    r = qg.gate(ex)
    assert not r.ok and r.reason == "negative_trap_has_name"


def test_negative_trap_with_zero_names_passes():
    raw = "Introduction to Statistics met on Tuesdays in the annex."
    ex = _sem_ex(raw, raw, [], "negative_trap")
    assert qg.gate(ex).ok


def test_person_category_missing_intended_token_dropped():
    # Labeled person_vs_place around token "Austin", but the passage is about a different name.
    raw = "Sydney helped me revise the essay after class."
    tgt = f"{tags.wrap('Sydney')} helped me revise the essay after class."
    ex = _sem_ex(raw, tgt, [Span(0, 6, "Sydney", True)], "person_vs_place", token="Austin")
    r = qg.gate(ex)
    assert not r.ok and r.reason == "missing_ambiguous_token"


def test_person_category_with_intended_token_present_passes():
    raw = "Austin helped me revise the essay after class."
    tgt = f"{tags.wrap('Austin')} helped me revise the essay after class."
    ex = _sem_ex(raw, tgt, [Span(0, 6, "Austin", True)], "person_vs_place", token="Austin")
    assert qg.gate(ex).ok


def test_person_category_without_token_field_is_not_judged_on_token():
    # Legacy/teacher rows that don't carry an intended token are not dropped for token-presence.
    raw = "Sydney helped me revise the essay after class."
    tgt = f"{tags.wrap('Sydney')} helped me revise the essay after class."
    ex = _sem_ex(raw, tgt, [Span(0, 6, "Sydney", True)], "person_vs_place", token=None)
    assert qg.gate(ex).ok


def test_possessive_without_apostrophe_s_dropped():
    raw = "Austin revised the whole essay overnight."
    tgt = f"{tags.wrap('Austin')} revised the whole essay overnight."
    ex = _sem_ex(raw, tgt, [Span(0, 6, "Austin", True)], "possessive", token="Austin")
    r = qg.gate(ex)
    assert not r.ok and r.reason == "possessive_not_possessive"


def test_possessive_with_apostrophe_s_passes():
    raw = "Austin's essay improved after peer review."
    tgt = f"{tags.wrap('Austin')}'s essay improved after peer review."
    ex = _sem_ex(raw, tgt, [Span(0, 6, "Austin", True)], "possessive", token="Austin")
    assert qg.gate(ex).ok


def test_eponymous_possessive_negative_passes():
    # "Joule's law" — a possessive that is NOT a person; zero name spans, still a valid possessive.
    raw = "Joule's law relates power to current and resistance."
    ex = _sem_ex(raw, raw, [], "possessive", token="Joule")
    assert qg.gate(ex).ok


def test_filter_counts_semantics_drops_by_reason():
    good = _ex(GOOD_RAW, GOOD_TGT, GOOD_SPANS, id_="ok")
    nt = _sem_ex("Skinner led the class.", f"{tags.wrap('Skinner')} led the class.",
                 [Span(0, 7, "Skinner", True)], "negative_trap", id_="nt")
    kept, drops = qg.filter_examples([good, nt])
    assert [e.id for e in kept] == ["ok"]
    assert drops == {"negative_trap_has_name": 1}
