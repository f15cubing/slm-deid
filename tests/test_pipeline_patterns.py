"""Deterministic pattern-PII detection. See pipeline/patterns.py."""

from __future__ import annotations

from pipeline import patterns


def _labels(text):
    return {(s.label, s.text) for s in patterns.detect(text)}


def test_detects_common_pattern_types_with_correct_offsets():
    text = "Reach me at sam@school.edu or (415) 555-0132 by Friday."
    spans = patterns.detect(text)
    by_label = {s.label: s for s in spans}
    assert "EMAIL" in by_label and by_label["EMAIL"].text == "sam@school.edu"
    assert "PHONE" in by_label and by_label["PHONE"].text == "(415) 555-0132"
    # offsets must index the original text
    for s in spans:
        assert text[s.start : s.end] == s.text


def test_ssn_credit_card_ip_url():
    assert ("SSN", "123-45-6789") in _labels("SSN 123-45-6789 on file")
    assert ("CREDIT_CARD", "4111 1111 1111 1111") in _labels("card 4111 1111 1111 1111 ok")
    assert ("IP", "192.168.0.1") in _labels("host 192.168.0.1 down")
    assert any(lbl == "URL" for lbl, _ in _labels("see https://example.com/x for more"))


def test_generic_id():
    assert ("ID", "STU00081234") in _labels("student STU00081234 enrolled")


def test_trailing_punctuation_trimmed():
    (span,) = [s for s in patterns.detect("mail me: sam@school.edu.") if s.label == "EMAIL"]
    assert span.text == "sam@school.edu"


def test_no_overlap_double_count():
    # The email host looks IP/URL-ish; only one span should survive over that region.
    text = "contact bob@10.0.0.5 now"
    spans = patterns.detect(text)
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            assert not (a.start < b.end and b.start < a.end)


def test_clean_text_has_no_patterns():
    assert patterns.detect("Chelsea helped me revise my thesis statement.") == []


def test_presidio_backend_falls_back_gracefully():
    # Whether or not presidio is importable, this must return spans (fallback = regex).
    spans = patterns.detect("email sam@school.edu", backend="presidio")
    assert any(s.label == "EMAIL" for s in spans)
