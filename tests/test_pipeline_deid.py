"""End-to-end pipeline orchestration. See pipeline/deid.py."""

from __future__ import annotations

import re

import pytest

from pipeline.deid import Deidentifier
from pipeline.stub import HeuristicNameStub
from src.infer import FunctionTagger

_ANY_MARKER = re.compile(r"⟨/?[A-Z_]+⟩")


def _strip(text):
    return _ANY_MARKER.sub("", text)


def test_tag_mode_is_integrity_preserving():
    text = "Marcus emailed sam@school.edu about the Newton method."
    # Tagger tags Marcus (a real name); patterns catch the email.
    tagger = FunctionTagger(lambda p: p.replace("Marcus", "⟨NAME⟩Marcus⟨/NAME⟩"))
    result = Deidentifier(tagger).deidentify(text, mode="tag")
    assert _strip(result.text) == text  # nothing but markers added
    labels = {e.label for e in result.entities}
    assert "NAME" in labels and "EMAIL" in labels


def test_mask_mode_replaces_with_labels():
    text = "Marcus emailed sam@school.edu."
    tagger = FunctionTagger(lambda p: p.replace("Marcus", "⟨NAME⟩Marcus⟨/NAME⟩"))
    out = Deidentifier(tagger).deidentify(text, mode="mask").text
    assert out == "[NAME] emailed [EMAIL]."


def test_surrogate_mode_removes_real_pii():
    text = "Marcus emailed sam@school.edu."
    tagger = FunctionTagger(lambda p: p.replace("Marcus", "⟨NAME⟩Marcus⟨/NAME⟩"))
    out = Deidentifier(tagger, surrogate_seed=1).deidentify(text, mode="surrogate").text
    assert "Marcus" not in out
    assert "sam@school.edu" not in out
    assert "@" in out  # replaced by a fake email, still readable


def test_pattern_wins_on_overlap_with_name():
    # If the tagger erroneously tags text that is actually a phone number, the pattern wins.
    text = "call 415-555-0132 today"
    tagger = FunctionTagger(lambda p: p.replace("415", "⟨NAME⟩415⟨/NAME⟩"))
    result = Deidentifier(tagger).deidentify(text, mode="mask")
    assert result.text == "call [PHONE] today"
    assert [e.label for e in result.entities] == ["PHONE"]


def test_drift_in_tagger_output_does_not_break_integrity():
    # Stub regenerates the passage; force whitespace drift to exercise projection end-to-end.
    text = "Marcus  met   Ada."

    def drifting(p):
        return HeuristicNameStub().tag(re.sub(r"\s+", " ", p))

    result = Deidentifier(FunctionTagger(drifting)).deidentify(text, mode="tag")
    assert _strip(result.text) == text


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        Deidentifier(FunctionTagger(lambda p: p)).deidentify("x", mode="nope")
