"""Offset-projection core — the integrity fix. See pipeline/project.py."""

from __future__ import annotations

import re

import pytest

from pipeline import project
from src.common import tags
from src.common.prompts import SHOWCASE

# Strip ANY ⟨X⟩ / ⟨/X⟩ marker (projection only emits NAME, but be generic).
_ANY = re.compile(r"⟨/?[A-Z_]+⟩")


def _oracle_tag(passage: str, names: list[str]) -> str:
    """Reference: tag the first occurrence of each name in a clean passage."""
    spans = []
    for n in names:
        i = passage.find(n)
        if i >= 0:
            spans.append(tags.TaggedSpan(i, i + len(n), n))
    return project.render_tagged(passage, spans)


def test_integrity_holds_by_construction_on_arbitrary_model_output():
    original = "Chelsea helped me revise my thesis."
    # A pathological "model output": drifted whitespace, a wrong char, a hallucinated tail.
    model = "⟨NAME⟩Chelsea⟨/NAME⟩  helped  me  revise my thsis. revise revise ⟨NAME⟩revise⟨/NAME⟩"
    out = project.project_tags(original, model)
    assert tags.unwrap(out) == original  # the whole point


@pytest.mark.parametrize("passage,note", SHOWCASE)
def test_integrity_on_showcase_passages(passage, note):
    # Even feeding the passage back verbatim (no tags), projection must be a lossless no-op.
    assert tags.unwrap(project.project_tags(passage, passage)) == passage


def test_whitespace_collapse_drift_projects_onto_original_offsets():
    original = "I met  Sarah   at the  library."  # irregular double spaces
    # Model normalizes the whitespace (drift) but correctly tags Sarah.
    model = "I met ⟨NAME⟩Sarah⟨/NAME⟩ at the library."
    result = project.project(original, model)
    assert result.dropped == 0
    assert [s.text for s in result.spans] == ["Sarah"]
    (span,) = result.spans
    assert original[span.start : span.end] == "Sarah"
    assert tags.unwrap(result.tagged) == original


def test_hallucinated_repetition_tail_is_dropped():
    original = "Thanks Sam."
    # Long-generation failure: model repeats and tags junk not present in the original.
    model = "Thanks ⟨NAME⟩Sam⟨/NAME⟩. ⟨NAME⟩Sam⟨/NAME⟩ ⟨NAME⟩Samuelll⟨/NAME⟩"
    result = project.project(original, model)
    assert tags.unwrap(result.tagged) == original
    # "Sam" maps (twice → merged/deduped to the single original occurrence); "Samuelll" drops.
    assert result.dropped >= 1
    assert all(original[s.start : s.end] == s.text for s in result.spans)


def test_clean_case_reproduces_spans():
    original = "Ada and Grace worked together."
    model = _oracle_tag(original, ["Ada", "Grace"])
    result = project.project(original, model)
    assert [(s.start, s.end) for s in result.spans] == [(0, 3), (8, 13)]
    assert result.dropped == 0


def test_overlapping_projected_spans_stay_well_formed():
    original = "Mary Jane spoke."
    # Two overlapping tags that both land on the same region.
    model = "⟨NAME⟩Mary Jane⟨/NAME⟩ ⟨NAME⟩Jane⟨/NAME⟩ spoke."
    out = project.project_tags(original, model)
    assert tags.is_well_formed(out)
    assert tags.unwrap(out) == original
