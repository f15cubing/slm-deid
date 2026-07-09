"""Projection-based integrity metric on the real eval path.

Demonstrates the ``pipeline/`` fix as a *number* the held-out eval reports: when the model's
name judgment is correct but its regenerated passage drifts (whitespace collapse, hallucinated
repetition tail), strict byte-identity scoring zeroes out recall and flags every item as an
integrity violation. Wrapping the tagger in :class:`pipeline.project.ProjectingTagger` restores
integrity (holds by construction) and recovers the true recall — same ``evaluate`` path, no
model, no GPU.
"""

from __future__ import annotations

import re

from pipeline.project import ProjectingTagger
from src.common import tags
from src.common.schema import Example, Span
from src.eval.run import evaluate
from src.infer import FunctionTagger

_NAMES = {"Sarah", "Marcus"}


def _example(ex_id: str, text: str, name: str) -> Example:
    """Build a valid Example tagging the first occurrence of ``name`` in ``text``."""
    i = text.index(name)
    target = text[:i] + tags.wrap(name) + text[i + len(name) :]
    spans = [Span(i, i + len(name), name, True)]
    return Example(
        id=ex_id,
        input=text,
        target=target,
        register="essay",
        category="real",
        source="real_crapii",
        spans=spans,
    ).validate()


def _drifting_tagger() -> FunctionTagger:
    """Correct judgment, wrong copy: collapses double-spaces (the real integrity failure)."""

    def fn(passage: str) -> str:
        collapsed = re.sub(r" {2,}", " ", passage)  # whitespace drift
        for name in _NAMES:
            collapsed = collapsed.replace(name, tags.wrap(name))
        return collapsed

    return FunctionTagger(fn, name="drift")


# Messy real-ish text: irregular double spaces the model will "helpfully" normalize.
EXAMPLES = [
    _example("d1", "I met  Sarah   at the  library yesterday.", "Sarah"),
    _example("d2", "Later  Marcus  helped   me revise the  essay.", "Marcus"),
]


def test_strict_scoring_fails_on_drift():
    report = evaluate(_drifting_tagger(), EXAMPLES, label="strict")
    # Every item mangled the text → integrity violated everywhere, recall collapses to 0.
    assert report.overall.integrity_violation_rate == 1.0
    assert report.overall.recall == 0.0
    assert report.overall.pass_rate == 0.0


def test_projection_restores_integrity_and_recovers_recall():
    report = evaluate(ProjectingTagger(_drifting_tagger()), EXAMPLES, label="proj")
    assert report.overall.integrity_violation_rate == 0.0  # holds by construction
    assert report.overall.recall == 1.0  # judgment recovered
    assert report.overall.pass_rate == 1.0


def test_projection_survives_repetition_tail():
    # Long-generation failure: model repeats a tagged tail not present in the original.
    ex = _example("r1", "Thanks  Sarah  for the notes.", "Sarah")

    def fn(passage: str) -> str:
        base = re.sub(r" {2,}", " ", passage).replace("Sarah", tags.wrap("Sarah"))
        return base + " " + tags.wrap("Sarahhh") + " " + tags.wrap("Sarahhh")

    report = evaluate(ProjectingTagger(FunctionTagger(fn)), [ex], label="proj-tail")
    assert report.overall.integrity_violation_rate == 0.0
    assert report.overall.recall == 1.0  # the real "Sarah" still tagged; junk tail dropped


def test_projecting_tagger_names_itself():
    assert ProjectingTagger(FunctionTagger(lambda p: p, name="tuned")).name == "tuned+proj"
