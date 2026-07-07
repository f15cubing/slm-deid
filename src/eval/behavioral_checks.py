"""Deterministic behavioral checks (Day 2, spec S2.2).

Pure functions, no model calls. Given the raw ``input``, the gold name spans, and a model's
``output`` string (tagged), decide PASS/FAIL and locate the errors:

- **integrity**  — ``unwrap(output) == input`` (byte-for-byte). If this fails the item is an
  automatic FAIL: the model altered text it was told to leave untouched.
- **leakage**    — gold ``is_name`` spans the model failed to tag (false negatives).
- **over_tag**   — spans the model tagged that are not gold name spans (false positives).

Spans are compared on a common coordinate system: gold offsets index into ``input``; predicted
offsets index into ``unwrap(output)``. When integrity holds these are the same string, so
``(start, end, text)`` tuples are directly comparable. When integrity fails, the coordinates are
not trustworthy, so every gold span counts as missed and every predicted span as a false tag —
which is the honest outcome for an item that mangled the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import tags
from src.common.schema import Example, Span

SpanKey = tuple[int, int, str]


def _gold_keys(example: Example) -> set[SpanKey]:
    return {(s.start, s.end, s.text) for s in example.name_spans()}


def _predicted_keys(output: str) -> set[SpanKey]:
    return {(t.start, t.end, t.text) for t in tags.tagged_spans(output)}


@dataclass
class CheckResult:
    """Per-item behavioral-check outcome."""

    integrity_ok: bool
    well_formed: bool
    missed: set[SpanKey] = field(default_factory=set)      # leakage (false negatives)
    false_tags: set[SpanKey] = field(default_factory=set)  # over-tag (false positives)
    matched: set[SpanKey] = field(default_factory=set)     # true positives

    @property
    def leaked(self) -> bool:
        return len(self.missed) > 0

    @property
    def over_tagged(self) -> bool:
        return len(self.false_tags) > 0

    @property
    def passed(self) -> bool:
        """PASS = well-formed, integrity holds, no leakage, no over-tagging."""
        return self.well_formed and self.integrity_ok and not self.leaked and not self.over_tagged

    # TP/FP/FN counts for aggregate entity-level metrics.
    @property
    def tp(self) -> int:
        return len(self.matched)

    @property
    def fp(self) -> int:
        return len(self.false_tags)

    @property
    def fn(self) -> int:
        return len(self.missed)


def integrity(input_text: str, output: str) -> bool:
    """True iff the model changed nothing but the tags."""
    return tags.unwrap(output) == input_text


def check(example: Example, output: str) -> CheckResult:
    """Run all behavioral checks for one model ``output`` against an ``example``."""
    well_formed = tags.is_well_formed(output)
    integrity_ok = well_formed and integrity(example.input, output)

    gold = _gold_keys(example)

    if not integrity_ok:
        # Coordinates are untrustworthy; count all gold as missed, all predicted as false.
        predicted = _predicted_keys(output) if well_formed else set()
        return CheckResult(
            integrity_ok=False,
            well_formed=well_formed,
            missed=set(gold),
            false_tags=set(predicted),
            matched=set(),
        )

    predicted = _predicted_keys(output)
    matched = gold & predicted
    return CheckResult(
        integrity_ok=True,
        well_formed=True,
        missed=gold - predicted,
        false_tags=predicted - gold,
        matched=matched,
    )


def leakage(example: Example, output: str) -> set[SpanKey]:
    """Gold name spans the model failed to tag."""
    return check(example, output).missed


def over_tag(example: Example, output: str) -> set[SpanKey]:
    """Spans the model tagged that are not gold name spans."""
    return check(example, output).false_tags


__all__ = [
    "CheckResult",
    "Span",
    "check",
    "integrity",
    "leakage",
    "over_tag",
]
