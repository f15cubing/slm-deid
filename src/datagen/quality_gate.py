"""Quality gate for generated examples (Day 2, spec S2.10).

Every generated item must pass before it can enter the dataset. The gate rejects on:

- **tag well-formedness** — balanced, non-nested ``⟨NAME⟩`` tags in ``target``.
- **integrity** — ``unwrap(target) == input`` (the model/teacher changed nothing but tags).
- **schema validity** — offsets align, tagged spans == gold name spans, enums valid.
- **category semantics** (Day-4) — the label must match the content: ``negative_trap`` tags no
  name, an ambiguous person-vs-* row actually contains its intended token, a ``possessive`` row is
  actually possessive. This makes the category labels trustworthy for targeted data iteration.
- **teacher disagreement** (optional) — if a second-pass verifier retagged the passage and
  disagrees with the first pass, drop the item (we only keep high-confidence labels).

Pure functions; no network. The teacher/verifier calls happen in ``teacher.py`` and pass their
results in here as strings to compare.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.common import tags
from src.common.schema import Example, SchemaError

# Categories whose label asserts a specific ambiguous surface is present in the passage.
_PERSON_AMBIGUOUS = {"person_vs_place", "person_vs_common", "person_vs_eponym"}


@dataclass
class GateResult:
    ok: bool
    reason: str = ""


def check_wellformed(target: str) -> GateResult:
    if not tags.is_well_formed(target):
        return GateResult(False, "malformed_tags")
    return GateResult(True)


def check_integrity(input_text: str, target: str) -> GateResult:
    if tags.unwrap(target) != input_text:
        return GateResult(False, "integrity")
    return GateResult(True)


def check_schema(example: Example) -> GateResult:
    try:
        example.validate()
    except SchemaError as e:
        return GateResult(False, f"schema:{e}")
    return GateResult(True)


def check_category_semantics(example: Example) -> GateResult:
    """Make the category LABEL trustworthy (Day-4 TASK 3; root problem 2).

    Day-3 data had ``negative_trap`` rows that actually tagged real names (Freud/Jung/Skinner), so
    labels could not be trusted. This gate enforces the meaning of each label:

    - ``negative_trap`` → ZERO name spans (a "negative" that tags a name is mislabeled).
    - ``person_vs_place`` / ``person_vs_common`` / ``person_vs_eponym`` → the intended ambiguous
      token (``example.ambiguous_token``, when set) must actually appear in the passage.
    - ``possessive`` → the passage must contain a possessive ("'s") form.
    """
    cat = example.category
    if cat == "negative_trap":
        if example.name_spans():
            return GateResult(False, "negative_trap_has_name")
        return GateResult(True)
    if cat == "possessive":
        if "'s" not in example.input and "\u2019s" not in example.input:
            return GateResult(False, "possessive_not_possessive")
        return GateResult(True)
    if cat in _PERSON_AMBIGUOUS:
        token = example.ambiguous_token
        if token and not re.search(
            rf"\b{re.escape(token)}\b", example.input, flags=re.IGNORECASE
        ):
            return GateResult(False, "missing_ambiguous_token")
        return GateResult(True)
    return GateResult(True)


def check_teacher_agreement(target: str, verifier_target: str | None) -> GateResult:
    """If a verifier retagged the same passage, its tagging must match (drop on disagreement)."""
    if verifier_target is None:
        return GateResult(True)  # no second pass -> nothing to disagree with
    if tags.unwrap(target) != tags.unwrap(verifier_target):
        return GateResult(False, "verifier_altered_text")
    a = {(s.start, s.end, s.text) for s in tags.tagged_spans(target)}
    b = {(s.start, s.end, s.text) for s in tags.tagged_spans(verifier_target)}
    if a != b:
        return GateResult(False, "verifier_disagreement")
    return GateResult(True)


def gate(example: Example, verifier_target: str | None = None) -> GateResult:
    """Run the full gate on one example. Returns the first failure, or ok."""
    for res in (
        check_wellformed(example.target),
        check_integrity(example.input, example.target),
        check_schema(example),
        check_category_semantics(example),
        check_teacher_agreement(example.target, verifier_target),
    ):
        if not res.ok:
            return res
    return GateResult(True)


def filter_examples(
    examples: list[Example],
    verifier_targets: list[str | None] | None = None,
) -> tuple[list[Example], dict[str, int]]:
    """Return (kept, drop_counts_by_reason). ``verifier_targets`` aligns 1:1 with ``examples``."""
    if verifier_targets is None:
        verifier_targets = [None] * len(examples)
    kept: list[Example] = []
    drops: dict[str, int] = {}
    for ex, vt in zip(examples, verifier_targets):
        res = gate(ex, vt)
        if res.ok:
            kept.append(ex)
        else:
            key = res.reason.split(":", 1)[0]
            drops[key] = drops.get(key, 0) + 1
    return kept, drops
