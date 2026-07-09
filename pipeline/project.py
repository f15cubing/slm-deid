"""Tag-by-offset projection — the architecture fix for the integrity failure mode.

**The problem.** The SLM's output contract is "regenerate the whole passage verbatim, adding
only ``⟨NAME⟩`` tags." On clean inputs that round-trips losslessly. On messy *real* text the
held-out CRAPII probe showed the judgment generalizes (recall ~0.88) but strict byte-identity
integrity fails ~100% — the model silently collapses double-spaces, drops zero-width chars,
normalizes newlines, or (on long generations) repeats a tail of tokens. None of that is a
judgment error; it's the "reproduce the passage" format drifting.

**The fix.** Never trust the model's copy of the text. Take only its *judgment* — which spans
are names — and re-apply it onto the ORIGINAL bytes by character offset. Concretely:

1. ``unwrap`` the model output to get its (possibly drifted) plain text and the name spans in
   *its* coordinate system.
2. Character-align the model's plain text to the original with :class:`difflib.SequenceMatcher`.
3. Map each name span's offsets through that alignment onto original offsets. Spans that don't
   align (e.g. a hallucinated repetition tail that isn't in the original) are dropped.
4. Emit the ORIGINAL text with tags inserted at the mapped offsets.

Because step 4 only ever *inserts markers into the original string*, integrity holds **by
construction**: ``tags.unwrap(project_tags(original, anything)) == original`` for all inputs.
This removes the integrity failure mode entirely, without retraining.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from src.common import tags
from src.common.tags import TaggedSpan


@dataclass(frozen=True)
class ProjectionResult:
    """Outcome of projecting a model's tagged output onto the original text."""

    spans: list[TaggedSpan]  # name spans on ORIGINAL coordinates (sorted, non-overlapping)
    tagged: str  # original text with NAME tags inserted (integrity-preserving)
    dropped: int  # model spans that could not be aligned onto the original (repetition tail)


def _offset_map(model_text: str, original: str) -> dict[int, int]:
    """Map each aligned index in ``model_text`` to its index in ``original``.

    Only positions inside a shared matching block appear in the map; positions the model
    inserted/changed (drift) are absent, which is exactly what lets us drop them.
    """
    sm = difflib.SequenceMatcher(a=model_text, b=original, autojunk=False)
    mapping: dict[int, int] = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            mapping[i + k] = j + k
    return mapping


def _merge_overlaps(spans: list[TaggedSpan]) -> list[TaggedSpan]:
    """Sort by start and coalesce overlapping/duplicate spans (keeps tags well-formed)."""
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[TaggedSpan] = []
    for sp in ordered:
        if merged and sp.start < merged[-1].end:
            prev = merged[-1]
            if sp.end <= prev.end:
                continue  # fully contained
            merged[-1] = TaggedSpan(prev.start, sp.end, "")  # text filled in by caller
        else:
            merged.append(sp)
    return merged


def project_spans(original: str, model_tagged: str) -> list[TaggedSpan]:
    """Project the model's name spans onto ``original`` offsets.

    Returns sorted, non-overlapping :class:`TaggedSpan`\\s whose ``(start, end)`` index into
    ``original`` and whose ``text`` is the corresponding original substring. Spans that cannot
    be aligned (drift/hallucination not present in the original) are silently dropped — use
    :func:`project` if you need the drop count.
    """
    return project(original, model_tagged).spans


def project(original: str, model_tagged: str) -> ProjectionResult:
    """Full projection: original-coordinate spans, the tagged original, and the drop count."""
    model_text = tags.unwrap(model_tagged)
    model_spans = tags.tagged_spans(model_tagged)  # offsets into model_text
    omap = _offset_map(model_text, original)

    projected: list[TaggedSpan] = []
    dropped = 0
    for sp in model_spans:
        mapped = [omap[i] for i in range(sp.start, sp.end) if i in omap]
        if not mapped:
            dropped += 1  # nothing in this span aligned to the original → drift; drop it
            continue
        start, end = min(mapped), max(mapped) + 1
        projected.append(TaggedSpan(start, end, original[start:end]))

    merged = _merge_overlaps(projected)
    # _merge_overlaps leaves coalesced spans with empty text; refill from the original slice.
    final = [TaggedSpan(s.start, s.end, original[s.start : s.end]) for s in merged]
    return ProjectionResult(spans=final, tagged=render_tagged(original, final), dropped=dropped)


def render_tagged(original: str, spans: list[TaggedSpan]) -> str:
    """Insert NAME tags into ``original`` at ``spans`` (sorted, non-overlapping).

    Integrity is guaranteed: the result differs from ``original`` only by inserted tag markers,
    so ``tags.unwrap(render_tagged(original, spans)) == original``.
    """
    out: list[str] = []
    cursor = 0
    for sp in sorted(spans, key=lambda s: s.start):
        out.append(original[cursor : sp.start])
        out.append(tags.wrap(original[sp.start : sp.end]))
        cursor = sp.end
    out.append(original[cursor:])
    return "".join(out)


def project_tags(original: str, model_tagged: str) -> str:
    """Convenience: return ``original`` re-tagged from the model's judgment (integrity-safe)."""
    return project(original, model_tagged).tagged
