"""A heuristic name-tagger stub for offline demos and tests — NOT the real judgment.

The whole point of the project is that a heuristic *cannot* do context-sensitive name judgment
(it over-tags "the Newton method", place names, common words). This stub exists only so the
pipeline plumbing — pattern detection, offset projection, surrogate replacement, all three
render modes — can be exercised on CPU with no model loaded. Swap it for a real
:class:`~src.infer.Tagger` (base or tuned adapter, via ``src.infer.load_hf_tagger``) for actual
de-identification.

It deliberately mimics the model's *output contract* (regenerate the passage with inline
``⟨NAME⟩`` tags) so the projection layer is exercised exactly as it will be in production.
"""

from __future__ import annotations

import re

from src.common import tags

# Capitalized tokens that are common sentence-initial / non-name words we should not tag, so the
# demo doesn't look absurd. This is a fig leaf, not judgment — the real model handles context.
_STOP = {
    "The",
    "A",
    "An",
    "I",
    "We",
    "They",
    "He",
    "She",
    "It",
    "This",
    "That",
    "My",
    "Our",
    "Newton",
    "Chelsea",
    "Darwin",
    "Florence",
    "Grace",
    "Hope",
    "Faith",
    "Bishop",
}

_CAP_TOKEN = re.compile(r"\b[A-Z][a-z]+\b")


class HeuristicNameStub:
    """Tags capitalized, non-sentence-initial, non-stoplisted tokens. Placeholder only."""

    name = "heuristic-stub"

    def tag(self, passage: str) -> str:
        spans: list[tuple[int, int]] = []
        for m in _CAP_TOKEN.finditer(passage):
            word = m.group(0)
            before = passage[: m.start()].rstrip()
            sentence_initial = m.start() == 0 or before.endswith((".", "!", "?", "\n"))
            if word in _STOP or sentence_initial:
                continue
            spans.append((m.start(), m.end()))

        out: list[str] = []
        cursor = 0
        for s, e in spans:
            out.append(passage[cursor:s])
            out.append(tags.wrap(passage[s:e]))
            cursor = e
        out.append(passage[cursor:])
        return "".join(out)
