"""The single source of truth for the NAME tag syntax (Day 1, spec S1.2).

The behavior: a passage is returned unchanged except that every span referring to a real
person's name is wrapped in ``⟨NAME⟩…⟨/NAME⟩``. Nothing else in the text may change, so the
core invariant everywhere in this project is::

    unwrap(target) == input   # byte-for-byte

Every other module imports the tag markers and helpers from here — never hard-code the tag
strings elsewhere. To switch to the ``@@…##`` (GPT-NER) fallback, change only this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- The locked syntax (decision recorded in docs/tasks/day-1.md) -----------------------
# Primary: mathematical angle brackets U+27E8 / U+27E9. These are single codepoints, so they
# never collide with the ASCII ``<`` / ``>`` a student might type in prose or code.
# NB: a single codepoint is NOT a single token. On Qwen3's byte-level BPE these markers FRAGMENT
# (OPEN=3 / CLOSE=4 tokens; 8 per tagged span) and are not special tokens — a deliberate
# collision-safety-over-efficiency trade. Integrity still holds (lossless round-trip). Pinned by
# tests/test_tag_tokenization.py; the 1-token variant (added special tokens) is a v-next A/B.
NAME_OPEN = "\u27e8NAME\u27e9"  # ⟨NAME⟩
NAME_CLOSE = "\u27e8/NAME\u27e9"  # ⟨/NAME⟩

# Matches one well-formed, non-nested tagged span; group "name" is the inner text.
_SPAN_RE = re.compile(
    re.escape(NAME_OPEN) + r"(?P<name>.*?)" + re.escape(NAME_CLOSE),
    flags=re.DOTALL,
)
# Any tag marker at all (used for unwrap + well-formedness accounting).
_ANY_MARKER_RE = re.compile("(" + re.escape(NAME_OPEN) + "|" + re.escape(NAME_CLOSE) + ")")


@dataclass(frozen=True)
class TaggedSpan:
    """A name span located against the *unwrapped* (raw) text offsets."""

    start: int
    end: int
    text: str


def wrap(text: str) -> str:
    """Wrap ``text`` in a single NAME tag."""
    return f"{NAME_OPEN}{text}{NAME_CLOSE}"


def unwrap(text: str) -> str:
    """Remove every NAME tag marker, leaving the underlying text.

    ``unwrap(wrap(x)) == x`` for any ``x`` (including empty). This is the function the
    integrity check relies on, so it must strip markers only — never touch other characters.
    """
    return _ANY_MARKER_RE.sub("", text)


def is_well_formed(text: str) -> bool:
    """True iff every open marker has a matching close, in order, with no nesting.

    Rejects: stray/unbalanced markers, a close before an open, and nested opens
    (``⟨NAME⟩a⟨NAME⟩b⟨/NAME⟩⟨/NAME⟩``). Empty spans (``⟨NAME⟩⟨/NAME⟩``) are considered
    well-formed *structurally* (a separate content check can reject them if desired).
    """
    depth = 0
    for marker in _ANY_MARKER_RE.findall(text):
        if marker == NAME_OPEN:
            depth += 1
            if depth > 1:  # nesting
                return False
        else:  # NAME_CLOSE
            depth -= 1
            if depth < 0:  # close before open
                return False
    return depth == 0


def tagged_spans(tagged: str) -> list[TaggedSpan]:
    """Return the name spans in ``tagged``, with offsets into the *unwrapped* text.

    Example::

        >>> tagged_spans("Hi ⟨NAME⟩Sam⟨/NAME⟩!")
        [TaggedSpan(start=3, end=6, text='Sam')]

    The returned offsets index into ``unwrap(tagged)`` so gold spans and model output can be
    compared on a common coordinate system (used by the Day-2 behavioral checks).
    """
    spans: list[TaggedSpan] = []
    raw_cursor = 0  # position in the unwrapped string
    last_end = 0  # position in the tagged string we've consumed up to
    for m in _SPAN_RE.finditer(tagged):
        # advance raw_cursor by the untagged text between the previous match and this one
        between = unwrap(tagged[last_end : m.start()])
        raw_cursor += len(between)
        name = m.group("name")
        spans.append(TaggedSpan(start=raw_cursor, end=raw_cursor + len(name), text=name))
        raw_cursor += len(name)
        last_end = m.end()
    return spans
