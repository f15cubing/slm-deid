"""Frontier-teacher distillation for ambiguous name passages (Day 2, spec S2.8).

The teacher generates short educational passages (essay or dialogue) that **deliberately contain an
ambiguous name/non-name token** for a target category, returned inline-tagged (persons wrapped in
``⟨NAME⟩…⟨/NAME⟩``). A **second pass** re-tags the same raw passage from scratch; disagreements are
dropped by the quality gate (``teacher.py`` produces the verifier tagging, ``quality_gate.py``
compares).

Network is at the edge: :class:`TeacherGenerator` takes ``complete(system, user) -> str`` callables
(mockable in tests). Real clients reuse the lazy factories in ``src.eval.judge``.
"""

from __future__ import annotations

from typing import Callable

from src.common import tags
from src.common.schema import Example, Span

Complete = Callable[[str, str], str]

_CATEGORY_HINT = {
    "person_vs_eponym": "include a surname that is ALSO a method/unit/theorem (Newton, Gauss, "
    "Pascal, Turing); tag it only when it names a person, not the method.",
    "person_vs_place": "include a token that is ALSO a place (Chelsea, Darwin, Florence, Jordan); "
    "tag it only when it names a person.",
    "person_vs_common": "include a name that is ALSO a common word (Grace, Hope, Faith, Mark, "
    "Rose, May, Bishop, Baker); tag it only when it names a person.",
    "first_name_only": "refer to a person by first name only, mid-sentence, no title.",
    "possessive": "use a possessive form; tag a person's possessive (Sarah's) but not an eponymous "
    "possessive (Newton's laws).",
    "third_party": "have the author mention a friend/teacher/parent by name (a third party).",
    "negative_trap": "include capitalized NON-names (course titles, brands, sentence-initial "
    "words) and tag NONE of them.",
    "easy": "a clear, unambiguous personal name (with a title is fine).",
}

GEN_SYSTEM = (
    "You generate training data for a name-tagging de-identifier. Produce ONE short, natural "
    "educational passage. Wrap every real person's name in "
    f"{tags.NAME_OPEN}…{tags.NAME_CLOSE} and tag NOTHING else (no emails, places, methods, "
    "or common words). Do not add commentary. Return ONLY the tagged passage."
)

VERIFY_SYSTEM = (
    "You are a strict re-tagger. Given a raw passage, return it UNCHANGED except that every real "
    f"person's name is wrapped in {tags.NAME_OPEN}…{tags.NAME_CLOSE}. Change nothing else. Return "
    "ONLY the tagged passage."
)


def _clean(text: str) -> str:
    """Strip code fences / stray whitespace a chat model may add."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


def parse_tagged(text: str) -> tuple[str, str, list[Span]]:
    """From inline-tagged ``text`` derive (input, target, gold name spans)."""
    target = _clean(text)
    raw = tags.unwrap(target)
    spans = [Span(s.start, s.end, s.text, True) for s in tags.tagged_spans(target)]
    return raw, target, spans


class TeacherGenerator:
    def __init__(self, gen: Complete, verify: Complete | None = None):
        self._gen = gen
        self._verify = verify

    def generate(self, category: str, register: str = "essay", id_: str = "gen") -> Example:
        hint = _CATEGORY_HINT.get(category, "")
        user = f"Register: {register}. Category: {category}. {hint}"
        raw, target, spans = parse_tagged(self._gen(GEN_SYSTEM, user))
        return Example(
            id=id_,
            input=raw,
            target=target,
            register=register,
            category=category,
            spans=spans,
            source="synthetic_teacher",
            quarantine=False,
        )

    def verify_tagging(self, input_text: str) -> str | None:
        """Second-pass re-tag of the raw passage; returns the verifier's tagged string."""
        if self._verify is None:
            return None
        return _clean(self._verify(VERIFY_SYSTEM, input_text))
