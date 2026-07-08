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

# Category hints for the single-passage generator. These deliberately use ONLY tokens from the
# curated, eval-disjoint bank (src/datagen/vocab.py) as illustrative examples — never an eval
# token — so the quarantined hard-cases set stays a clean generalization test (Day-4 root
# problem 1; enforced by tests/test_teacher.py::test_category_hints_no_longer_seed_eval_tokens).
_CATEGORY_HINT = {
    "person_vs_eponym": "include a surname that is ALSO a unit/law/method (e.g. Euler, Joule, "
    "Hertz); tag it only when it names a person, not the unit/method.",
    "person_vs_place": "include a token that is ALSO a place (e.g. Austin, Sydney, Savannah); "
    "tag it only when it names a person.",
    "person_vs_common": "include a name that is ALSO a common word (e.g. Joy, Dawn, Iris); "
    "tag it only when it names a person.",
    "first_name_only": "refer to a person by first name only, mid-sentence, no title.",
    "possessive": "use a possessive form; tag a person's possessive (e.g. Joy's) but not an "
    "eponymous possessive (e.g. Joule's law).",
    "third_party": "have the author mention a friend/teacher/parent by name (a third party).",
    "negative_trap": "include capitalized NON-names (course titles, brands, sentence-initial "
    "words) and tag NONE of them.",
    "easy": "a clear, unambiguous personal name (with a title is fine).",
}

# How the NON-person sense of each ambiguous surface reads, used to phrase the minimal-pair
# prompt. Keeps the pair a true contrast on the identical token (or, for possessives, on the
# identical possessive FORM).
_NONPERSON_SENSE = {
    "person_vs_eponym": "the unit / law / method named after someone (not a person here)",
    "person_vs_place": "the place (a city/region, not a person here)",
    "person_vs_common": "the ordinary common word (not a person here)",
    "possessive": "an eponymous possessive (a law/method, e.g. Joule's law), not a person",
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


def _pair_user(token: str, category: str, register: str, sense: str) -> str:
    """User prompt for one half of a minimal pair. ``sense`` is 'person' or 'nonperson'.

    The ``SENSE=<sense>`` marker keeps the two halves trivially distinguishable (real teacher +
    mocked teacher in tests).
    """
    if sense == "person":
        directive = (
            f'Write about a PERSON whose name is "{token}". Use "{token}" itself as the person\'s '
            f"name and wrap it in {tags.NAME_OPEN}{token}{tags.NAME_CLOSE}. Do NOT introduce any "
            f"OTHER real person by name — only this person is named."
        )
    else:
        nonperson = _NONPERSON_SENSE.get(category, "its non-person sense")
        directive = (
            f'Write ONLY about "{token}" as {nonperson}. Do NOT mention ANY person by name, and '
            f"tag NOTHING at all."
        )
    return (
        f"Register: {register}. Category: {category}. SENSE={sense}. "
        f"Write ONE short, natural educational passage. {directive}"
    )


class TeacherGenerator:
    def __init__(self, gen: Complete, verify: Complete | None = None):
        self._gen = gen
        self._verify = verify

    def _example(
        self,
        tagged: str,
        *,
        category: str,
        register: str,
        id_: str,
        token: str | None,
    ) -> Example:
        raw, target, spans = parse_tagged(tagged)
        return Example(
            id=id_,
            input=raw,
            target=target,
            register=register,
            category=category,
            spans=spans,
            source="synthetic_teacher",
            quarantine=False,
            ambiguous_token=token,
        )

    def generate(
        self,
        category: str,
        register: str = "essay",
        id_: str = "gen",
        token: str | None = None,
    ) -> Example:
        hint = _CATEGORY_HINT.get(category, "")
        user = f"Register: {register}. Category: {category}. {hint}"
        return self._example(
            self._gen(GEN_SYSTEM, user),
            category=category,
            register=register,
            id_=id_,
            token=token,
        )

    def generate_pair(
        self,
        category: str,
        *,
        person_token: str,
        nonperson_token: str | None = None,
        register: str = "essay",
        id_prefix: str = "pair",
    ) -> tuple[Example, Example]:
        """Generate a MATCHED minimal pair for one ambiguous surface.

        Returns ``(person_example, nonperson_example)``: the first uses the token as a person
        (tagged), the second uses the identical token (or, for possessives, the eponymous
        counterpart ``nonperson_token``) in its non-person sense (untagged). Both carry their
        ambiguous token so the category-semantics gate and the token-leakage guard can act on them.
        """
        nonperson_token = nonperson_token or person_token
        person = self._example(
            self._gen(GEN_SYSTEM, _pair_user(person_token, category, register, "person")),
            category=category,
            register=register,
            id_=f"{id_prefix}-{person_token}-person",
            token=person_token,
        )
        nonperson = self._example(
            self._gen(GEN_SYSTEM, _pair_user(nonperson_token, category, register, "nonperson")),
            category=category,
            register=register,
            id_=f"{id_prefix}-{nonperson_token}-nonperson",
            token=nonperson_token,
        )
        return person, nonperson

    def verify_tagging(self, input_text: str) -> str | None:
        """Second-pass re-tag of the raw passage; returns the verifier's tagged string."""
        if self._verify is None:
            return None
        return _clean(self._verify(VERIFY_SYSTEM, input_text))
