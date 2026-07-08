"""Curated, eval-DISJOINT vocabulary for teacher data-gen (Day 4, TASK 2).

The Day-3 error analysis found two data problems. This module fixes root problem 1: the teacher's
category hints used to seed the *exact* eval tokens (Newton/Chelsea/Grace/…), so the quarantined
hard-cases set was never a clean generalization test. Here we provide a curated token pool the
teacher may use to build minimal pairs that shares **no token** with the eval set.

Two guarantees, both checked by ``tests/test_vocab.py`` (which derives the eval vocabulary from
``eval/hardcases`` at test time):

- every bank token is absent from the quarantined eval vocabulary, and
- the bank excludes every token on :data:`BLOCKLIST` (the eval tokens + tokens the plan reserves
  for the eval categories).

No network here — this is a static bank plus a pure ``eval_vocab`` reader.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.common.schema import read_jsonl

# Case-insensitive tokens that must NEVER appear in generated training data: every ambiguous
# surface used by the quarantined eval set, plus the tokens ``docs/plan.md`` enumerates for the
# eval categories. The curated bank below is verified disjoint from this set.
BLOCKLIST: frozenset[str] = frozenset(
    {
        # person / ambiguous surfaces used in eval + plan
        "newton", "gauss", "pascal", "turing", "riemann", "chelsea", "darwin", "florence",
        "jordan", "madison", "devon", "grace", "hope", "faith", "bishop", "rose", "may",
        "mark", "bill", "baker", "sarah", "ohm", "sam", "alvarez", "liang", "rivera", "omar",
        "priya", "johnson", "ada", "alan", "maria", "gonzalez", "bob",
        # non-person eval traps
        "psychology", "biology", "android", "march", "april", "red cross",
    }
)

# --- the curated bank (verified disjoint from eval + blocklist) -------------------------
# Places that double as personal names (person_vs_place minimal pairs).
PLACES: tuple[str, ...] = (
    "Austin", "Sydney", "Brooklyn", "Savannah", "Georgia", "Sierra", "Phoenix", "Odessa",
    "Adelaide", "Charlotte", "Houston", "Dakota", "Camden", "Raleigh", "Memphis", "Salem",
)
# Common words / given names (person_vs_common minimal pairs). Also reused as the person token
# for possessive pairs ("Joy's essay" vs the eponymous "Joule's law").
COMMON_WORDS: tuple[str, ...] = (
    "Joy", "Melody", "Daisy", "Iris", "Pearl", "Dawn", "Autumn", "Miles", "Frank", "Rich",
    "Drew", "Sky", "Sunny", "Ivy", "Holly", "Robin", "Jay", "Dale", "Wade",
)
# Surnames that double as units/laws/methods (person_vs_eponym + eponymous-possessive negatives).
EPONYMS: tuple[str, ...] = (
    "Euler", "Fourier", "Hertz", "Watt", "Joule", "Kelvin", "Ampere", "Boole", "Planck",
    "Curie", "Faraday", "Volta", "Bunsen", "Coulomb", "Tesla", "Doppler", "Richter", "Mach",
)

# Per-category token pool for minimal-pair generation. For ``possessive`` the person half draws a
# name from here and the non-person half draws an eponym (see :data:`EPONYMS`).
VOCAB_BANK: dict[str, tuple[str, ...]] = {
    "person_vs_place": PLACES,
    "person_vs_common": COMMON_WORDS,
    "person_vs_eponym": EPONYMS,
    "possessive": COMMON_WORDS,
}


def tokens_for(category: str) -> tuple[str, ...]:
    """Bank tokens available for ``category`` (empty tuple if none)."""
    return VOCAB_BANK.get(category, ())


def all_bank_tokens() -> set[str]:
    """Every token in the bank (deduplicated across categories)."""
    out: set[str] = set()
    for toks in VOCAB_BANK.values():
        out.update(toks)
    out.update(EPONYMS)  # used as the non-person half of possessive pairs
    return out


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def eval_vocab(eval_dir: str = "eval") -> set[str]:
    """Derive the eval vocabulary: every lowercased word token in ``eval/**/*.jsonl``.

    Used to (a) assert the bank is disjoint from eval (``tests/test_vocab.py``) and (b) drop any
    generated example whose ambiguous target token overlaps the eval set (token-level leakage
    guard in ``generate.py``). Reading ``eval`` is safe — the leakage ceiling forbids the reverse
    direction (eval text flowing INTO training), which this very function helps enforce.
    """
    p = Path(eval_dir)
    vocab: set[str] = set()
    if not p.exists():
        return vocab
    for f in p.rglob("*.jsonl"):
        for ex in read_jsonl(f):
            vocab |= _words(ex.input)
            vocab |= _words(ex.target)
    return vocab


def token_words(token: str) -> set[str]:
    """Lowercased word tokens inside ``token`` (handles multi-word names like 'Maria Gonzalez')."""
    return _words(token)
