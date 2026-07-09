"""Guard: the held-out-names probe uses names the model has genuinely never seen.

``eval/heldout_names`` is a generalization test — it only means something if every ambiguous
surface in it is absent from everything the model could have learned from:

- the training / val / co-occurrence splits (passage text AND ``ambiguous_token`` labels),
- the data-gen vocab bank + ``BLOCKLIST`` (the teacher's whole name pool), and
- the existing ``eval/hardcases`` vocabulary (so this is a fresh, non-overlapping probe).

If any of those pick up a held-out name, the "never seen before" claim is false and this test
fails. (Schema validity / quarantine / handbuilt-source are already enforced for every
``eval/**/*.jsonl`` file by ``tests/test_no_eval_leakage.py``.) Regenerate the set with
``python scripts/gen_heldout_names_testset.py`` — its filter drops colliding names automatically.
"""

import re
from pathlib import Path

import pytest

from src.common.schema import read_jsonl
from src.datagen import vocab

REPO = Path(__file__).resolve().parents[1]
HELDOUT = REPO / "eval" / "heldout_names" / "heldout_names.jsonl"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _training_vocab() -> set[str]:
    vocab_words: set[str] = set()
    for rel in ("data/splits/train.jsonl", "data/splits/val.jsonl", "data/cooccur/cooccur.jsonl"):
        p = REPO / rel
        if not p.exists():
            continue
        for ex in read_jsonl(p):
            vocab_words |= _words(ex.input)
            vocab_words |= _words(ex.target)
            if ex.ambiguous_token:
                vocab_words |= _words(ex.ambiguous_token)
    return vocab_words


def _heldout_examples():
    assert HELDOUT.exists(), (
        "eval/heldout_names/heldout_names.jsonl missing — "
        "run python scripts/gen_heldout_names_testset.py"
    )
    return read_jsonl(HELDOUT)


def test_heldout_names_absent_from_training():
    """No held-out ambiguous name (nor any word inside it) appears in the training text/labels."""
    examples = _heldout_examples()
    tokens = {ex.ambiguous_token for ex in examples if ex.ambiguous_token}
    assert tokens, "held-out set has no ambiguous_token values to check"

    train_vocab = _training_vocab()
    if not train_vocab:
        pytest.skip("no training splits present — nothing to be disjoint from")

    offenders = {t: sorted(_words(t) & train_vocab) for t in tokens if _words(t) & train_vocab}
    assert not offenders, f"held-out names leak into training vocab (NOT unseen): {offenders}"


def test_heldout_names_absent_from_bank_and_blocklist():
    """Held-out names are outside the teacher's vocab bank + BLOCKLIST (never a data-gen seed)."""
    bank = {w for tok in vocab.all_bank_tokens() for w in _words(tok)}
    block = {w for tok in vocab.BLOCKLIST for w in _words(tok)}
    tokens = {ex.ambiguous_token for ex in _heldout_examples() if ex.ambiguous_token}

    bank_hits = {t: sorted(_words(t) & bank) for t in tokens if _words(t) & bank}
    block_hits = {t: sorted(_words(t) & block) for t in tokens if _words(t) & block}
    assert not bank_hits, f"held-out names overlap the data-gen bank: {bank_hits}"
    assert not block_hits, f"held-out names overlap the BLOCKLIST: {block_hits}"


def test_heldout_names_absent_from_existing_eval():
    """Fresh probe: held-out names don't reuse the eval/hardcases surfaces."""
    hc_vocab = vocab.eval_vocab(str(REPO / "eval" / "hardcases"))
    tokens = {ex.ambiguous_token for ex in _heldout_examples() if ex.ambiguous_token}
    offenders = {t: sorted(_words(t) & hc_vocab) for t in tokens if _words(t) & hc_vocab}
    assert not offenders, f"held-out names overlap eval/hardcases (not a fresh probe): {offenders}"
