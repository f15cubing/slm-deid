"""Tokenization contract for the NAME tag markers (automates the Day-1 tag-syntax finding).

The ``⟨NAME⟩`` / ``⟨/NAME⟩`` markers are single Unicode CODEPOINTS (U+27E8/9), chosen for
collision-safety: they can't be confused with the ASCII ``<`` / ``>`` / ``@@`` / ``##`` a student
might type in prose, markdown, or code. But a single codepoint is NOT a single TOKEN — on Qwen3's
byte-level BPE these markers FRAGMENT (OPEN=3, CLOSE=4 tokens; 8 per tagged span) and are NOT
registered as special tokens. Keeping them is a deliberate collision-safety-over-token-efficiency
trade (see docs/tasks/day-1.md); the 1-token alternative (register them as *added* special tokens)
is a v-next A/B on the docs/plan.md stretch ladder.

These tests pin BOTH facts so the decision stays verified, not assumed:
  * integrity-critical: the markers round-trip losslessly (so ``unwrap(output) == input`` can hold);
  * the markers currently fragment (multi-token) and are not atomic special tokens.

Needs the real Qwen3 tokenizer (cached locally). Skips cleanly if transformers/the tokenizer are
unavailable offline, so the pure-Python suite still runs anywhere.
"""

import os

import pytest

from src.common import tags

MODEL = "Qwen/Qwen3-1.7B"


def _tokenizer():
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    try:
        # cache-only: loads from the local HF cache if present, raises immediately otherwise
        # (no network attempt, so this is fast both when cached and when absent / in CI).
        return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception as e:  # not cached locally
        pytest.skip(f"Qwen3 tokenizer not cached locally: {e}")


def test_name_markers_round_trip_losslessly():
    # The integrity invariant (unwrap(output) == input) relies on byte-level BPE reconstructing the
    # markers exactly, even though they fragment. This property MUST hold.
    tok = _tokenizer()
    for s in (
        tags.NAME_OPEN,
        tags.NAME_CLOSE,
        f"{tags.wrap('Newton')} met {tags.wrap('Grace')} in Florence.",
    ):
        assert tok.decode(tok.encode(s, add_special_tokens=False)) == s


def test_name_markers_fragment_and_are_not_special_tokens():
    # Documents the Day-1 finding as an automated check: the markers are multi-token on Qwen3
    # (OPEN=3 / CLOSE=4 => 8 per span) and are NOT atomic special tokens. If either marker ever
    # becomes a single id, someone registered them as added tokens (the v-next efficiency A/B) —
    # update this test then.
    tok = _tokenizer()
    unk = getattr(tok, "unk_token_id", None)
    assert len(tok.encode(tags.NAME_OPEN, add_special_tokens=False)) > 1
    assert len(tok.encode(tags.NAME_CLOSE, add_special_tokens=False)) > 1
    assert tok.convert_tokens_to_ids(tags.NAME_OPEN) in (None, unk)
    assert tok.convert_tokens_to_ids(tags.NAME_CLOSE) in (None, unk)
