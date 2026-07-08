"""Day 4, TASK 2 — the curated, eval-DISJOINT data-gen vocabulary bank.

Root problem 1 (Day-3 error analysis): the teacher's category hints SEEDED the exact eval tokens
(Newton/Chelsea/Grace/…), so the held-out set was never a clean generalization test. This bank is
the fix: a curated token pool the teacher may use that shares NO token with the quarantined eval
set. These tests derive the eval vocabulary from ``eval/hardcases`` at test time and assert the
bank is disjoint from it and from the explicit blocklist.
"""

from src.datagen import vocab


def test_eval_vocab_is_derived_and_nonempty():
    ev = vocab.eval_vocab("eval")
    assert ev, "eval vocabulary should be derived from eval/hardcases"
    # spot-check: known eval tokens are present
    assert {"newton", "chelsea", "grace"} <= ev


def test_bank_is_disjoint_from_eval_vocabulary():
    ev = vocab.eval_vocab("eval")
    overlap = {t for t in vocab.all_bank_tokens() if t.lower() in ev}
    assert not overlap, f"bank tokens leak into the eval vocabulary: {sorted(overlap)}"


def test_bank_excludes_every_blocklisted_token():
    lowered = {t.lower() for t in vocab.all_bank_tokens()}
    hits = {b for b in vocab.BLOCKLIST if b in lowered}
    assert not hits, f"blocklisted (eval) tokens present in bank: {sorted(hits)}"


def test_every_targeted_category_has_tokens():
    for category in ("person_vs_place", "person_vs_common", "person_vs_eponym", "possessive"):
        assert vocab.tokens_for(category), f"no bank tokens for {category}"


def test_blocklist_surfaces_in_flags_eval_names_anywhere():
    # the "Charles Darwin" hole: an eval surface must be detected anywhere in a passage, even when
    # it is not the intended ambiguous token.
    assert "darwin" in vocab.blocklist_surfaces_in("Charles Darwin studied finches for years.")
    assert "red cross" in vocab.blocklist_surfaces_in("The Red Cross ran a first-aid workshop.")
    # a clean passage built only from bank tokens flags nothing
    assert vocab.blocklist_surfaces_in("Austin met Sydney near the harbor in Savannah.") == set()
