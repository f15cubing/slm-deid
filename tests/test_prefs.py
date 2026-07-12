"""Preference-pair builder (src/train/prefs.py) — CPU/pure, no GPU or trl.

Covers: each deterministic perturbation produces a valid off-spec rejected (integrity preserved,
well-formed, differs from the gold target); category-appropriate strategy selection; Stage-A
on-policy negatives; the hybrid orchestrator; and the eval-leakage guard (a hard ceiling).
"""

from __future__ import annotations

from src.common import tags
from src.common.schema import Example, Span
from src.train.prefs import (
    Pair,
    build_pairs,
    eval_leak_count,
    make_missed_name,
    make_over_tag,
    make_wrong_boundary,
    pair_from_model_output,
    stage_b_pair,
    to_trl,
)


def _mk(id_: str, text: str, name_spans: list[tuple[int, int]], category: str = "person_vs_common"):
    """Build a validated Example: gold target = ``text`` with NAME tags at ``name_spans``."""
    spans = [Span(s, e, text[s:e], True) for s, e in name_spans]
    # Target = text with tags inserted at the gold spans (only markers added → integrity holds).
    out, cursor = [], 0
    for s, e in sorted(name_spans):
        out.append(text[cursor:s])
        out.append(tags.wrap(text[s:e]))
        cursor = e
    out.append(text[cursor:])
    target = "".join(out)
    return Example(
        id=id_,
        input=text,
        target=target,
        category=category,
        spans=spans,
        source="synthetic_teacher",
    ).validate()


# A name + a capitalized non-name ("Library") → over-tag candidate available.
EX_NAME = _mk("ex-name", "Miles said hello at the Library today", [(0, 5)])
# No gold name; a capitalized trap word.
EX_TRAP = _mk("ex-trap", "The Table of contents lists every Chapter", [], category="negative_trap")
# Possessive after the name → wrong-boundary can swallow the "'s". (Aria is blocklist-safe;
# Sarah/Priya/Devon are eval-vocab surfaces the leakage guard would correctly flag.)
EX_POSS = _mk("ex-poss", "Aria's essay improved", [(0, 4)], category="possessive")


def _integrity_ok(ex: Example, rejected: str) -> bool:
    return tags.is_well_formed(rejected) and tags.unwrap(rejected) == ex.input


def test_over_tag_adds_an_extra_tag():
    rej = make_over_tag(EX_NAME)
    assert rej is not None and rej != EX_NAME.target
    assert _integrity_ok(EX_NAME, rej)
    # More tagged spans than gold (the extra is the capitalized non-name).
    assert len(tags.tagged_spans(rej)) == len(EX_NAME.name_spans()) + 1
    assert "Library" in {t.text for t in tags.tagged_spans(rej)}


def test_missed_name_drops_a_gold_tag():
    rej = make_missed_name(EX_NAME)
    assert rej is not None and rej != EX_NAME.target
    assert _integrity_ok(EX_NAME, rej)
    assert len(tags.tagged_spans(rej)) == len(EX_NAME.name_spans()) - 1


def test_wrong_boundary_swallows_possessive():
    rej = make_wrong_boundary(EX_POSS)
    assert rej is not None and rej != EX_POSS.target
    assert _integrity_ok(EX_POSS, rej)
    assert "Aria's" in {t.text for t in tags.tagged_spans(rej)}


def test_wrong_boundary_preserves_integrity_with_adjacent_names():
    # Two adjacent names; the last-in-LIST span (Ann) is NOT positionally last. Extending it must
    # not swallow the following gold name (Bob) — that would overlap and break unwrap==input.
    # Build the Example with spans deliberately out of positional order to stress the fix.
    text = "Ann Bob left early"
    spans = [Span(4, 7, "Bob", True), Span(0, 3, "Ann", True)]  # unsorted on purpose
    ex = Example(
        id="adj",
        input=text,
        target="⟨NAME⟩Ann⟨/NAME⟩ ⟨NAME⟩Bob⟨/NAME⟩ left early",
        category="third_party",
        spans=spans,
        source="synthetic_teacher",
    ).validate()
    rej = make_wrong_boundary(ex)
    if rej is not None:
        assert tags.is_well_formed(rej) and tags.unwrap(rej) == text  # integrity preserved
    # And the orchestrator never emits a non-integrity deterministic negative regardless.
    pair = stage_b_pair(ex, preferred="wrong_boundary")
    assert pair is None or tags.unwrap(pair.rejected) == text


def test_negative_trap_only_supports_over_tag():
    # No gold name → missed/boundary are inapplicable; over-tag targets the trap word.
    assert make_missed_name(EX_TRAP) is None
    assert make_wrong_boundary(EX_TRAP) is None
    pair = stage_b_pair(EX_TRAP, preferred="missed_name")
    assert pair is not None and pair.strategy == "over_tag"
    assert "Table" in {t.text for t in tags.tagged_spans(pair.rejected)}


def test_stage_b_pair_prefers_requested_strategy_when_applicable():
    assert stage_b_pair(EX_NAME, preferred="missed_name").strategy == "missed_name"
    assert stage_b_pair(EX_NAME, preferred="over_tag").strategy == "over_tag"


def test_pair_from_model_output_keeps_genuine_errors_only():
    wrong = "⟨NAME⟩Miles⟨/NAME⟩ said hello at the ⟨NAME⟩Library⟨/NAME⟩ today"  # over-tags Library
    pair = pair_from_model_output(EX_NAME, wrong)
    assert pair is not None and pair.strategy == "on_policy" and pair.rejected == wrong
    # A correct output (== gold) yields no natural negative.
    assert pair_from_model_output(EX_NAME, EX_NAME.target) is None
    # A passing-but-differently-spaced-correct output also yields nothing (judgment matches).
    assert pair_from_model_output(EX_NAME, "  " + EX_NAME.target) is None


def test_build_pairs_prefers_on_policy_when_available():
    wrong = "⟨NAME⟩Miles⟨/NAME⟩ said hello at the ⟨NAME⟩Library⟨/NAME⟩ today"
    pairs = build_pairs([EX_NAME], model_outputs={EX_NAME.id: wrong})
    assert len(pairs) == 1 and pairs[0].strategy == "on_policy"


def test_build_pairs_rotates_strategies_for_coverage():
    # Three name-bearing examples with no model outputs → round-robin over the strategies.
    exs = [_mk(f"r{i}", "Miles waved near the Market today", [(0, 5)]) for i in range(3)]
    strategies = {p.strategy for p in build_pairs(exs)}
    assert len(strategies) >= 2  # not all the same → coverage across failure modes


def test_all_rejected_differ_from_chosen():
    pairs = build_pairs([EX_NAME, EX_TRAP, EX_POSS])
    assert pairs and all(p.rejected != p.chosen for p in pairs)
    assert all(
        p.chosen == {"ex-name": EX_NAME, "ex-trap": EX_TRAP, "ex-poss": EX_POSS}[p.id].target
        for p in pairs
    )


def test_per_category_cap_balances():
    exs = [_mk(f"c{i}", "Miles waved at the Market", [(0, 5)]) for i in range(5)]
    pairs = build_pairs(exs, per_category_cap=2)
    assert len(pairs) == 2  # all same category, capped


def test_to_trl_conversational_shape():
    trl = to_trl(stage_b_pair(EX_NAME))
    assert [m["role"] for m in trl["prompt"]] == ["system", "user"]
    assert trl["chosen"][0]["role"] == "assistant"
    assert trl["rejected"][0]["role"] == "assistant"
    assert trl["chosen"][0]["content"] == EX_NAME.target


def test_eval_leak_guard_is_zero_on_disjoint_pairs():
    # Synthetic inputs share no surface with the quarantined eval sets → guard reports 0.
    pairs = build_pairs([EX_NAME, EX_TRAP, EX_POSS])
    assert eval_leak_count(pairs, "eval") == 0


def test_eval_leak_guard_flags_an_eval_surface():
    # A pair whose passage carries a BLOCKLIST eval surface ("Newton") must be flagged.
    leaky = Pair("leak", "person_vs_eponym", "Newton was frustrated", "x", "y", "over_tag")
    assert eval_leak_count([leaky], "eval") >= 1
