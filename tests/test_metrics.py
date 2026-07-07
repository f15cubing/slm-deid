"""Day 2, spec S2.3 — metrics on a hand-made fixture with known expected values."""

import math

from src.common import tags
from src.common.schema import Example, Span
from src.eval import metrics as M


def _ex(id_, raw, target, spans, category="easy", pg=None):
    return Example(
        id=id_, input=raw, target=target, spans=spans, category=category, paraphrase_group=pg
    ).validate()


def test_fbeta_recall_weighted():
    # F5 should sit far closer to recall than precision.
    p, r = 0.5, 1.0
    f5 = M.fbeta(p, r, 5.0)
    assert f5 > 0.9  # recall-dominated
    # symmetric sanity: equal p=r -> f5 == p
    assert math.isclose(M.fbeta(0.8, 0.8, 5.0), 0.8, rel_tol=1e-9)


def test_perfect_run():
    raw = "Sarah wrote this."
    tgt = f"{tags.wrap('Sarah')} wrote this."
    ex = _ex("a", raw, tgt, [Span(0, 5, "Sarah", True)])
    m = M.compute([ex], [tgt])
    assert (m.tp, m.fp, m.fn) == (1, 0, 0)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f5 == 1.0
    assert m.leakage_rate == 0.0 and m.over_tag_rate == 0.0
    assert m.integrity_violation_rate == 0.0 and m.pass_rate == 1.0


def test_mixed_run_counts_and_rates():
    # ex1: perfect (1 TP). ex2: leaked the name (1 FN). ex3: over-tagged an eponym (1 FP).
    raw1 = "Ada coded."
    tgt1 = f"{tags.wrap('Ada')} coded."
    ex1 = _ex("1", raw1, tgt1, [Span(0, 3, "Ada", True)])

    raw2 = "Thanks Sam."
    tgt2 = f"Thanks {tags.wrap('Sam')}."
    ex2 = _ex("2", raw2, tgt2, [Span(7, 10, "Sam", True)])
    out2 = raw2  # leaked

    raw3 = "The Newton method works."
    ex3 = _ex("3", raw3, raw3, [Span(4, 10, "Newton", False)])
    out3 = f"The {tags.wrap('Newton')} method works."  # over-tag

    m = M.compute([ex1, ex2, ex3], [tgt1, out2, out3])
    assert (m.tp, m.fp, m.fn) == (1, 1, 1)
    assert math.isclose(m.precision, 0.5)
    assert math.isclose(m.recall, 0.5)
    # rates: 1 of 3 leaked, 1 of 3 over-tagged, 0 integrity fails, 1 of 3 fully passed
    assert math.isclose(m.leakage_rate, 1 / 3)
    assert math.isclose(m.over_tag_rate, 1 / 3)
    assert m.integrity_violation_rate == 0.0
    assert math.isclose(m.pass_rate, 1 / 3)


def test_integrity_violation_rate():
    raw = "Sarah wrote this."
    tgt = f"{tags.wrap('Sarah')} wrote this."
    ex = _ex("a", raw, tgt, [Span(0, 5, "Sarah", True)])
    bad = f"{tags.wrap('Sarah')} wrote"  # dropped text
    m = M.compute([ex], [bad])
    assert m.integrity_violation_rate == 1.0
    assert m.pass_rate == 0.0
    assert m.fn == 1  # gold counts as missed on integrity fail


def test_consistency_across_paraphrases():
    # Group pg1: both pass -> consistent. Group pg2: one pass one fail -> inconsistent.
    raw = "Ada coded."
    tgt = f"{tags.wrap('Ada')} coded."
    a1 = _ex("a1", raw, tgt, [Span(0, 3, "Ada", True)], pg="pg1")
    a2 = _ex("a2", raw, tgt, [Span(0, 3, "Ada", True)], pg="pg1")

    b1 = _ex("b1", raw, tgt, [Span(0, 3, "Ada", True)], pg="pg2")
    b2 = _ex("b2", raw, tgt, [Span(0, 3, "Ada", True)], pg="pg2")

    outs = [tgt, tgt, tgt, raw]  # b2 leaks -> pg2 inconsistent
    m = M.compute([a1, a2, b1, b2], outs)
    assert m.consistency == 0.5  # 1 of 2 groups consistent


def test_consistency_none_without_groups():
    raw = "Ada coded."
    tgt = f"{tags.wrap('Ada')} coded."
    ex = _ex("a", raw, tgt, [Span(0, 3, "Ada", True)])
    assert M.compute([ex], [tgt]).consistency is None


def test_by_category_and_markdown():
    raw = "The Newton method works."
    ex = _ex("3", raw, raw, [Span(4, 10, "Newton", False)], category="person_vs_eponym")
    per_cat = M.by_category([ex], [raw])
    assert "person_vs_eponym" in per_cat
    table = M.markdown_table({"base": M.compute([ex], [raw])})
    assert "| model | n |" in table and "base" in table
