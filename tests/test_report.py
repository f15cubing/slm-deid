"""Spec S3.5 — offline reporting (src.eval.report).

Covers: recompute-from-outputs reproduces a report's saved overall metrics (both for
self-contained reports and for older ones lacking tp/fp/fn), table generation with CIs, and
the recall/pass invariant regression that would have caught the hand-transcribed doc bug
(a category whose items all contain gold names cannot have recall==0 while pass_rate>0).
"""

import glob
import os

import pytest

from src.common import tags
from src.common.schema import Example, Span, write_jsonl
from src.eval import metrics as M
from src.eval import report
from src.eval.run import evaluate
from src.infer import FunctionTagger

REAL_BASE = sorted(glob.glob("outputs/eval_reports/base-*.json"))
REAL_TUNED = sorted(glob.glob("outputs/eval_reports/tuned-*.json"))
HAVE_EVAL = os.path.isdir("eval/hardcases")
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mini_report.json")


def _mixed_examples() -> list[Example]:
    """Two categories; 'easy' items all contain gold names (for the invariant test)."""
    e: list[Example] = []
    raw = "Grace lent me notes."
    tgt = f"{tags.wrap('Grace')} lent me notes."
    e.append(
        Example(
            id="c1",
            input=raw,
            target=tgt,
            spans=[Span(0, 5, "Grace", True)],
            category="person_vs_common",
        ).validate()
    )
    raw = "handled with grace"
    e.append(
        Example(
            id="c2",
            input=raw,
            target=raw,
            spans=[Span(13, 18, "grace", False)],
            category="person_vs_common",
        ).validate()
    )
    raw = "Dear Mr. Johnson,"
    tgt = f"Dear Mr. {tags.wrap('Johnson')},"
    e.append(
        Example(
            id="e1", input=raw, target=tgt, spans=[Span(9, 16, "Johnson", True)], category="easy"
        ).validate()
    )
    raw = "Ada coded."
    tgt = f"{tags.wrap('Ada')} coded."
    e.append(
        Example(
            id="e2", input=raw, target=tgt, spans=[Span(0, 3, "Ada", True)], category="easy"
        ).validate()
    )
    return e


def _report_dict(tagger: FunctionTagger) -> dict:
    return evaluate(tagger, _mixed_examples(), label=tagger.name).to_dict()


def test_evaluate_saves_per_item_counts():
    rep = _report_dict(FunctionTagger(lambda p: {e.input: e.target for e in _mixed_examples()}[p]))
    assert rep["outputs"], "expected per-item outputs"
    for o in rep["outputs"]:
        assert {"tp", "fp", "fn"} <= set(o), "each per-item entry must carry tp/fp/fn"


def test_recompute_from_outputs_matches_saved_overall():
    gold = {e.input: e.target for e in _mixed_examples()}
    rep = _report_dict(FunctionTagger(lambda p: gold[p], name="perfect"))
    records = report.reconstruct_items(rep)  # has tp/fp/fn -> no eval load needed
    recomputed = report.recompute_overall(records)
    for k in M.CI_METRICS:
        assert round(recomputed[k], 4) == round(rep["overall"][k], 4), k


def test_recompute_from_old_report_without_counts(tmp_path):
    gold = {e.input: e.target for e in _mixed_examples()}
    rep = _report_dict(FunctionTagger(lambda p: gold[p], name="perfect"))
    for o in rep["outputs"]:  # simulate a report written before tp/fp/fn existed
        for k in ("tp", "fp", "fn"):
            o.pop(k)
    split = tmp_path / "eval.jsonl"
    write_jsonl(split, _mixed_examples())
    records = report.reconstruct_items(rep, eval_split=str(split))
    recomputed = report.recompute_overall(records)
    for k in M.CI_METRICS:
        assert round(recomputed[k], 4) == round(rep["overall"][k], 4), k


def test_old_report_without_counts_and_no_eval_raises():
    gold = {e.input: e.target for e in _mixed_examples()}
    rep = _report_dict(FunctionTagger(lambda p: gold[p]))
    for o in rep["outputs"]:
        for k in ("tp", "fp", "fn"):
            o.pop(k)
    with pytest.raises((KeyError, FileNotFoundError, Exception)):
        report.reconstruct_items(rep, eval_split="does/not/exist.jsonl")


def test_load_saved_json_fixture_and_render_tables():
    """Table generation from a committed saved-JSON fixture (the load_report path)."""
    rep = report.load_report(FIXTURE)
    # recompute-from-outputs reproduces the fixture's saved overall metrics
    records = report.reconstruct_items(rep)
    recomputed = report.recompute_overall(records)
    for k in M.CI_METRICS:
        assert round(recomputed[k], 4) == round(rep["overall"][k], 4), k
    # and the rendered tables carry CIs + the safety-critical columns
    lr = report.build_labeled_report(rep, n_resamples=200, seed=3)
    md = report.render_tables([lr])
    assert "person_vs_common" in md and "easy" in md
    assert "integrity_violation_rate" in md and "consistency" in md
    assert "0.667" in md  # the fixture's recall point value, rendered
    assert "[" in md and "]" in md
    assert report.recall_pass_invariant_violations(records) == []


def test_render_tables_have_cis_and_key_columns():
    gold = {e.input: e.target for e in _mixed_examples()}
    lr = report.build_labeled_report(
        _report_dict(FunctionTagger(lambda p: gold[p], name="base")),
        label="base",
        n_resamples=100,
        seed=1,
    )
    md = report.render_tables([lr])
    assert "| model | n |" in md
    assert "integrity_violation_rate" in md
    assert "consistency" in md
    assert "person_vs_common" in md and "easy" in md
    assert "[" in md and "]" in md  # a bootstrap CI is rendered in each metric cell


def test_render_overall_has_delta_row_for_two_reports():
    gold = {e.input: e.target for e in _mixed_examples()}
    base = report.build_labeled_report(
        _report_dict(FunctionTagger(lambda p: p, name="base")),
        label="base",
        n_resamples=50,
        seed=1,
    )
    tuned = report.build_labeled_report(
        _report_dict(FunctionTagger(lambda p: gold[p], name="tuned")),
        label="tuned",
        n_resamples=50,
        seed=1,
    )
    md = report.render_overall_table([base, tuned])
    assert "Δ (tuned−base)" in md


# --- the regression: the invariant that would have caught the doc bug -------------------
def test_all_named_category_zero_recall_forces_zero_pass():
    """recall==0 in an all-named category must imply pass_rate==0 (the impossible doc row)."""
    easy = [ex for ex in _mixed_examples() if ex.category == "easy"]
    outs = [ex.input for ex in easy]  # tag nothing -> leak every gold name
    per_cat = M.by_category(easy, outs)
    m = per_cat["easy"]
    assert m.recall == 0.0
    assert m.pass_rate == 0.0  # base 'easy' pass_rate 0.667 in the old doc was impossible


def test_recall_pass_invariant_flags_impossible_row():
    # Fabricate the doc bug directly: all items have a gold name, zero recall, yet one passes.
    bad = [
        report.ItemRecord("a", "easy", M.ItemResult(0, 0, 1, False, True, False, False)),
        report.ItemRecord("b", "easy", M.ItemResult(0, 0, 1, True, False, False, False)),
    ]
    assert report.recall_pass_invariant_violations(bad) == ["easy"]


def test_recall_pass_invariant_ok_on_consistent_records():
    ok = [
        report.ItemRecord("a", "easy", M.ItemResult(0, 0, 1, False, True, False, False)),
        report.ItemRecord("b", "easy", M.ItemResult(1, 0, 0, True, False, False, False)),
    ]
    assert report.recall_pass_invariant_violations(ok) == []


# --- real saved reports (present locally; skipped on a fresh checkout) ------------------
@pytest.mark.skipif(not (REAL_BASE and HAVE_EVAL), reason="real base report / eval set absent")
def test_recompute_matches_real_base_report():
    rep = report.load_report(REAL_BASE[-1])
    records = report.reconstruct_items(rep, eval_split="eval/hardcases")
    recomputed = report.recompute_overall(records)
    for k in M.CI_METRICS:
        assert round(recomputed[k], 4) == round(rep["overall"][k], 4), k


@pytest.mark.skipif(
    not (REAL_BASE and REAL_TUNED and HAVE_EVAL), reason="real reports / eval set absent"
)
def test_real_reports_respect_recall_pass_invariant():
    for path in (REAL_BASE[-1], REAL_TUNED[-1]):
        rep = report.load_report(path)
        records = report.reconstruct_items(rep, eval_split="eval/hardcases")
        assert report.recall_pass_invariant_violations(records) == [], path
