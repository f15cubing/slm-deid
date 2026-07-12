"""CPU test for src/demo.py — proves the base-vs-tuned demo wiring without loading a model.

Uses FunctionTagger fakes: a "base" that over-tags everything and a "tuned" that only tags the
person cases. Verifies run_demo pairs outputs correctly, differ/tag-count helpers work, and the
rendered block shows both columns.
"""

from src.common import prompts
from src.demo import format_demo, run_demo
from src.infer import FunctionTagger


def _over_tagger():
    # Naive base: wraps the first whitespace token of every passage in NAME tags (over-tags).
    def fn(p: str) -> str:
        head, _, rest = p.partition(" ")
        return f"⟨NAME⟩{head}⟨/NAME⟩ {rest}" if rest else p

    return FunctionTagger(fn, name="base")


def _identity_tagger():
    return FunctionTagger(lambda p: p, name="tuned")


def test_run_demo_pairs_base_and_tuned_over_showcase():
    rows = run_demo(_over_tagger(), _identity_tagger())
    assert len(rows) == len(prompts.SHOWCASE)
    # Every row carries the intended-judgment note and both model outputs.
    for row, (passage, note) in zip(rows, prompts.SHOWCASE):
        assert row.passage == passage
        assert row.note == note
        assert row.base and row.tuned


def test_differ_and_tag_counts_reflect_outputs():
    rows = run_demo(_over_tagger(), _identity_tagger())
    # The over-tagger inserts exactly one tag per passage; identity inserts none → they differ.
    assert all(r.base_tags == 1 for r in rows)
    assert all(r.tuned_tags == 0 for r in rows)
    assert all(r.differ for r in rows)


def test_custom_passages_are_honored():
    rows = run_demo(
        _identity_tagger(),
        _identity_tagger(),
        passages=[("Sam waved.", "person → tag")],
    )
    assert len(rows) == 1
    assert not rows[0].differ  # identical taggers → no disagreement


def test_format_demo_renders_both_columns():
    rows = run_demo(_over_tagger(), _identity_tagger())
    out = format_demo(rows)
    assert "base" in out and "tuned" in out
    assert "models disagree" in out
    # header reports the disagreement count
    assert f"{len(rows)}/{len(rows)} passages" in out
