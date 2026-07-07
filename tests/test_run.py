"""Day 2, spec S2.5 — the evaluate/compare scaffold (model-free, FunctionTagger)."""

from src.common import tags
from src.common.schema import Example, Span
from src.eval.run import _load_examples, compare, evaluate
from src.infer import FunctionTagger


def _examples():
    raw1 = "Ada coded."
    tgt1 = f"{tags.wrap('Ada')} coded."
    e1 = Example(id="1", input=raw1, target=tgt1, spans=[Span(0, 3, "Ada", True)],
                 category="easy").validate()
    raw2 = "The Newton method works."
    e2 = Example(id="2", input=raw2, target=raw2, spans=[Span(4, 10, "Newton", False)],
                 category="person_vs_eponym").validate()
    return [e1, e2]


def test_evaluate_with_perfect_tagger():
    examples = _examples()
    gold = {ex.input: ex.target for ex in examples}
    perfect = FunctionTagger(lambda p: gold[p], name="perfect")
    rep = evaluate(perfect, examples)
    assert rep.label == "perfect"
    assert rep.n == 2
    assert rep.overall.pass_rate == 1.0
    assert rep.overall.recall == 1.0
    assert "person_vs_eponym" in rep.by_category
    assert all(item["pass"] for item in rep.outputs)


def test_evaluate_with_over_tagging_tagger():
    examples = _examples()
    # A tagger that tags EVERY capitalized word -> over-tags the Newton eponym.
    def overtag(p: str) -> str:
        out = p
        for w in ("Ada", "Newton"):
            out = out.replace(w, tags.wrap(w))
        return out

    rep = evaluate(FunctionTagger(overtag, name="overtag"), examples)
    assert rep.overall.over_tag_rate > 0.0
    assert rep.by_category["person_vs_eponym"].over_tag_rate == 1.0


def test_compare_table_lists_all_labels():
    examples = _examples()
    gold = {ex.input: ex.target for ex in examples}
    r1 = evaluate(FunctionTagger(lambda p: gold[p], name="base"), examples)
    r2 = evaluate(FunctionTagger(lambda p: p, name="tuned"), examples)  # tags nothing
    table = compare([r1, r2])
    assert "base" in table and "tuned" in table and "| model | n |" in table


def test_load_examples_from_quarantined_dir():
    # The real hard-cases set loads and validates through the loader.
    examples = _load_examples("eval/hardcases")
    assert len(examples) >= 40
    assert all(ex.quarantine for ex in examples)
