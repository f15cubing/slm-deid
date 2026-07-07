"""Day 2 — the Tagger contract (model-free parts)."""

from src.common import tags
from src.infer import FunctionTagger, Tagger, tag_all


def test_function_tagger_satisfies_protocol():
    t = FunctionTagger(lambda p: p, name="identity")
    assert isinstance(t, Tagger)
    assert t.tag("hello") == "hello"


def test_tag_all_maps_over_passages():
    # A trivial rules tagger that wraps the literal word "Sarah".
    def rules(p: str) -> str:
        return p.replace("Sarah", tags.wrap("Sarah"))

    t = FunctionTagger(rules)
    out = tag_all(t, ["Sarah wrote.", "no names here"])
    assert out == [f"{tags.wrap('Sarah')} wrote.", "no names here"]
