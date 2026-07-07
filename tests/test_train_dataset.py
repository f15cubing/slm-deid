"""Day 2 — pure parts of the SFT dataset builder (model-free)."""

from src.common import tags
from src.common.schema import Example, Span
from src.train.dataset import RESPONSE_TEMPLATE, example_to_messages


def test_example_to_messages_shape():
    raw = "Sarah wrote this."
    tgt = f"{tags.wrap('Sarah')} wrote this."
    ex = Example(id="a", input=raw, target=tgt, spans=[Span(0, 5, "Sarah", True)]).validate()
    msgs = example_to_messages(ex)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == raw
    assert msgs[2]["content"] == tgt


def test_response_template_is_assistant_marker():
    assert "assistant" in RESPONSE_TEMPLATE
