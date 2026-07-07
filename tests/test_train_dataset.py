"""Day 2 — pure parts of the SFT dataset builder (model-free)."""

from src.common import tags
from src.common.schema import Example, Span
from src.train.dataset import example_to_messages, example_to_prompt_completion


def _ex():
    raw = "Sarah wrote this."
    tgt = f"{tags.wrap('Sarah')} wrote this."
    return Example(id="a", input=raw, target=tgt, spans=[Span(0, 5, "Sarah", True)]).validate()


def test_example_to_messages_shape():
    msgs = example_to_messages(_ex())
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == "Sarah wrote this."
    assert msgs[2]["content"] == f"{tags.wrap('Sarah')} wrote this."


def test_example_to_prompt_completion_shape():
    pc = example_to_prompt_completion(_ex())
    assert set(pc) == {"prompt", "completion"}
    assert [m["role"] for m in pc["prompt"]] == ["system", "user"]
    assert pc["prompt"][1]["content"] == "Sarah wrote this."
    assert pc["completion"] == [
        {"role": "assistant", "content": f"{tags.wrap('Sarah')} wrote this."}
    ]
