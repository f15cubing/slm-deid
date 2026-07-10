"""Offline tests for the frontier-model eval wrapper (scripts/eval_frontier.py).

The API client is injected as a fake ``complete(system, user) -> str`` so these run with no network
and no key. They lock the two frontier-specific behaviors: output cleanup and the retry loop. The
scoring pipeline itself is already covered by the FunctionTagger tests in test_eval_run.py.
"""

import pytest

from scripts.eval_frontier import _postprocess, make_frontier_tagger
from src.common.tags import NAME_CLOSE, NAME_OPEN

TAGGED = f"{NAME_OPEN}Sam{NAME_CLOSE} helped me revise."


def test_postprocess_strips_whitespace():
    assert _postprocess("  hello  \n") == "hello"


def test_postprocess_strips_plain_code_fence():
    assert _postprocess(f"```\n{TAGGED}\n```") == TAGGED


def test_postprocess_strips_fence_with_language_hint():
    assert _postprocess(f"```text\n{TAGGED}\n```") == TAGGED


def test_postprocess_leaves_tags_untouched():
    # No tag manipulation — byte-identical except the fence/whitespace.
    assert _postprocess(TAGGED) == TAGGED


def test_postprocess_does_not_drop_content_on_partial_fence():
    # Opening fence but NO matching closing fence: not a fully-fenced block, so leave it alone
    # rather than silently dropping the tagged content (finding 2 from review).
    raw = f"```\n{TAGGED}"
    assert _postprocess(raw) == raw


def test_postprocess_does_not_touch_content_sharing_the_fence_line():
    # Content on the same line as the opening fence must survive (first line isn't a bare fence),
    # so no unwrap happens and the text is returned as-is (finding 2 from review).
    raw = f"```{TAGGED}\n```"
    assert _postprocess(raw) == raw


def test_tagger_uses_shared_system_prompt_and_passage():
    seen = {}

    def fake_complete(system, user):
        seen["system"] = system
        seen["user"] = user
        return f"```\n{TAGGED}\n```"

    tagger = make_frontier_tagger(fake_complete, name="frontier-test")
    out = tagger.tag("Sam helped me revise.")

    assert out == TAGGED  # postprocessed
    assert tagger.name == "frontier-test"
    assert seen["user"] == "Sam helped me revise."
    assert "personal names" in seen["system"].lower()  # the locked SYSTEM_PROMPT was passed


def test_tagger_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("scripts.eval_frontier.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky(system, user):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return TAGGED

    tagger = make_frontier_tagger(flaky, retries=3)
    assert tagger.tag("x") == TAGGED
    assert calls["n"] == 3


def test_tagger_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("scripts.eval_frontier.time.sleep", lambda *_: None)

    def always_fail(system, user):
        raise ConnectionError("down")

    tagger = make_frontier_tagger(always_fail, retries=2)
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        tagger.tag("x")
