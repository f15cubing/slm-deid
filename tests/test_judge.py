"""Day 2, spec S2.4 — LLM-as-judge parsing + behavioral cross-check (mocked client)."""

import json

from src.common import tags
from src.common.schema import Example, Span
from src.eval.judge import JudgeScore, LLMJudge, parse_judge_json


def _ex():
    raw = "Sarah wrote this."
    tgt = f"{tags.wrap('Sarah')} wrote this."
    return Example(id="j", input=raw, target=tgt, spans=[Span(0, 5, "Sarah", True)]).validate()


def _fake(payload: dict):
    return lambda system, user: json.dumps(payload)


def test_parse_tolerates_prose_and_fences():
    raw = "Sure!\n```json\n{\"spec_adherence\": 2, \"robustness\": 1}\n```\nDone."
    d = parse_judge_json(raw)
    assert d["spec_adherence"] == 2 and d["robustness"] == 1


def test_perfect_output_agrees_with_behavioral():
    ex = _ex()
    judge = LLMJudge(_fake({
        "spec_adherence": 2, "robustness": 2, "task_quality": 2,
        "consistency": 2, "rationale": "ok",
    }))
    s = judge.score(ex, ex.target)
    assert isinstance(s, JudgeScore)
    assert s.total == 8
    assert s.behavioral_pass is True
    assert s.disagreement is False


def test_disagreement_flagged_when_judge_too_generous():
    ex = _ex()
    # Model leaked the name (integrity ok but under-tagged) -> behavioral FAIL,
    # but the judge wrongly says spec_adherence=2 -> disagreement must be flagged.
    judge = LLMJudge(_fake(
        {"spec_adherence": 2, "robustness": 2, "task_quality": 2, "consistency": 2}
    ))
    s = judge.score(ex, ex.input)  # nothing tagged
    assert s.behavioral_pass is False
    assert s.disagreement is True


def test_scores_clamped_to_0_2():
    ex = _ex()
    judge = LLMJudge(_fake({
        "spec_adherence": 5, "robustness": -3, "task_quality": "2", "consistency": None,
    }))
    s = judge.score(ex, ex.target)
    assert s.spec_adherence == 2
    assert s.robustness == 0
    assert s.task_quality == 2
    assert s.consistency == 0
