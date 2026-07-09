"""LLM-as-judge over the 4 rubric dimensions (Day 2, spec S2.4).

Scores each model output 0-2 on: spec_adherence, robustness, task_quality, consistency. The judge
is reference-based (it sees the gold target), deterministic (temp=0), and returns strict JSON.

Design keeps the network dependency at the edge: :class:`LLMJudge` takes a ``complete`` callable
``(system, user) -> str`` so it is trivially mockable in tests. Real clients are built lazily by
:func:`build_openai_complete` / :func:`build_anthropic_complete` (no key wired in the repo).

The judge does NOT override the deterministic behavioral checks — it complements them. When the
judge's spec_adherence disagrees with the behavioral PASS/FAIL, the disagreement is flagged so we
can audit the judge rather than trust it blindly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from src.common import tags
from src.common.schema import Example
from src.eval.behavioral_checks import check

Complete = Callable[[str, str], str]  # (system, user) -> raw text

DIMENSIONS = ("spec_adherence", "robustness", "task_quality", "consistency")

JUDGE_SYSTEM = (
    "You are a strict evaluator for a name-tagging de-identification task. The model must return "
    "the input unchanged except that every real person's name is wrapped in "
    f"{tags.NAME_OPEN}…{tags.NAME_CLOSE}, with no other text altered and no non-person span "
    "tagged. Score 0-2 on each dimension (0=fails, 1=partial, 2=perfect) and return ONLY compact "
    "JSON with keys: spec_adherence, robustness, task_quality, consistency, rationale."
)


def _build_user_prompt(example: Example, output: str) -> str:
    return (
        f"INPUT:\n{example.input}\n\n"
        f"GOLD (correct tagging):\n{example.target}\n\n"
        f"MODEL OUTPUT:\n{output}\n\n"
        "Score the MODEL OUTPUT against GOLD. spec_adherence = only-and-all person names tagged, "
        "text otherwise byte-identical. robustness = would this judgment hold under rewordings. "
        "task_quality = overall correctness. consistency = internal coherence of the tagging. "
        "Return ONLY the JSON object."
    )


@dataclass
class JudgeScore:
    spec_adherence: int
    robustness: int
    task_quality: int
    consistency: int
    rationale: str = ""
    behavioral_pass: bool | None = None
    # True when judge's spec-adherence verdict disagrees with the deterministic behavioral check.
    disagreement: bool = False

    @property
    def total(self) -> int:
        return self.spec_adherence + self.robustness + self.task_quality + self.consistency

    def as_dict(self) -> dict:
        return {
            "spec_adherence": self.spec_adherence,
            "robustness": self.robustness,
            "task_quality": self.task_quality,
            "consistency": self.consistency,
            "total": self.total,
            "rationale": self.rationale,
            "behavioral_pass": self.behavioral_pass,
            "disagreement": self.disagreement,
        }


def _clamp02(v) -> int:
    try:
        i = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(2, i))


def parse_judge_json(raw: str) -> dict:
    """Extract the JSON object from a judge response (tolerates surrounding prose/fences)."""
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in judge output: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


class LLMJudge:
    def __init__(self, complete: Complete, name: str = "llm-judge"):
        self._complete = complete
        self.name = name

    def score(self, example: Example, output: str) -> JudgeScore:
        raw = self._complete(JUDGE_SYSTEM, _build_user_prompt(example, output))
        data = parse_judge_json(raw)
        score = JudgeScore(
            spec_adherence=_clamp02(data.get("spec_adherence")),
            robustness=_clamp02(data.get("robustness")),
            task_quality=_clamp02(data.get("task_quality")),
            consistency=_clamp02(data.get("consistency")),
            rationale=str(data.get("rationale", "")),
        )
        # Cross-check against the deterministic behavioral verdict.
        behavioral_pass = check(example, output).passed
        score.behavioral_pass = behavioral_pass
        judge_says_perfect = score.spec_adherence == 2
        score.disagreement = judge_says_perfect != behavioral_pass
        return score


# --- lazy real clients (no key wired in the repo) --------------------------------------
def build_openai_complete(model: str | None = None, temperature: float = 0.0) -> Complete:
    """OpenAI-compatible teacher/judge client.

    Works against OpenAI directly OR any OpenAI-compatible gateway (e.g. the TrueFoundry LLM
    Gateway): the ``OpenAI`` SDK reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` from the
    environment, so pointing at a gateway is pure env config. Only the model id must change —
    gateways namespace it (``openai-main/gpt-4o``) — so default it from ``TEACHER_MODEL`` when the
    caller doesn't pass one explicitly.
    """
    import os

    from openai import OpenAI  # lazy

    model = model or os.environ.get("TEACHER_MODEL", "gpt-4o")
    client = OpenAI()  # honors OPENAI_API_KEY + OPENAI_BASE_URL

    def complete(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    return complete


def build_anthropic_complete(
    model: str = "claude-3-5-sonnet-latest", temperature: float = 0.0, max_tokens: int = 512
) -> Complete:
    from anthropic import Anthropic  # lazy

    client = Anthropic()

    def complete(system: str, user: str) -> str:
        resp = client.messages.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

    return complete
