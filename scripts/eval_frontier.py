"""Evaluate a FRONTIER API model on the name-judgment task (reference ceiling, "just to see").

The frontier model is given the **same** system prompt (`prompts.SYSTEM_PROMPT`) and the same
passage the small models get, and its output is scored by the **same** behavioral checks and metrics
(`src.eval.run.evaluate`). It is wrapped in a `FunctionTagger`, so no eval-harness core changes are
needed — this is a thin composition of existing pure pieces.

This is a REFERENCE, not an apples-to-apples contestant: the frontier model is far larger than the
1.7B target. Read it as a ceiling. It is scored against the quarantined eval sets' **independent,
hand-built gold labels** — never against its own labels — so there is no circularity.

Runs on CPU (pure API); no GPU needed. Needs `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for a gateway).

    OPENAI_API_KEY=... python -m scripts.eval_frontier --split eval/hardcases --model gpt-4o
    # gateway (set OPENAI_BASE_URL + a model the gateway accepts; note some newer/reasoning models
    # reject a non-default temperature or need max_completion_tokens — that lives in
    # build_openai_complete / judge.py, out of scope here):
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://gateway/... \
        python -m scripts.eval_frontier --split eval/adversarial --model gpt-4o
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Callable

from src.common import prompts
from src.eval.judge import build_openai_complete
from src.eval.run import _load_examples, compare, evaluate, write_report
from src.infer import FunctionTagger

# A bare code-fence line: ``` optionally followed by a language hint and nothing else.
_BARE_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*$")


def _postprocess(raw: str) -> str:
    """Strip whitespace, and — for the frontier reference ONLY — a fully code-fenced wrapper.

    Deliberate asymmetry, documented: the small **contestants** are scored on plain ``.strip()``
    (``HFTagger.tag``), with no fence forgiveness. The frontier is a **ceiling reference**, not an
    apples-to-apples contestant, and frontier chat models routinely wrap output in a Markdown fence
    despite "Output ONLY the tagged text". Rather than let that formatting tic tank its integrity
    score, we unwrap a fence — but only a *fully* fenced block: the first line is a bare fence
    (``` optionally + a language hint, nothing else) AND the last line is a bare ```. That guard
    means we never drop real content that shares a line with a fence, nor touch a passage that
    merely starts with ``` without a matching close. Tag markers and every other character stay
    byte-identical, so the integrity check (output-minus-tags == input) is preserved.
    """
    text = raw.strip()
    lines = text.split("\n")
    if len(lines) >= 2 and _BARE_FENCE.match(lines[0]) and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1]).strip()
    return text


def make_frontier_tagger(
    complete: Callable[[str, str], str],
    name: str = "gpt-frontier",
    retries: int = 3,
) -> FunctionTagger:
    """Wrap a ``complete(system, user) -> str`` API client as a Tagger.

    ``complete`` is injected so this is unit-testable offline with a fake (no network). Each passage
    is one call with the shared system prompt; transient errors are retried with linear backoff so a
    single blip doesn't sink a whole eval run.
    """

    def tag(passage: str) -> str:
        last: Exception | None = None
        for attempt in range(retries):
            try:
                return _postprocess(complete(prompts.SYSTEM_PROMPT, passage))
            except Exception as e:  # noqa: BLE001 — surface after retries
                last = e
                if attempt < retries - 1:  # no backoff after the final attempt
                    time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"frontier call failed after {retries} attempts: {last}")

    return FunctionTagger(tag, name=name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a frontier API model (reference ceiling).")
    ap.add_argument("--split", default="eval/hardcases", help="JSONL file or dir of eval examples")
    ap.add_argument(
        "--model",
        default=None,
        help="OpenAI(-compatible) model id (default: $TEACHER_MODEL or gpt-4o)",
    )
    ap.add_argument("--label", default=None, help="report label (default: frontier-<model>)")
    ap.add_argument("--report-dir", default="data/eval_reports")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it (and OPENAI_BASE_URL for a gateway) first."
        )

    model = args.model or os.environ.get("TEACHER_MODEL", "gpt-4o")
    label = args.label or f"frontier-{model}"
    complete = build_openai_complete(model=model)
    tagger = make_frontier_tagger(complete, name=label)

    examples = _load_examples(args.split)
    print(f"loaded {len(examples)} eval examples from {args.split}; frontier model = {model}")
    report = evaluate(tagger, examples, label=label)
    path = write_report(report, args.report_dir)
    print(f"[{label}] wrote {path}")
    print("\n" + compare([report]))


if __name__ == "__main__":
    main()
