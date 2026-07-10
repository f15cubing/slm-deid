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
    # gateway:
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://gateway/... \
        python -m scripts.eval_frontier --split eval/adversarial --model gpt551
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable

from src.common import prompts
from src.eval.judge import build_openai_complete
from src.eval.run import _load_examples, compare, evaluate, write_report
from src.infer import FunctionTagger


def _postprocess(raw: str) -> str:
    """Minimal cleanup mirroring HFTagger's ``.strip()``.

    Frontier models sometimes wrap output in a Markdown code fence despite "Output ONLY the tagged
    text"; strip a single leading/trailing fence so integrity scoring isn't lost to formatting. We
    do NOT touch the tag markers or any other character — that would bias the fairness of the eval.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening ``` (and any language hint on that line)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
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
