"""Base-vs-tuned evaluation scaffold (Day 2, spec S2.5).

Core (`evaluate`, `compare`) is pure and unit-tested with a FunctionTagger. The CLI loads real
models via ``src.infer.load_hf_tagger`` (GPU/Colab) and writes a JSON report + a markdown table.

    # on Colab, after training:
    python -m src.eval.run --split eval/hardcases --model base
    python -m src.eval.run --split eval/hardcases --model outputs/sft-v1 --label tuned
    python -m src.eval.run --split eval/hardcases --compare base outputs/sft-v1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.common.schema import Example, read_jsonl
from src.eval import metrics as M
from src.eval.judge import LLMJudge
from src.infer import Tagger, load_hf_tagger, tag_all


@dataclass
class EvalReport:
    label: str
    n: int
    overall: M.Metrics
    by_category: dict[str, M.Metrics]
    # per-item dicts: {id, category, input, output, pass, leaked, over_tagged, integrity_ok}
    outputs: list[dict] = field(default_factory=list)
    judge_summary: dict | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "overall": self.overall.as_row(),
            "by_category": {c: m.as_row() for c, m in self.by_category.items()},
            "judge_summary": self.judge_summary,
            "outputs": self.outputs,
        }


def _load_examples(split: str) -> list[Example]:
    p = Path(split)
    files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    examples: list[Example] = []
    for f in files:
        examples.extend(read_jsonl(f))
    return [ex.validate() for ex in examples]


def evaluate(
    tagger: Tagger,
    examples: list[Example],
    label: str | None = None,
    judge: LLMJudge | None = None,
) -> EvalReport:
    """Run ``tagger`` over ``examples`` and compute the full report."""
    label = label or getattr(tagger, "name", "model")
    inputs = [ex.input for ex in examples]
    outputs = tag_all(tagger, inputs)

    from src.eval.behavioral_checks import check

    per_item = []
    for ex, out in zip(examples, outputs):
        r = check(ex, out)
        per_item.append(
            {
                "id": ex.id,
                "category": ex.category,
                "input": ex.input,
                "output": out,
                "pass": r.passed,
                "leaked": r.leaked,
                "over_tagged": r.over_tagged,
                "integrity_ok": r.integrity_ok,
            }
        )

    judge_summary = None
    if judge is not None:
        scores = [judge.score(ex, out) for ex, out in zip(examples, outputs)]
        n = len(scores) or 1
        judge_summary = {
            dim: round(sum(getattr(s, dim) for s in scores) / n, 3)
            for dim in ("spec_adherence", "robustness", "task_quality", "consistency")
        }
        judge_summary["mean_total"] = round(sum(s.total for s in scores) / n, 3)
        judge_summary["disagreement_rate"] = round(
            sum(1 for s in scores if s.disagreement) / n, 3
        )

    return EvalReport(
        label=label,
        n=len(examples),
        overall=M.compute(examples, outputs),
        by_category=M.by_category(examples, outputs),
        outputs=per_item,
        judge_summary=judge_summary,
    )


def compare(reports: list[EvalReport]) -> str:
    """Markdown overall table across reports (base vs tuned vs ...)."""
    return M.markdown_table({r.label: r.overall for r in reports})


def write_report(report: EvalReport, report_dir: str = "data/eval_reports") -> Path:
    d = Path(report_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{report.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate name-tagging on the hard cases.")
    ap.add_argument("--split", default="eval/hardcases", help="JSONL file or dir of eval examples")
    ap.add_argument("--model", default="base", help="'base' or a path to a LoRA adapter")
    ap.add_argument("--label", default=None, help="report label (default: base/tuned)")
    ap.add_argument("--compare", nargs="+", default=None, help="two+ model ids to compare")
    ap.add_argument("--report-dir", default="data/eval_reports")
    args = ap.parse_args()

    examples = _load_examples(args.split)
    print(f"loaded {len(examples)} eval examples from {args.split}")

    model_ids = args.compare if args.compare else [args.model]
    reports: list[EvalReport] = []
    for mid in model_ids:
        tagger = load_hf_tagger() if mid == "base" else load_hf_tagger(adapter=mid)
        label = args.label if (args.label and len(model_ids) == 1) else tagger.name
        report = evaluate(tagger, examples, label=label)
        path = write_report(report, args.report_dir)
        print(f"[{label}] wrote {path}")
        reports.append(report)

    print("\n" + compare(reports))


if __name__ == "__main__":
    main()
