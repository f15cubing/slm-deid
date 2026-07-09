"""Ad-hoc held-out eval: base vs a tuned adapter on a real-text slice, with a raised generation
cap so long passages aren't truncated (the CLI's HFTagger defaults to max_new_tokens=256, which
would cut real CRAPII essays mid-output and fail integrity for reasons unrelated to name judgment).

Reuses the production eval path (evaluate / write_report / compare) — only the tagger's
max_new_tokens differs. Not a committed entry point; a probe for external validity.

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -u scripts/eval_heldout.py \
        --split data/_heldout/crapii_heldout.jsonl --tuned outputs/sft-v3-mps --max-new-tokens 768

Pass ``--project`` to add a third row where the tuned model's output is passed through the
pipeline's tag-by-offset projection (``pipeline.project.ProjectingTagger``) before scoring. On
messy real text this is expected to collapse ``integrity_violation_rate`` toward 0 (integrity
holds by construction) and recover the recall/precision that strict byte-identity scoring zeroes
out on whitespace drift — quantifying the ``pipeline/`` fix on the same eval path.
"""

from __future__ import annotations

import argparse

from src.eval.run import _load_examples, compare, evaluate, write_report
from src.infer import load_hf_tagger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--tuned", required=True, help="path to the tuned LoRA adapter")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--report-dir", default="outputs/eval_reports_heldout")
    ap.add_argument(
        "--project",
        action="store_true",
        help="also score the tuned model with tag-by-offset projection (integrity-safe output)",
    )
    args = ap.parse_args()

    examples = _load_examples(args.split)
    print(f"loaded {len(examples)} held-out examples from {args.split}")

    reports = []
    for mid in ("base", args.tuned):
        adapter = None if mid == "base" else mid
        tagger = load_hf_tagger(adapter=adapter, max_new_tokens=args.max_new_tokens)
        report = evaluate(tagger, examples, label=("base" if adapter is None else "tuned"))
        path = write_report(report, args.report_dir)
        print(f"[{report.label}] wrote {path}")
        reports.append(report)

        if args.project and adapter is not None:
            from pipeline.project import ProjectingTagger

            proj = ProjectingTagger(tagger, name="tuned+proj")
            report = evaluate(proj, examples, label="tuned+proj")
            path = write_report(report, args.report_dir)
            print(f"[{report.label}] wrote {path}")
            reports.append(report)

    print("\n" + compare(reports))


if __name__ == "__main__":
    main()
