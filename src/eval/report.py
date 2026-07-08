"""Offline reporting: render base-vs-tuned markdown tables from saved eval reports (spec S3.5).

This module reads one or more **already-saved** JSON eval reports (written by
``src.eval.run``) and renders the markdown tables that go into ``docs/results.md`` — an
Overall table and a Per-category table, each metric shown **with a 95% bootstrap CI**, with
``integrity_violation_rate`` and ``consistency`` surfaced as first-class columns.

Everything here is **offline and model-free**:

- Point metrics come straight from each report's saved ``overall`` / ``by_category`` blocks
  (the harness compute is authoritative — see :func:`recompute_overall` for the tested
  guarantee that recomputing from per-item outputs reproduces those numbers).
- CIs are bootstrapped over the **per-item** results. New reports carry per-item
  ``tp/fp/fn``; **older reports that predate that field** are repaired offline by re-running
  :func:`src.eval.behavioral_checks.check` against the quarantined eval set (join by item
  ``id``). No model is ever loaded and no network is touched.

CLI::

    python -m src.eval.report base=outputs/eval_reports/base-*.json \\
                              tuned=outputs/eval_reports/tuned-*.json
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.common.schema import Example, read_jsonl
from src.eval import behavioral_checks as bc
from src.eval import metrics as M

# Display order for per-category rows (matches the eval set's category spread).
PREFERRED_CATEGORY_ORDER = [
    "person_vs_common",
    "person_vs_place",
    "person_vs_eponym",
    "first_name_only",
    "possessive",
    "third_party",
    "negative_trap",
    "adversarial",
    "easy",
    "real",
]

# Columns for the two tables (keys index the saved report dicts + MetricsCI fields).
OVERALL_COLUMNS = [
    "precision",
    "recall",
    "f5",
    "leakage_rate",
    "over_tag_rate",
    "integrity_violation_rate",
    "pass_rate",
    "consistency",
]
CATEGORY_COLUMNS = [
    "recall",
    "over_tag_rate",
    "integrity_violation_rate",
    "pass_rate",
    "consistency",
]
# Compact display headers (F5 reads better than f5; the rest are kept verbatim so the
# safety-critical metric names are unmistakable in the doc).
_HEADER = {"f5": "F5"}


@dataclass
class ItemRecord:
    """One reconstructed per-item result plus its category (for grouping)."""

    id: str
    category: str
    result: M.ItemResult


@dataclass
class LabeledReport:
    """A saved report bundled with its label, per-item records and bootstrap CIs."""

    label: str
    report: dict
    records: list[ItemRecord]
    overall_ci: M.MetricsCI
    category_ci: dict[str, M.MetricsCI]


# --- loading / reconstruction ----------------------------------------------------------
def load_report(path: str | Path) -> dict:
    """Load a saved JSON eval report."""
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _has_counts(o: dict) -> bool:
    return all(k in o for k in ("tp", "fp", "fn"))


def _item_from_dict(o: dict) -> M.ItemResult:
    return M.ItemResult(
        tp=int(o["tp"]),
        fp=int(o["fp"]),
        fn=int(o["fn"]),
        passed=bool(o["pass"]),
        leaked=bool(o["leaked"]),
        over_tagged=bool(o["over_tagged"]),
        integrity_violation=not bool(o["integrity_ok"]),
    )


def _load_eval_by_id(split: str) -> dict[str, Example]:
    p = Path(split)
    files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    by_id: dict[str, Example] = {}
    for f in files:
        for ex in read_jsonl(f):
            by_id[ex.id] = ex.validate()
    return by_id


def reconstruct_items(report: dict, *, eval_split: str = "eval/hardcases") -> list[ItemRecord]:
    """Rebuild per-item results from a saved report.

    Uses saved ``tp/fp/fn`` when present; otherwise recomputes them offline via
    :func:`behavioral_checks.check` against ``eval_split`` (join by item ``id``).
    """
    outputs = report.get("outputs") or []
    needs_gold = any(not _has_counts(o) for o in outputs)
    by_id = _load_eval_by_id(eval_split) if needs_gold else {}

    records: list[ItemRecord] = []
    for o in outputs:
        if _has_counts(o):
            records.append(ItemRecord(o["id"], o.get("category", "unknown"), _item_from_dict(o)))
            continue
        ex = by_id.get(o["id"])
        if ex is None:
            raise KeyError(
                f"report item {o['id']!r} lacks tp/fp/fn and was not found in eval split "
                f"{eval_split!r}; cannot recompute counts offline"
            )
        res = M.ItemResult.from_check_result(bc.check(ex, o["output"]))
        records.append(ItemRecord(o["id"], ex.category, res))
    return records


def group_by_category(records: list[ItemRecord]) -> dict[str, list[ItemRecord]]:
    buckets: dict[str, list[ItemRecord]] = defaultdict(list)
    for r in records:
        buckets[r.category].append(r)
    return dict(buckets)


def build_labeled_report(
    report: dict,
    *,
    label: str | None = None,
    eval_split: str = "eval/hardcases",
    n_resamples: int = 1000,
    seed: int = 20260707,
) -> LabeledReport:
    """Reconstruct per-item records and bootstrap the overall + per-category CIs."""
    records = reconstruct_items(report, eval_split=eval_split)
    overall_ci = M.bootstrap_cis([r.result for r in records], n_resamples=n_resamples, seed=seed)
    category_ci: dict[str, M.MetricsCI] = {}
    for cat, recs in group_by_category(records).items():
        category_ci[cat] = M.bootstrap_cis(
            [r.result for r in recs], n_resamples=n_resamples, seed=seed
        )
    return LabeledReport(
        label=label or str(report.get("label", "model")),
        report=report,
        records=records,
        overall_ci=overall_ci,
        category_ci=category_ci,
    )


# --- invariants (would have caught the hand-transcribed doc bug) ------------------------
def recompute_overall(records: list[ItemRecord]) -> dict[str, float]:
    """Aggregate CI-metric point values from per-item records (no consistency).

    Reproduces a report's saved ``overall`` (minus consistency); see the recompute test.
    """
    return M.point_metrics_from_items([r.result for r in records])


def recall_pass_invariant_violations(records: list[ItemRecord]) -> list[str]:
    """Categories that violate the recall/pass invariant.

    A category whose items **all** contain at least one gold name (``tp + fn >= 1``) cannot
    have ``recall == 0`` while ``pass_rate > 0``: zero recall means every item missed a gold
    name (leaked), so none can pass. Correctly computed metrics never violate this — the Day-3
    doc's ``easy`` row (base recall 0.0 but pass 0.667) did, because it was hand-typed. Any
    non-empty return here flags an impossible (corrupted / mis-transcribed) report.
    """
    violations: list[str] = []
    for cat, recs in group_by_category(records).items():
        if not recs:
            continue
        if all((r.result.tp + r.result.fn) >= 1 for r in recs):
            total_tp = sum(r.result.tp for r in recs)
            passes = sum(1 for r in recs if r.result.passed)
            if total_tp == 0 and passes > 0:
                violations.append(cat)
    return violations


# --- rendering -------------------------------------------------------------------------
def _fmt_point(value: float | None, ndigits: int = 3) -> str:
    return "–" if value is None else f"{value:.{ndigits}f}"


def _cell(point: float | None, iv: M.Interval | None) -> str:
    if point is None:
        return "–"
    if iv is None:
        return _fmt_point(point)
    return f"{_fmt_point(point)} {iv.fmt(2)}"


def _header_cells(columns: list[str]) -> str:
    return " | ".join(_HEADER.get(c, c) for c in columns)


def _ci_for(ci: M.MetricsCI, key: str) -> M.Interval | None:
    return ci.interval(key) if key in M.CI_METRICS else None


def render_overall_table(labeled: list[LabeledReport]) -> str:
    header = "| model | n | " + _header_cells(OVERALL_COLUMNS) + " |"
    sep = "|" + "---|" * (len(OVERALL_COLUMNS) + 2)
    lines = [header, sep]
    for lr in labeled:
        overall = lr.report.get("overall", {})
        n = overall.get("n", lr.report.get("n", ""))
        cells = [lr.label, str(n)]
        for c in OVERALL_COLUMNS:
            cells.append(_cell(overall.get(c), _ci_for(lr.overall_ci, c)))
        lines.append("| " + " | ".join(cells) + " |")

    # Honest win+cost framing: a point-delta row when exactly two reports are compared.
    if len(labeled) == 2:
        a, b = labeled
        oa, ob = a.report.get("overall", {}), b.report.get("overall", {})
        cells = [f"Δ ({b.label}−{a.label})", ""]
        for c in OVERALL_COLUMNS:
            va, vb = oa.get(c), ob.get(c)
            cells.append("–" if va is None or vb is None else f"{vb - va:+.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_category_table(labeled: list[LabeledReport]) -> str:
    header = "| category | n | model | " + _header_cells(CATEGORY_COLUMNS) + " |"
    sep = "|" + "---|" * (len(CATEGORY_COLUMNS) + 3)
    lines = [header, sep]

    cats_present = {c for lr in labeled for c in lr.report.get("by_category", {})}
    ordered = [c for c in PREFERRED_CATEGORY_ORDER if c in cats_present]
    ordered += sorted(c for c in cats_present if c not in PREFERRED_CATEGORY_ORDER)

    for cat in ordered:
        n_values = (
            lr.report["by_category"][cat].get("n")
            for lr in labeled
            if cat in lr.report.get("by_category", {})
        )
        n = next(n_values, "")
        for lr in labeled:
            per_cat = lr.report.get("by_category", {})
            if cat not in per_cat:
                continue
            row = per_cat[cat]
            ci = lr.category_ci.get(cat)
            cells = [cat, str(n), lr.label]
            for c in CATEGORY_COLUMNS:
                cells.append(_cell(row.get(c), _ci_for(ci, c) if ci else None))
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_tables(labeled: list[LabeledReport]) -> str:
    return (
        "## Overall\n\n"
        + render_overall_table(labeled)
        + "\n\n## Per-category (base → tuned within each category)\n\n"
        + render_category_table(labeled)
        + "\n"
    )


# --- CLI -------------------------------------------------------------------------------
def _parse_report_arg(tok: str) -> tuple[str | None, str]:
    """Parse ``LABEL=PATH`` or bare ``PATH`` (glob allowed)."""
    if "=" in tok:
        label, path = tok.split("=", 1)
        return label, path
    return None, tok


def _resolve_path(pattern: str) -> str:
    if Path(pattern).exists():
        return pattern
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no report file matches {pattern!r}")
    return matches[-1]  # newest by name (timestamps sort lexicographically)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render base-vs-tuned markdown tables (with CIs).")
    ap.add_argument("reports", nargs="+", help="report files as PATH or LABEL=PATH (globs ok)")
    ap.add_argument("--eval-split", default="eval/hardcases", help="eval set for offline recompute")
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    args = ap.parse_args()

    labeled: list[LabeledReport] = []
    for tok in args.reports:
        label, pattern = _parse_report_arg(tok)
        path = _resolve_path(pattern)
        report = load_report(path)
        labeled.append(
            build_labeled_report(
                report,
                label=label,
                eval_split=args.eval_split,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
        )

    for lr in labeled:
        bad = recall_pass_invariant_violations(lr.records)
        if bad:
            raise SystemExit(f"[{lr.label}] impossible recall/pass in categories: {bad}")

    md = render_tables(labeled)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
