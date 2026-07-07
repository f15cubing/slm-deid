"""Aggregate metrics for the eval harness (Day 2, spec S2.3).

Everything is entity-level (a "span" is an entity mention here). We report:

- **precision / recall / F5** on NAME spans (β=5 — recall-weighted, matching the Learning Agency
  Lab scoring and SPOV 2: leakage is the safety-critical error).
- **leakage_rate** — fraction of items with ≥1 missed name.
- **over_tag_rate** — fraction of items with ≥1 false tag.
- **integrity_violation_rate** — fraction of items where output-minus-tags != input.
- **pass_rate** — fraction of items that fully pass (well-formed, integrity, no leak, no over-tag).
- **consistency** — fraction of paraphrase groups whose items all share the same pass/fail
  verdict (reliability across rewordings; SPOV: consistency is the point).

Also provides per-category breakdown and a markdown table renderer for the base-vs-tuned report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.common.schema import Example
from src.eval.behavioral_checks import CheckResult, check


def fbeta(precision: float, recall: float, beta: float = 5.0) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


@dataclass
class Metrics:
    n: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f5: float
    leakage_rate: float
    over_tag_rate: float
    integrity_violation_rate: float
    pass_rate: float
    consistency: float | None  # None when there are no paraphrase groups

    def as_row(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f5": round(self.f5, 4),
            "leakage_rate": round(self.leakage_rate, 4),
            "over_tag_rate": round(self.over_tag_rate, 4),
            "integrity_violation_rate": round(self.integrity_violation_rate, 4),
            "pass_rate": round(self.pass_rate, 4),
            "consistency": (None if self.consistency is None else round(self.consistency, 4)),
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute(examples: list[Example], outputs: list[str]) -> Metrics:
    """Compute aggregate metrics over paired (example, model output) lists."""
    if len(examples) != len(outputs):
        raise ValueError("examples and outputs must be the same length")

    results: list[CheckResult] = [check(ex, out) for ex, out in zip(examples, outputs)]

    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    n = len(results)
    leaks = sum(1 for r in results if r.leaked)
    overs = sum(1 for r in results if r.over_tagged)
    integ = sum(1 for r in results if not r.integrity_ok)
    passes = sum(1 for r in results if r.passed)

    # Consistency: group by paraphrase_group; a group is "consistent" iff all its items
    # share the same pass/fail verdict. Items without a group are ignored.
    groups: dict[str, list[bool]] = defaultdict(list)
    for ex, r in zip(examples, results):
        if ex.paraphrase_group:
            groups[ex.paraphrase_group].append(r.passed)
    consistency: float | None
    if groups:
        consistent = sum(1 for verdicts in groups.values() if len(set(verdicts)) == 1)
        consistency = _safe_div(consistent, len(groups))
    else:
        consistency = None

    return Metrics(
        n=n,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f5=fbeta(precision, recall, 5.0),
        leakage_rate=_safe_div(leaks, n),
        over_tag_rate=_safe_div(overs, n),
        integrity_violation_rate=_safe_div(integ, n),
        pass_rate=_safe_div(passes, n),
        consistency=consistency,
    )


def by_category(examples: list[Example], outputs: list[str]) -> dict[str, Metrics]:
    """Per-category metrics (the crux — where the base is expected to wobble)."""
    buckets: dict[str, tuple[list[Example], list[str]]] = defaultdict(lambda: ([], []))
    for ex, out in zip(examples, outputs):
        buckets[ex.category][0].append(ex)
        buckets[ex.category][1].append(out)
    return {cat: compute(exs, outs) for cat, (exs, outs) in buckets.items()}


_COLUMNS = [
    "precision",
    "recall",
    "f5",
    "leakage_rate",
    "over_tag_rate",
    "integrity_violation_rate",
    "pass_rate",
    "consistency",
]


def markdown_table(named_metrics: dict[str, Metrics]) -> str:
    """Render {label: Metrics} as a markdown table (e.g. {'base': ..., 'tuned': ...})."""
    header = "| model | n | " + " | ".join(_COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COLUMNS) + 2)
    lines = [header, sep]
    for label, m in named_metrics.items():
        row = m.as_row()
        cells = [label, str(m.n)] + [
            ("-" if row[c] is None else f"{row[c]}") for c in _COLUMNS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
