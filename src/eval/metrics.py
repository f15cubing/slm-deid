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

import math
import random
from collections import defaultdict
from collections.abc import Sequence
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


def _check_all(examples: list[Example], outputs: list[str]) -> list[CheckResult]:
    if len(examples) != len(outputs):
        raise ValueError("examples and outputs must be the same length")
    return [check(ex, out) for ex, out in zip(examples, outputs)]


def compute(examples: list[Example], outputs: list[str]) -> Metrics:
    """Compute aggregate metrics over paired (example, model output) lists."""
    results: list[CheckResult] = _check_all(examples, outputs)

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


# --- Bootstrap confidence intervals (spec S3.5) ----------------------------------------
# The point metrics above are single numbers on a small held-out set (n=51 overall; a few
# per-category cells are n=3), so they are noisy. A 95% percentile bootstrap over the
# *per-item* check results gives an honest uncertainty band without any model calls.

# The rate metrics are per-item means; precision/recall/F5 are recomputed from resampled
# per-item (tp, fp, fn). Consistency is a paraphrase-group statistic (not a per-item mean),
# so it is reported as a point value without a bootstrap CI.
CI_METRICS = (
    "precision",
    "recall",
    "f5",
    "leakage_rate",
    "over_tag_rate",
    "integrity_violation_rate",
    "pass_rate",
)


@dataclass(frozen=True)
class ItemResult:
    """The minimal per-item signal the bootstrap resamples over.

    Built either from a :class:`CheckResult` (fresh eval) or from a saved per-item report
    entry (offline reporting), so CIs can be computed without re-running any model.
    """

    tp: int
    fp: int
    fn: int
    passed: bool
    leaked: bool
    over_tagged: bool
    integrity_violation: bool  # == (not integrity_ok)

    @classmethod
    def from_check_result(cls, r: CheckResult) -> ItemResult:
        return cls(
            tp=r.tp,
            fp=r.fp,
            fn=r.fn,
            passed=r.passed,
            leaked=r.leaked,
            over_tagged=r.over_tagged,
            integrity_violation=not r.integrity_ok,
        )


@dataclass(frozen=True)
class Interval:
    """A closed [low, high] confidence interval."""

    low: float
    high: float

    def rounded(self, ndigits: int = 4) -> Interval:
        return Interval(round(self.low, ndigits), round(self.high, ndigits))

    def fmt(self, ndigits: int = 2) -> str:
        return f"[{self.low:.{ndigits}f}, {self.high:.{ndigits}f}]"


@dataclass(frozen=True)
class MetricsCI:
    """95% bootstrap CIs, one :class:`Interval` per metric in :data:`CI_METRICS`."""

    precision: Interval
    recall: Interval
    f5: Interval
    leakage_rate: Interval
    over_tag_rate: Interval
    integrity_violation_rate: Interval
    pass_rate: Interval

    def interval(self, metric: str) -> Interval:
        return getattr(self, metric)

    def as_dict(self, ndigits: int = 4) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for m in CI_METRICS:
            iv = getattr(self, m).rounded(ndigits)
            out[m] = (iv.low, iv.high)
        return out


def item_results(examples: list[Example], outputs: list[str]) -> list[ItemResult]:
    """Per-item results for the bootstrap (same check as :func:`compute`)."""
    return [ItemResult.from_check_result(r) for r in _check_all(examples, outputs)]


def point_metrics_from_items(items: Sequence[ItemResult]) -> dict[str, float]:
    """Aggregate the :data:`CI_METRICS` from per-item results (no consistency).

    This mirrors :func:`compute` exactly, so recomputing from saved per-item outputs
    reproduces a report's saved ``overall`` values (a tested invariant).
    """
    n = len(items)
    tp = sum(i.tp for i in items)
    fp = sum(i.fp for i in items)
    fn = sum(i.fn for i in items)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f5": fbeta(precision, recall, 5.0),
        "leakage_rate": _safe_div(sum(1 for i in items if i.leaked), n),
        "over_tag_rate": _safe_div(sum(1 for i in items if i.over_tagged), n),
        "integrity_violation_rate": _safe_div(sum(1 for i in items if i.integrity_violation), n),
        "pass_rate": _safe_div(sum(1 for i in items if i.passed), n),
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default method); ``q`` in [0, 1]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(lo)]
    frac = pos - lo
    return sorted_vals[int(lo)] * (1 - frac) + sorted_vals[int(hi)] * frac


def bootstrap_cis(
    items: Sequence[ItemResult],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260707,
) -> MetricsCI:
    """Percentile bootstrap CIs over per-item results.

    Determinism: the resampling uses a ``random.Random(seed)`` seeded once, so repeated
    calls with the same ``items`` and ``seed`` return byte-identical intervals.
    """
    n = len(items)
    if n == 0:
        zero = Interval(0.0, 0.0)
        return MetricsCI(**{m: zero for m in CI_METRICS})

    rng = random.Random(seed)
    population = list(range(n))
    dists: dict[str, list[float]] = {m: [] for m in CI_METRICS}
    for _ in range(n_resamples):
        sample = [items[i] for i in rng.choices(population, k=n)]
        stats = point_metrics_from_items(sample)
        for m in CI_METRICS:
            dists[m].append(stats[m])

    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, Interval] = {}
    for m in CI_METRICS:
        ordered = sorted(dists[m])
        intervals[m] = Interval(_percentile(ordered, alpha), _percentile(ordered, 1.0 - alpha))
    return MetricsCI(**intervals)


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
        cells = [label, str(m.n)] + [("-" if row[c] is None else f"{row[c]}") for c in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
