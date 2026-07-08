"""Spec S3.5 — bootstrap confidence intervals (metrics.bootstrap_cis).

Covers determinism under a fixed seed (the hard requirement), valid bounds/ordering, the
degenerate point-mass case, and feeding the bootstrap from real per-item check results.
"""

from src.common import tags
from src.common.schema import Example, Span
from src.eval import metrics as M


def _items(n_pass: int, n_leak: int, n_over: int) -> list[M.ItemResult]:
    """Build a per-item list with known pass/leak/over-tag signals."""
    items: list[M.ItemResult] = []
    for _ in range(n_pass):  # perfect item: 1 TP, passes
        items.append(
            M.ItemResult(
                1, 0, 0, passed=True, leaked=False, over_tagged=False, integrity_violation=False
            )
        )
    for _ in range(n_leak):  # missed a name: 1 FN, leaked
        items.append(
            M.ItemResult(
                0, 0, 1, passed=False, leaked=True, over_tagged=False, integrity_violation=False
            )
        )
    for _ in range(n_over):  # tagged a non-name: 1 FP, over-tagged
        items.append(
            M.ItemResult(
                0, 1, 0, passed=False, leaked=False, over_tagged=True, integrity_violation=False
            )
        )
    return items


def test_bootstrap_is_deterministic_under_fixed_seed():
    items = _items(6, 3, 3)
    a = M.bootstrap_cis(items, n_resamples=500, seed=123)
    b = M.bootstrap_cis(items, n_resamples=500, seed=123)
    assert a == b
    assert a.as_dict() == b.as_dict()


def test_bootstrap_bounds_and_ordering():
    ci = M.bootstrap_cis(_items(6, 3, 3), n_resamples=500, seed=7)
    for m in M.CI_METRICS:
        iv = ci.interval(m)
        assert 0.0 <= iv.low <= iv.high <= 1.0


def test_bootstrap_degenerate_all_pass_is_point_mass():
    # Every item is perfect -> every resample is perfect -> zero-width intervals.
    ci = M.bootstrap_cis(_items(10, 0, 0), n_resamples=200, seed=0)
    assert (ci.pass_rate.low, ci.pass_rate.high) == (1.0, 1.0)
    assert (ci.recall.low, ci.recall.high) == (1.0, 1.0)
    assert (ci.precision.low, ci.precision.high) == (1.0, 1.0)
    assert (ci.leakage_rate.low, ci.leakage_rate.high) == (0.0, 0.0)


def test_bootstrap_empty_items_is_safe():
    ci = M.bootstrap_cis([], n_resamples=50, seed=0)
    for m in M.CI_METRICS:
        assert ci.interval(m) == M.Interval(0.0, 0.0)


def test_item_results_feed_bootstrap_deterministically():
    # Exercise the real check() path: one pass, one leak.
    raw = "Ada coded."
    tgt = f"{tags.wrap('Ada')} coded."
    ex = Example(id="1", input=raw, target=tgt, spans=[Span(0, 3, "Ada", True)]).validate()
    items = M.item_results([ex, ex], [tgt, raw])  # tgt passes, raw leaks "Ada"
    assert [i.passed for i in items] == [True, False]
    a = M.bootstrap_cis(items, n_resamples=200, seed=5)
    b = M.bootstrap_cis(items, n_resamples=200, seed=5)
    assert a == b
    assert 0.0 <= a.pass_rate.low <= a.pass_rate.high <= 1.0
