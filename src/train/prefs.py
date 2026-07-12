"""Preference-pair construction for DPO (Day-5 stretch rung 1).

DPO needs ``{prompt, chosen, rejected}`` triples: a prompt, an on-spec completion (``chosen``),
and an off-spec one (``rejected``). ``chosen`` is free — it is exactly the gold ``target`` every
training row already carries. The work is producing a *good* ``rejected`` — a completion that is
wrong in the way the model actually tends to be wrong (over-tag / missed name / wrong boundary),
so the preference signal sharpens spec-adherence rather than teaching an arbitrary contrast.

**Hybrid sourcing (the locked design).**

- **Stage A — on-policy.** Sample the SFT model over the train inputs (done in the Colab
  notebook, GPU) and keep the rows where it *genuinely errs* (``behavioral_checks.check`` says
  not-passed). The model's own wrong output becomes ``rejected``. Highest-signal: DPO learns
  against real failure modes. :func:`pair_from_model_output` is the pure hook.
- **Stage B — deterministic backfill.** Where the model was already correct (no natural
  negative), synthesize ``rejected`` by perturbing the gold spans (:func:`make_over_tag`,
  :func:`make_missed_name`, :func:`make_wrong_boundary`). Guarantees every failure type and
  category is represented even where the SFT model is strong.

Every perturbation moves/adds/removes only tag *markers*, never other characters, so
``unwrap(rejected) == unwrap(chosen) == input``: the rejected is a pure **judgment** error, not a
text-integrity error, and stays well-formed. That keeps the DPO contrast about *which spans are
names* — the behavior we are training.

**Leakage.** Pairs are built only from the train split (already eval-disjoint) and no perturbation
touches the passage text, so no eval surface can be introduced. :func:`eval_leak_count` is the
belt-and-suspenders guard (a hard ceiling; reused by the test + the notebook).

Pure/CPU and unit-tested. The heavy ``datasets`` build (:func:`build_trl_dataset`) is lazy so this
module imports without GPU deps.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from src.common import prompts, tags
from src.common.schema import Example
from src.eval.behavioral_checks import check

# One capitalized word (the over-tag candidate pool: sentence-initial caps, brands, course words).
_CAP_WORD = re.compile(r"[A-Z][A-Za-z]+")
# Capitalized function words to skip as over-tag targets: tagging "The"/"And" is a degenerate
# negative the model never actually produces, so it teaches nothing. Prefer a content word.
_CAP_STOPWORDS = frozenset(
    {
        "The",
        "A",
        "An",
        "And",
        "But",
        "Or",
        "Nor",
        "So",
        "Yet",
        "For",
        "If",
        "In",
        "On",
        "At",
        "Of",
        "To",
        "As",
        "By",
        "It",
        "He",
        "She",
        "We",
        "They",
        "This",
        "That",
        "These",
        "Those",
        "My",
        "Our",
        "Their",
        "His",
        "Her",
        "Its",
        "When",
        "While",
        "Then",
        "Also",
    }
)
# Strategies rotated across Stage-B examples so all three failure modes are represented.
_ROTATION = ("over_tag", "missed_name", "wrong_boundary")


@dataclass(frozen=True)
class Pair:
    """One DPO preference triple, on ``build_messages`` prompt form.

    ``chosen``/``rejected`` are the assistant *content* strings; :func:`to_trl` wraps them into
    the conversational message lists TRL's ``DPOTrainer`` consumes.
    """

    id: str
    category: str
    input: str  # raw passage (dedup + leakage guard)
    chosen: str  # gold tagged target (on-spec)
    rejected: str  # off-spec completion
    strategy: str  # on_policy | over_tag | missed_name | wrong_boundary


# --- rendering: build a tagged string from a chosen set of spans ---------------------------


def _render(text: str, spans: list[tuple[int, int]]) -> str:
    """Insert NAME tags into ``text`` at ``spans`` (non-overlapping; any order).

    Only ever inserts markers, so ``unwrap(_render(text, spans)) == text`` for all inputs.
    """
    out: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        out.append(text[cursor:start])
        out.append(tags.wrap(text[start:end]))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _gold_spans(ex: Example) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in ex.name_spans()]


def _cap_nonname_spans(ex: Example) -> list[tuple[int, int]]:
    """Capitalized words in the input that are NOT (and don't overlap) a gold name span.

    These are the natural over-tag traps: a capitalized common word, brand, or sentence-initial
    non-name that the model is tempted to tag.
    """
    gold = _gold_spans(ex)
    spans: list[tuple[int, int]] = []
    for m in _CAP_WORD.finditer(ex.input):
        s, e = m.start(), m.end()
        if any(not (e <= gs or s >= ge) for gs, ge in gold):
            continue  # overlaps a gold name
        if m.group() in _CAP_STOPWORDS:
            continue  # skip degenerate function-word targets ("The", "And", …)
        spans.append((s, e))
    return spans


# --- Stage B: deterministic perturbations --------------------------------------------------


def make_over_tag(ex: Example) -> str | None:
    """Rejected = gold tagging PLUS one extra tag around a capitalized non-name (over-tag)."""
    extras = _cap_nonname_spans(ex)
    if not extras:
        return None
    candidate = _render(ex.input, _gold_spans(ex) + [extras[0]])
    return candidate if candidate != ex.target else None


def make_missed_name(ex: Example) -> str | None:
    """Rejected = gold tagging with ONE name span left untagged (leakage / missed name)."""
    gold = _gold_spans(ex)
    if not gold:
        return None
    # Drop the last gold span (deterministic); keep the rest tagged.
    candidate = _render(ex.input, gold[:-1])
    return candidate if candidate != ex.target else None


def make_wrong_boundary(ex: Example) -> str | None:
    """Rejected = gold tagging with ONE span's boundary wrong (swallows a possessive or the
    following word — the eponymous-possessive over-tag failure mode)."""
    gold = _gold_spans(ex)
    if not gold:
        return None
    # Extend the POSITIONALLY-last span (by end offset), not gold[-1]: the last-in-list span may
    # not be last in the text, and extending a non-final span could swallow a following gold name,
    # producing overlapping spans that _render would duplicate (breaking unwrap==input). The
    # positionally-last span can only extend over trailing non-name text, so integrity is preserved.
    start, end = max(gold, key=lambda s: s[1])
    rest = ex.input[end:]
    if rest[:2] == "'s":  # "Sarah" -> "Sarah's"
        new_end = end + 2
    else:
        m = re.match(r"\s+([A-Za-z]+)", rest)  # extend over the next word
        if not m:
            return None
        new_end = end + m.end()
    others = [sp for sp in gold if sp != (start, end)]
    candidate = _render(ex.input, others + [(start, new_end)])
    return candidate if candidate != ex.target else None


_STRATEGY_FN = {
    "over_tag": make_over_tag,
    "missed_name": make_missed_name,
    "wrong_boundary": make_wrong_boundary,
}


def stage_b_pair(ex: Example, preferred: str | None = None) -> Pair | None:
    """Synthesize a Pair for ``ex`` by perturbation.

    Tries ``preferred`` first (for round-robin coverage), then the remaining strategies, taking
    the first that yields a rejected differing from the gold target. Returns None if none apply
    (e.g. a negative_trap with no capitalized token — rare).
    """
    order = ([preferred] if preferred else []) + [s for s in _ROTATION if s != preferred]
    for strat in order:
        rejected = _STRATEGY_FN[strat](ex)
        # Deterministic negatives MUST be pure judgment errors (markers-only): enforce the
        # integrity invariant here so a future perturbation bug can never leak an altered passage
        # into training. (Stage-A on-policy negatives may legitimately drift, so no guard there.)
        if rejected is not None and tags.unwrap(rejected) == ex.input:
            return Pair(ex.id, ex.category, ex.input, ex.target, rejected, strat)
    return None


# --- Stage A: on-policy negatives from the model's own output ------------------------------


def pair_from_model_output(ex: Example, output: str) -> Pair | None:
    """Pair from a real SFT-model ``output`` iff it is a genuine error (not passing) and differs
    from the gold target. Returns None when the model already matched the gold judgment (so the
    caller falls back to :func:`stage_b_pair`)."""
    output = output.strip()
    if not output or output == ex.target:
        return None
    if check(ex, output).passed:
        return None  # model got it right — no natural negative here
    return Pair(ex.id, ex.category, ex.input, ex.target, output, "on_policy")


# --- orchestration -------------------------------------------------------------------------


def build_pairs(
    examples: list[Example],
    model_outputs: dict[str, str] | None = None,
    seed: int = 0,
    per_category_cap: int | None = None,
) -> list[Pair]:
    """Build the hybrid preference set: one pair per example where possible.

    ``model_outputs`` maps ``example.id -> raw SFT output`` (Stage A). For each example, an
    on-policy pair is used when the model genuinely erred; otherwise a deterministic Stage-B pair
    (strategies round-robined for coverage). ``per_category_cap`` optionally balances the set by
    keeping at most N pairs per category (seeded shuffle).
    """
    model_outputs = model_outputs or {}
    pairs: list[Pair] = []
    for i, ex in enumerate(examples):
        pair = None
        if ex.id in model_outputs:
            pair = pair_from_model_output(ex, model_outputs[ex.id])
        if pair is None:
            pair = stage_b_pair(ex, preferred=_ROTATION[i % len(_ROTATION)])
        if pair is not None:
            pairs.append(pair)

    if per_category_cap is not None:
        rng = random.Random(seed)
        by_cat: dict[str, list[Pair]] = {}
        for p in pairs:
            by_cat.setdefault(p.category, []).append(p)
        capped: list[Pair] = []
        for cat in sorted(by_cat):
            group = by_cat[cat]
            rng.shuffle(group)
            capped.extend(group[:per_category_cap])
        pairs = capped
    return pairs


# --- serialization + TRL dataset -----------------------------------------------------------


def to_trl(pair: Pair) -> dict:
    """Conversational ``{prompt, chosen, rejected}`` for TRL's DPOTrainer."""
    return {
        "prompt": prompts.build_messages(pair.input),
        "chosen": [{"role": "assistant", "content": pair.chosen}],
        "rejected": [{"role": "assistant", "content": pair.rejected}],
    }


def dump_pairs(pairs: list[Pair]) -> list[dict]:
    """Serialize pairs to plain dicts (for saving to JSONL / inspection)."""
    return [
        {
            "id": p.id,
            "category": p.category,
            "input": p.input,
            "chosen": p.chosen,
            "rejected": p.rejected,
            "strategy": p.strategy,
        }
        for p in pairs
    ]


def build_trl_dataset(pairs: list[Pair]):
    """Turn pairs into a HuggingFace Dataset in TRL conversational form (Colab; needs datasets)."""
    from datasets import Dataset

    return Dataset.from_list([to_trl(p) for p in pairs])


# --- leakage guard (hard ceiling) ----------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def eval_leak_count(pairs: list[Pair], eval_dir: str = "eval") -> int:
    """Number of pairs whose passage matches a quarantined eval input, or contains any eval
    surface from ``vocab.BLOCKLIST``. Must be 0 — pairs are drawn from the eval-disjoint train
    split and no perturbation edits the passage, so this is a guard, not a filter."""
    from pathlib import Path

    from src.common.schema import read_jsonl
    from src.datagen import vocab

    p = Path(eval_dir)
    eval_inputs: set[str] = set()
    if p.exists():
        eval_inputs = {_norm(ex.input) for f in p.rglob("*.jsonl") for ex in read_jsonl(f)}
    leaks = 0
    for pair in pairs:
        if _norm(pair.input) in eval_inputs or vocab.blocklist_surfaces_in(pair.input):
            leaks += 1
    return leaks
