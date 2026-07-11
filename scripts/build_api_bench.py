"""Generate a fresh, QUARANTINED API benchmark via the live teacher — reusing datagen's guards.

This produces a new quarantined test split (`benchmarks/api_bench/`) of teacher-written passages
across the hard categories. It reuses `src.datagen.build_dataset`, so every row goes through the
SAME quality gates and the SAME three eval-leakage guards (token / surface / passage de-leak) as
training data — the no-eval-leakage hard ceiling is enforced, not re-implemented.

Gold-label trust (honest scope — do NOT overstate):
- **Minimal-pair rows** (person_vs_common / _place / _eponym / possessive): gold is largely **by
  construction** — the disposition is fixed (person-half tags its token; non-person-half tags
  NOTHING, enforced), so the withhold labels especially do not depend on the teacher's opinion.
- **Single-passage rows** (first_name_only / third_party, ~1/4 of the set): gold IS the teacher's
  own tagging, checked only by a second pass from the *same* model. A correlated teacher+verifier
  error can pass the gate, and a model trained on the same teacher (gpt551) can be flattered on
  exactly these rows. Treat those categories' numbers with that caveat; the pair rows are the
  trustworthy core. (Cross-checking against a *different* frontier model via
  `scripts/eval_frontier.py` is the mitigation; large frontier-vs-gold disagreement flags labels.)

Vocab scope: tokens come from the datagen bank, disjoint from the EXISTING eval sets but OVERLAPPING
the training banks. So this tests generalization to NEW phrasings of KNOWN ambiguous surfaces — it
complements, not replaces, the held-out `eval/ood_probe`.

Needs `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`/`TEACHER_MODEL` for a gateway). ~2-3 teacher calls per
pair; runs on CPU. The committed benchmark was built with `--seed 11 --scale 2.5` (92 rows); the
default `--seed 7 --scale 1.0` gives a smaller ~37-row batch. Note: `negative_trap` singles tend to
be fully dropped (the teacher slips names into "no-name" passages), so that category may be absent —
the pairs' non-person halves still exercise withholding.

    OPENAI_API_KEY=... python -m scripts.build_api_bench --seed 11 --scale 2.5 \
        --out benchmarks/api_bench/api_bench.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

from src.common.schema import write_jsonl
from src.datagen.generate import DatagenConfig, build_dataset
from src.datagen.teacher import TeacherGenerator


def build_config(scale: float, seed: int) -> DatagenConfig:
    """A small, balanced benchmark recipe (~70 examples at scale 1.0). val_frac=0 → one split."""
    return DatagenConfig(
        minimal_pairs={
            "person_vs_common": 8,
            "person_vs_place": 6,
            "possessive": 5,
            "person_vs_eponym": 5,
        },
        category_counts={
            "negative_trap": 10,
            "first_name_only": 8,
            "third_party": 6,
        },
        scale=scale,
        seed=seed,
        val_frac=0.0,
        eval_dir="eval",
    )


def quarantine(examples: list) -> list:
    """Stamp every row quarantine=True.

    The flag is a marker/assertion (no datagen or training code branches on it). Actual protection
    against training on this data is structural: it lives under `benchmarks/` (never a training
    input path) and `tests/test_api_bench.py` fails if any row appears in a training split.
    """
    return [dataclasses.replace(ex, quarantine=True) for ex in examples]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a quarantined API benchmark (live teacher).")
    ap.add_argument("--out", default="benchmarks/api_bench/api_bench.jsonl")
    ap.add_argument("--seed", type=int, default=7, help="new seed: passages disjoint from training")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7, help="teacher sampling (variety)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it (+ OPENAI_BASE_URL/TEACHER_MODEL for a gateway)."
        )

    cfg = build_config(args.scale, args.seed)
    # Fail fast if the eval dir is missing: the token/passage leakage guards read it, and an
    # absent path makes them silently no-op (only the static BLOCKLIST guard would fire). Run
    # from the repo root so the hard ceiling is actually enforced during generation.
    if not Path(cfg.eval_dir).is_dir():
        raise SystemExit(
            f"eval dir {cfg.eval_dir!r} not found (cwd={os.getcwd()}). Run from the repo root so "
            "the eval-leakage guards can read the quarantined eval set."
        )

    from src.eval.judge import build_openai_complete

    complete = build_openai_complete(temperature=args.temperature)
    teacher = TeacherGenerator(gen=complete, verify=complete)

    train, val, drops = build_dataset(cfg, teacher)
    bench = quarantine(train + val)
    for ex in bench:
        ex.validate()  # integrity + span + source, re-checked after stamping

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out), bench)
    print(f"[api_bench] wrote {len(bench)} quarantined examples -> {out}")
    # All three guard counts (all must be 0 for the hard ceiling): eval_token_leak / surface / leak.
    guards = {k: drops.get(k, 0) for k in ("eval_token_leak", "eval_surface_leak", "eval_leak")}
    print(f"[api_bench] eval-leakage guards (all must be 0): {guards}")
    print(f"[api_bench] full drop counts: {drops}")


if __name__ == "__main__":
    main()
