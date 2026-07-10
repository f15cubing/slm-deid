"""Generate a fresh, QUARANTINED API benchmark via the live teacher — reusing datagen's guards.

This produces a new quarantined test split (`benchmarks/api_bench/`) of teacher-written passages
across the hard categories. It reuses `src.datagen.build_dataset`, so every row goes through the
SAME quality gates and the SAME three eval-leakage guards (token / surface / passage de-leak) as
training data —
the no-eval-leakage hard ceiling is enforced, not re-implemented. Gold labels are **by
construction** (the minimal-pair disposition is known: person-half tags its token, non-person-half
tags nothing), NOT by the teacher's own tagging opinion — so a model trained on the same teacher
(gpt551) is not flattered by grading against teacher labels.

IMPORTANT (honest scope): tokens come from the datagen vocab bank, which is disjoint from the
EXISTING eval sets but OVERLAPS the training banks. So this benchmark tests generalization to NEW,
naturally-worded passages on KNOWN ambiguous surfaces — it complements, not replaces, the held-out
`eval/ood_probe` (vocab disjoint from training too). Read gpt551/authored numbers on it as in-vocab
robustness, and cross-check against the frontier (run `scripts/eval_frontier.py` on it).

Needs `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`/`TEACHER_MODEL` for a gateway). ~2-3 teacher calls per
pair; runs on CPU.

    OPENAI_API_KEY=... python -m scripts.build_api_bench --out benchmarks/api_bench/api_bench.jsonl
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
    """Stamp every row quarantine=True (kept out of training by the leakage guard + tests)."""
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

    from src.eval.judge import build_openai_complete

    complete = build_openai_complete(temperature=args.temperature)
    teacher = TeacherGenerator(gen=complete, verify=complete)

    cfg = build_config(args.scale, args.seed)
    train, val, drops = build_dataset(cfg, teacher)
    bench = quarantine(train + val)
    for ex in bench:
        ex.validate()  # integrity + span + source, re-checked after stamping

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out), bench)
    print(f"[api_bench] wrote {len(bench)} quarantined examples -> {out}")
    print(f"[api_bench] leakage-guard drops (must show eval_leak=0): {drops}")


if __name__ == "__main__":
    main()
