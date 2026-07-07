"""Data-gen orchestrator (Day 2 wiring; Day 3 scales it up).

Pipeline: teacher-distilled ambiguous passages (per category) + Faker pattern-type negatives ->
second-pass verify -> quality gate -> eval-leakage filter -> train/val split -> JSONL.

Core (`build_dataset`) is pure/testable with a mock teacher. The CLI loads `configs/datagen.yaml`
and builds a real teacher via the lazy OpenAI/Anthropic factories in `src.eval.judge`.

    python -m src.datagen.generate --config configs/datagen.yaml
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.common.schema import Example, read_jsonl, write_jsonl
from src.datagen.negatives import generate_negatives
from src.datagen.quality_gate import filter_examples
from src.datagen.teacher import TeacherGenerator


@dataclass
class DatagenConfig:
    category_counts: dict[str, int] = field(default_factory=dict)
    negatives: int = 0
    seed: int = 0
    val_frac: float = 0.1
    out_dir: str = "data"
    eval_dir: str = "eval"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _eval_inputs(eval_dir: str) -> set[str]:
    p = Path(eval_dir)
    if not p.exists():
        return set()
    return {_norm(ex.input) for f in p.rglob("*.jsonl") for ex in read_jsonl(f)}


def deleak_and_split(
    examples: list[Example],
    eval_dir: str = "eval",
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[Example], list[Example], int]:
    """Drop any example whose input matches a quarantined eval input, then split train/val.

    Returns (train, val, n_eval_leak_dropped). Used by both the teacher pipeline and the
    real-data (CRAPII + negatives) assembly in the Day-3 notebook.
    """
    eval_inputs = _eval_inputs(eval_dir)
    deleaked = [ex for ex in examples if _norm(ex.input) not in eval_inputs]
    n_leak = len(examples) - len(deleaked)
    rng = random.Random(seed)
    rng.shuffle(deleaked)
    n_val = int(len(deleaked) * val_frac)
    return deleaked[n_val:], deleaked[:n_val], n_leak


def build_dataset(
    cfg: DatagenConfig,
    teacher: TeacherGenerator,
) -> tuple[list[Example], list[Example], dict[str, int]]:
    """Generate + gate + de-leak + split. Returns (train, val, drop_counts)."""
    raw: list[Example] = []
    verifier_targets: list[str | None] = []

    # 1) teacher-distilled ambiguous passages per category
    for category, count in cfg.category_counts.items():
        for i in range(count):
            ex = teacher.generate(category, id_=f"gen-{category}-{i:04d}")
            raw.append(ex)
            verifier_targets.append(teacher.verify_tagging(ex.input))

    # 2) Faker pattern-type negatives (already valid; no verifier needed)
    negs = generate_negatives(cfg.negatives, seed=cfg.seed) if cfg.negatives else []
    raw.extend(negs)
    verifier_targets.extend([None] * len(negs))

    # 3) quality gate
    kept, drops = filter_examples(raw, verifier_targets)

    # 4) eval-leakage filter (hard ceiling) + deterministic train/val split
    train, val, n_leak = deleak_and_split(kept, cfg.eval_dir, cfg.val_frac, cfg.seed)
    drops["eval_leak"] = n_leak
    return train, val, drops


def write_splits(train: list[Example], val: list[Example], out_dir: str = "data") -> dict[str, int]:
    counts = {
        "train": write_jsonl(Path(out_dir) / "splits" / "train.jsonl", train),
        "val": write_jsonl(Path(out_dir) / "splits" / "val.jsonl", val),
    }
    return counts


def _load_yaml(path: str) -> DatagenConfig:
    import yaml  # lazy

    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return DatagenConfig(**d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datagen.yaml")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    from src.eval.judge import build_anthropic_complete, build_openai_complete

    if args.provider == "anthropic":
        complete = build_anthropic_complete()
    else:
        complete = build_openai_complete()
    teacher = TeacherGenerator(gen=complete, verify=complete)

    train, val, drops = build_dataset(cfg, teacher)
    counts = write_splits(train, val, cfg.out_dir)
    print(f"train={counts['train']} val={counts['val']} drops={drops}")


if __name__ == "__main__":
    main()
