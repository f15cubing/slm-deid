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
from src.datagen import vocab
from src.datagen.negatives import generate_negatives
from src.datagen.quality_gate import filter_examples
from src.datagen.teacher import TeacherGenerator


@dataclass
class DatagenConfig:
    # Teacher single passages per category (context variety).
    category_counts: dict[str, int] = field(default_factory=dict)
    # Matched minimal PAIRS per category (each entry -> that many pairs -> 2x examples). This is
    # the Day-4 bulk that teaches person-vs-non-person contrast on identical surfaces.
    minimal_pairs: dict[str, int] = field(default_factory=dict)
    negatives: int = 0
    # Multiplies every count above; the one knob to scale the whole recipe for the real run.
    scale: float = 1.0
    seed: int = 0
    val_frac: float = 0.1
    out_dir: str = "data"
    eval_dir: str = "eval"


def _scaled(count: int, scale: float) -> int:
    return int(round(count * scale))


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


def drop_eval_token_overlap(
    examples: list[Example], eval_dir: str = "eval"
) -> tuple[list[Example], int]:
    """Token-level eval-leakage guard (Day-4 TASK 4; a hard ceiling, in addition to the
    passage-level de-leak in :func:`deleak_and_split`).

    Drops any generated example whose intended ambiguous token overlaps the eval vocabulary — so
    the training data can never re-use an eval surface (Newton/Chelsea/…) even inside a *different*
    passage that the exact-passage de-leak would miss. Reads ``eval`` only to derive the forbidden
    vocabulary; nothing flows the other way.
    """
    ev = vocab.eval_vocab(eval_dir)
    if not ev:
        return list(examples), 0
    kept: list[Example] = []
    dropped = 0
    for ex in examples:
        tok = ex.ambiguous_token
        if tok and (vocab.token_words(tok) & ev):
            dropped += 1
            continue
        kept.append(ex)
    return kept, dropped


def build_dataset(
    cfg: DatagenConfig,
    teacher: TeacherGenerator,
) -> tuple[list[Example], list[Example], dict[str, int]]:
    """Generate + gate + de-leak + split. Returns (train, val, drop_counts)."""
    raw: list[Example] = []
    verifier_targets: list[str | None] = []
    rng = random.Random(cfg.seed)

    # 1) matched minimal pairs — person [tagged] vs identical non-person [untagged]. Tokens come
    #    from the curated, eval-DISJOINT bank; possessive pairs contrast a person's possessive
    #    with an eponymous possessive negative (distinct eponym token).
    for category, n_pairs in cfg.minimal_pairs.items():
        tokens = list(vocab.tokens_for(category))
        if not tokens:
            continue
        rng.shuffle(tokens)
        eponyms = list(vocab.EPONYMS)
        rng.shuffle(eponyms)
        for i in range(_scaled(n_pairs, cfg.scale)):
            person_token = tokens[i % len(tokens)]
            nonperson_token = eponyms[i % len(eponyms)] if category == "possessive" else None
            person, nonperson = teacher.generate_pair(
                category,
                person_token=person_token,
                nonperson_token=nonperson_token,
                id_prefix=f"pair-{category}-{i:04d}",
            )
            for ex in (person, nonperson):
                raw.append(ex)
                verifier_targets.append(teacher.verify_tagging(ex.input))

    # 2) teacher-distilled single passages per category (context variety)
    for category, count in cfg.category_counts.items():
        for i in range(_scaled(count, cfg.scale)):
            ex = teacher.generate(category, id_=f"gen-{category}-{i:04d}")
            raw.append(ex)
            verifier_targets.append(teacher.verify_tagging(ex.input))

    # 3) Faker pattern-type negatives (already valid; no verifier needed)
    n_neg = _scaled(cfg.negatives, cfg.scale)
    negs = generate_negatives(n_neg, seed=cfg.seed) if n_neg else []
    raw.extend(negs)
    verifier_targets.extend([None] * len(negs))

    # 4) quality gate (incl. Day-4 category-semantics)
    kept, drops = filter_examples(raw, verifier_targets)

    # 5) eval-leakage (hard ceiling): token-level guard THEN passage-level de-leak + split
    kept, n_token_leak = drop_eval_token_overlap(kept, cfg.eval_dir)
    train, val, n_leak = deleak_and_split(kept, cfg.eval_dir, cfg.val_frac, cfg.seed)
    drops["eval_token_leak"] = n_token_leak
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
