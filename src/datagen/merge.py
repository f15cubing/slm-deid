"""Merge dataset batches (+ optional CRAPII real slice) into deduped, leakage-guarded splits.

Lets us build the training set from several SHORT generation batches (each of which completes
reliably) instead of one long run that can be interrupted, and fold in the CRAPII real slice. All
sources are concatenated, deduped by normalized input, re-run through the SAME eval-leakage guards
(token + passage-surface + exact-passage), then split fresh into train/val.

    python -m src.datagen.merge --out data \
        --sources data/splits/train.jsonl data/splits/val.jsonl data/_b1/splits/train.jsonl \
        --crapii data/raw/cleaned_repository_pii_train.json --crapii-limit 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.schema import Example, read_jsonl
from src.datagen.generate import (
    _norm,
    deleak_and_split,
    drop_eval_surface_overlap,
    drop_eval_token_overlap,
    write_splits,
)


def _dedup(examples: list[Example]) -> list[Example]:
    """Drop examples with a duplicate normalized input (keep the first)."""
    seen: set[str] = set()
    out: list[Example] = []
    for ex in examples:
        key = _norm(ex.input)
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def merge(
    sources: list[str],
    crapii_path: str | None = None,
    crapii_limit: int = 0,
    crapii_max_chars: int = 3000,
    eval_dir: str = "eval",
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[Example], list[Example], dict[str, int]]:
    """Concat sources (+ optional CRAPII) -> dedup -> leakage guards -> fresh train/val split."""
    pool: list[Example] = []
    for s in sources:
        p = Path(s)
        if p.exists():
            pool.extend(read_jsonl(p))
    n_sources = len(pool)

    if crapii_path and Path(crapii_path).exists():
        from src.datagen.real_data import load_crapii

        pool.extend(
            load_crapii(
                crapii_path,
                limit=(crapii_limit or None),
                max_chars=crapii_max_chars,
                names_only=True,
            )
        )
    n_crapii = len(pool) - n_sources

    deduped = _dedup(pool)
    n_dupes = len(pool) - len(deduped)
    guarded, n_tok = drop_eval_token_overlap(deduped, eval_dir)
    guarded, n_surf = drop_eval_surface_overlap(guarded)
    train, val, n_leak = deleak_and_split(guarded, eval_dir, val_frac, seed)

    stats = {
        "sources_loaded": n_sources,
        "crapii_added": n_crapii,
        "dupes_dropped": n_dupes,
        "eval_token_leak": n_tok,
        "eval_surface_leak": n_surf,
        "eval_leak": n_leak,
        "train": len(train),
        "val": len(val),
    }
    return train, val, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", required=True, help="JSONL split files to merge")
    ap.add_argument("--crapii", default=None, help="optional CRAPII json/jsonl path")
    ap.add_argument("--crapii-limit", type=int, default=0)
    ap.add_argument("--crapii-max-chars", type=int, default=3000)
    ap.add_argument("--eval-dir", default="eval")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data", help="out_dir; writes {out}/splits/{train,val}.jsonl")
    args = ap.parse_args()

    train, val, stats = merge(
        args.sources,
        crapii_path=args.crapii,
        crapii_limit=args.crapii_limit,
        crapii_max_chars=args.crapii_max_chars,
        eval_dir=args.eval_dir,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    counts = write_splits(train, val, args.out)
    print(f"train={counts['train']} val={counts['val']} stats={stats}")


if __name__ == "__main__":
    main()
