"""Publish the v3 training/validation splits to the HuggingFace Datasets Hub (checklist #1).

HARD-CEILING SCRIPT. The quarantined evaluation sets are physically separate and must **never** be
published alongside the training data. Before uploading anything this script:

  1. loads every split file and asserts **no row is `quarantine=true`**;
  2. re-runs the eval-leakage check (`src/eval` normalization) and asserts **0 overlap** between
     the split inputs and every quarantined `eval/**/*.jsonl` input — the same guarantee
     `tests/test_no_eval_leakage.py` enforces in CI;
  3. only then uploads the split `.jsonl` files + `docs/dataset-card-v3.md` (as `README.md`).

Only files you pass via `--splits-dir` are ever uploaded — `eval/` is never in scope. If either
check fails the script exits non-zero and uploads nothing.

    python scripts/push_dataset.py \
        --splits-dir /content/drive/MyDrive/slm-deid-v3/splits \
        --repo-id <user>/slm-deid-name-judgment

Auth: `huggingface-cli login` or `HF_TOKEN`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # allow `python scripts/push_dataset.py` from anywhere
    sys.path.insert(0, str(REPO))

from src.common.schema import read_jsonl  # noqa: E402

EVAL_DIR = REPO / "eval"


def _norm(text: str) -> str:
    """Same normalization as tests/test_no_eval_leakage.py (whitespace-collapsed, lowercased)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _eval_inputs() -> set[str]:
    return {_norm(ex.input) for f in sorted(EVAL_DIR.rglob("*.jsonl")) for ex in read_jsonl(f)}


def check_splits(split_files: list[Path]) -> tuple[int, set[str]]:
    """Run both hard-ceiling checks. Exit non-zero on any violation. Returns (row_count, inputs)."""
    if not split_files:
        sys.exit("no split .jsonl files found — nothing to publish")

    eval_inputs = _eval_inputs()
    if not eval_inputs:
        sys.exit("no eval/**/*.jsonl found — cannot verify leakage guard; refusing to publish")

    rows = 0
    seen_inputs: set[str] = set()
    quarantined: list[str] = []
    leaked: list[str] = []
    for f in split_files:
        for ex in read_jsonl(f):
            rows += 1
            if ex.quarantine:
                quarantined.append(f"{f.name}:{ex.id}")
            n = _norm(ex.input)
            seen_inputs.add(n)
            if n in eval_inputs:
                leaked.append(f"{f.name}:{ex.id}")

    if quarantined:
        sys.exit(f"REFUSING TO PUSH — split contains quarantine=true rows: {quarantined}")
    if leaked:
        sys.exit(
            "REFUSING TO PUSH — EVAL LEAKAGE: these split rows duplicate quarantined eval "
            f"inputs: {leaked}"
        )
    print(
        f"Leak-guard PASSED: {rows} rows across {len(split_files)} files; "
        f"0 quarantine rows, 0 overlap vs {len(eval_inputs)} eval inputs."
    )
    return rows, seen_inputs


def push(
    splits_dir: Path,
    repo_id: str,
    card: Path,
    private: bool,
    dry_run: bool = False,
) -> None:
    splits_dir = splits_dir.resolve()
    if not splits_dir.is_dir():
        sys.exit(f"splits dir not found: {splits_dir}")
    split_files = sorted(splits_dir.glob("*.jsonl"))

    check_splits(split_files)

    print(f"Splits dir : {splits_dir}")
    print(f"Files      : {[p.name for p in split_files]}")
    print(f"Repo id    : {repo_id} ({'private' if private else 'public'})")
    print(f"Card       : {card} → README.md")
    if dry_run:
        print("[dry-run] guards passed; not uploading.")
        return

    from huggingface_hub import HfApi, upload_file  # lazy: Colab-only dep

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for f in split_files:
        upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"data/{f.name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {f.name}",
        )
    if card.is_file():
        upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    print(f"Done → https://huggingface.co/datasets/{repo_id}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=REPO / "data" / "splits",
        help="Directory of split .jsonl files (default data/splits; point at Drive for gpt551).",
    )
    parser.add_argument("--repo-id", required=True, help="Target Hub dataset repo.")
    parser.add_argument("--card", type=Path, default=REPO / "docs" / "dataset-card-v3.md")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run the guards + summary, do not upload."
    )
    args = parser.parse_args(argv)
    push(args.splits_dir, args.repo_id, args.card, args.private, args.dry_run)


if __name__ == "__main__":
    main()
