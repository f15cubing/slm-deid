"""Publish the tuned LoRA adapter + model card to the HuggingFace Hub (submission checklist #2).

Uploads a local adapter directory (e.g. the `sft-v3-gpt551` folder from Drive) and `MODEL_CARD.md`
(as the repo's `README.md`) to a model repo. Run it from Colab where the adapter is mounted:

    python scripts/push_to_hub.py \
        --adapter /content/drive/MyDrive/slm-deid-v3/sft-v3-gpt551 \
        --repo-id <user>/slm-deid-name-judgment

Auth: `huggingface-cli login` first, or set `HF_TOKEN`.

Hard-ceiling guard: this refuses to upload if any file under the adapter directory looks like
quarantined eval data (an `eval/` path or a `quarantine`-named file). An adapter dir should contain
only LoRA weights + config; nothing from `eval/` ever belongs in a published artifact. The check is
belt-and-suspenders on top of the repo's `tests/test_no_eval_leakage.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Filenames we positively expect in a LoRA adapter dir — used only for a friendly warning.
_EXPECTED = {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}


def find_eval_leak(adapter_dir: Path) -> list[str]:
    """Return any files under ``adapter_dir`` that look like quarantined eval data.

    A published adapter must never carry eval material. We flag any path containing an ``eval``
    directory segment or a ``quarantine`` token in the name. Returns the offending relative paths.
    """
    offenders: list[str] = []
    for p in adapter_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(adapter_dir)
        parts = {seg.lower() for seg in rel.parts}
        name = p.name.lower()
        if "eval" in parts or "quarantine" in name or "hardcases" in name:
            offenders.append(str(rel))
    return offenders


def push(
    adapter_dir: Path,
    repo_id: str,
    card: Path,
    private: bool,
    dry_run: bool = False,
) -> None:
    adapter_dir = adapter_dir.resolve()
    if not adapter_dir.is_dir():
        sys.exit(f"adapter dir not found: {adapter_dir}")

    offenders = find_eval_leak(adapter_dir)
    if offenders:
        sys.exit(
            "REFUSING TO PUSH — adapter dir contains files that look like quarantined eval "
            f"data (hard ceiling): {offenders}"
        )

    files = sorted(p.name for p in adapter_dir.iterdir() if p.is_file())
    if not (_EXPECTED & set(files)):
        print(f"WARNING: no adapter weights ({_EXPECTED}) found in {adapter_dir}; found {files}")

    print(f"Adapter dir : {adapter_dir}")
    print(f"Files       : {files}")
    print(f"Repo id     : {repo_id} ({'private' if private else 'public'})")
    print(f"Model card  : {card} → README.md")
    if dry_run:
        print("[dry-run] leak-guard passed; not uploading.")
        return

    from huggingface_hub import HfApi, upload_file  # lazy: Colab-only dep

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add sft-v3-gpt551 LoRA adapter",
    )
    if card.is_file():
        upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
    print(f"Done → https://huggingface.co/{repo_id}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path, help="Local adapter directory.")
    parser.add_argument("--repo-id", required=True, help="Target Hub repo, e.g. user/slm-deid.")
    parser.add_argument("--card", type=Path, default=REPO / "MODEL_CARD.md")
    parser.add_argument("--private", action="store_true", help="Create the repo private.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run the leak-guard + summary, do not upload."
    )
    args = parser.parse_args(argv)
    push(args.adapter, args.repo_id, args.card, args.private, args.dry_run)


if __name__ == "__main__":
    main()
