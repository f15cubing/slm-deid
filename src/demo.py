"""Base-vs-tuned inference demo (Day 5 submission deliverable).

Runs the **prompted base** and the **fine-tuned** model side-by-side on a fixed set of
ambiguous passages (``prompts.SHOWCASE`` — "Newton was frustrated" [tag] vs "the Newton method"
[don't], "I visited Chelsea" [don't] vs "Chelsea helped me" [tag], first-name-only, …) so the
contrast the whole project is about is visible in one screen: the base wobbles / over-tags, the
tune holds.

This is the **real model** demo — NOT the ``pipeline/cli.py --demo`` heuristic stub. It loads the
LoRA adapter over the base through the same :func:`src.infer.load_hf_tagger` the eval harness uses,
so what you see here is what the eval measured.

Two entry points:

- :func:`run_demo` / :func:`format_demo` take Tagger instances and are pure — unit-testable on CPU
  with :class:`src.infer.FunctionTagger` fakes, no model required.
- :func:`main` (``python -m src.demo --adapter <path>``) loads the real models on Colab/GPU and
  prints the table. Point ``--adapter`` at the ``sft-v3-gpt551`` adapter dir (e.g. on Drive).

Eval-only: this never writes text back into training, so there is no eval-leakage path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.common import prompts
from src.common.tags import NAME_CLOSE, NAME_OPEN
from src.infer import Tagger, load_hf_tagger


@dataclass
class DemoRow:
    """One showcase passage tagged by both models."""

    passage: str
    note: str
    base: str
    tuned: str

    @property
    def base_tags(self) -> int:
        return self.base.count(NAME_OPEN)

    @property
    def tuned_tags(self) -> int:
        return self.tuned.count(NAME_OPEN)

    @property
    def differ(self) -> bool:
        """Did the two models produce different output? (the interesting rows)."""
        return self.base.strip() != self.tuned.strip()


def run_demo(
    base: Tagger,
    tuned: Tagger,
    passages: list[tuple[str, str]] | None = None,
) -> list[DemoRow]:
    """Tag each showcase passage with both models and pair the results.

    ``passages`` is a list of ``(passage, note)`` pairs; defaults to ``prompts.SHOWCASE``. The
    note is the human-readable intended judgment (not a gold label) — it tells a viewer what the
    right call is so they can see which model made it.
    """
    items = passages if passages is not None else prompts.SHOWCASE
    return [
        DemoRow(passage=p, note=note, base=base.tag(p), tuned=tuned.tag(p)) for p, note in items
    ]


def format_demo(rows: list[DemoRow]) -> str:
    """Render the demo rows as a readable side-by-side block for a terminal / screen recording."""
    lines: list[str] = []
    bar = "=" * 88
    lines.append(bar)
    lines.append("  De-Id name-judgment demo — PROMPTED BASE  vs  FINE-TUNED (sft-v3-gpt551)")
    lines.append(f"  tag markers: {NAME_OPEN}…{NAME_CLOSE}")
    lines.append(bar)
    for i, r in enumerate(rows, 1):
        flag = "  ← models disagree" if r.differ else ""
        lines.append("")
        lines.append(f"[{i}] {r.passage}")
        lines.append(f"    intended : {r.note}{flag}")
        lines.append(f"    base     : {r.base}")
        lines.append(f"    tuned    : {r.tuned}")
    disagree = sum(1 for r in rows if r.differ)
    lines.append("")
    lines.append(bar)
    lines.append(f"  {disagree}/{len(rows)} passages the base and tune tag differently.")
    lines.append(bar)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the tuned LoRA adapter dir (e.g. the sft-v3-gpt551 folder on Drive).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Base model override; defaults to the backend's canonical base (4-bit on CUDA).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["unsloth", "hf"],
        help="Force a backend; auto-detected (unsloth on CUDA, hf on MPS/CPU) when omitted.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    print("Loading prompted base …")
    base = load_hf_tagger(
        model_name=args.model_name,
        adapter=None,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
    )
    print("Loading fine-tuned model (base + adapter) …")
    tuned = load_hf_tagger(
        model_name=args.model_name,
        adapter=args.adapter,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
    )
    rows = run_demo(base, tuned)
    print(format_demo(rows))


if __name__ == "__main__":
    main()
