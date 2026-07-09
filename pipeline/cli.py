"""Command-line entry point for the de-id pipeline.

Examples::

    # offline demo (heuristic stub tagger — plumbing only, not real judgment)
    echo "Email sam@school.edu — Chelsea helped me revise." | python -m pipeline.cli --demo

    # real de-identification with the tuned adapter
    python -m pipeline.cli --adapter outputs/sft-v3-mps --mode surrogate < essay.txt

Modes: ``tag`` (default, integrity-preserving), ``mask``, ``surrogate``.
"""

from __future__ import annotations

import argparse
import sys

from pipeline.deid import RENDER_MODES, Deidentifier


def _build_tagger(args):
    if args.demo or (not args.adapter and not args.base):
        from pipeline.stub import HeuristicNameStub

        if not args.demo:
            print(
                "[pipeline] no --adapter/--base given; using the heuristic stub "
                "(demo plumbing only, NOT real name judgment). Pass --demo to silence this.",
                file=sys.stderr,
            )
        return HeuristicNameStub()

    from src.infer import load_hf_tagger

    return load_hf_tagger(model_name=args.base, adapter=args.adapter)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end de-identification pipeline.")
    ap.add_argument("text", nargs="?", help="text to de-identify (default: read stdin)")
    ap.add_argument("--mode", choices=RENDER_MODES, default="tag")
    ap.add_argument("--adapter", help="path to a trained LoRA adapter (tuned model)")
    ap.add_argument("--base", help="base model name/path (default: backend canonical)")
    ap.add_argument("--pattern-backend", choices=["regex", "presidio"], default="regex")
    ap.add_argument("--surrogate-seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="use the offline heuristic stub tagger")
    args = ap.parse_args(argv)

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.rstrip("\n")

    deid = Deidentifier(
        _build_tagger(args),
        pattern_backend=args.pattern_backend,
        surrogate_seed=args.surrogate_seed,
    )
    result = deid.deidentify(text, mode=args.mode)
    print(result.text)
    if result.dropped_spans:
        print(
            f"[pipeline] dropped {result.dropped_spans} unaligned model span(s) "
            "(output drift; original text preserved)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
