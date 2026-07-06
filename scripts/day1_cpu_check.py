"""CPU/MPS fallback for the Day-1 GPU checkpoints (no unsloth / no CUDA needed).

Validates the parts of docs/tasks/day-1.md that need the real Qwen3 tokenizer/model:
  - tag-syntax tokenizer test: do the ⟨NAME⟩ markers survive tokenization? (Day-1 decision)
  - S1.3: serialize one training example via the chat template, non-thinking; write the artifact
  - S1.1: one real generation, assert no <think> leak  (skip with --no-generate; it's slow on CPU)

Usage:
    python scripts/day1_cpu_check.py                # tokenizer test + serialize + one generation
    python scripts/day1_cpu_check.py --no-generate  # skip the slow generation step
"""

from __future__ import annotations

import argparse
import pathlib

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import prompts, tags

# Full-precision base (CPU/MPS). The unsloth 4bit repo is CUDA-only; the chat template is identical.
MODEL_NAME = "Qwen/Qwen3-1.7B"
ARTIFACT = pathlib.Path("docs/tasks/artifacts/day1-serialized-example.txt")


def tokenizer_tag_test(tokenizer) -> bool:
    print("\n== Tag-syntax tokenizer test ==")
    ok = True
    for name, marker in [("OPEN", tags.NAME_OPEN), ("CLOSE", tags.NAME_CLOSE)]:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        roundtrip = tokenizer.decode(ids)
        good = roundtrip == marker
        ok = ok and good
        status = "OK" if good else f"BROKEN -> {roundtrip!r}"
        print(f"  {name} {marker!r}: {len(ids)} tokens, roundtrip {status}")
    # A whole tagged sentence should unwrap back to the raw sentence after a decode round-trip.
    raw = "Chelsea helped me, but I visited Chelsea in London."
    tagged = f"{tags.wrap('Chelsea')} helped me, but I visited Chelsea in London."
    dec = tokenizer.decode(tokenizer.encode(tagged, add_special_tokens=False))
    print(f"  sentence round-trip preserved: {dec == tagged}")
    print(f"  unwrap(decoded) == raw: {tags.unwrap(dec) == raw}")
    return ok


def serialize_example(tokenizer) -> str:
    print("\n== S1.3 serialize training example ==")
    ex_input = "Chelsea helped me revise, but last year I also visited Chelsea in London."
    ex_target = (
        f"{tags.wrap('Chelsea')} helped me revise, but last year I also visited Chelsea in London."
    )
    assert tags.unwrap(ex_target) == ex_input, "integrity invariant broken in the example"
    serialized = prompts.serialize(
        tokenizer,
        prompts.build_training_messages(ex_input, ex_target),
        add_generation_prompt=False,
    )
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(serialized, encoding="utf-8")
    print(serialized)
    print(f"[OK] wrote {ARTIFACT}")
    return serialized


def one_generation(tokenizer) -> None:
    import torch

    print("\n== S1.1 one generation (non-thinking) ==")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  loading {MODEL_NAME} on {device} (slow on CPU/MPS)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    passage, note = prompts.SHOWCASE[0]
    input_ids = tokenizer.apply_chat_template(
        prompts.build_messages(passage),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=64, do_sample=False)
    completion = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    print(f"  INPUT   : {passage}")
    print(f"  EXPECTED: {note}")
    print(f"  OUTPUT  : {completion}")
    assert "<think>" not in completion and "</think>" not in completion, "thinking leaked!"
    print("  [OK] S1.1 — model responds, no thinking leak.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-generate", action="store_true", help="skip the slow generation step")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tag_ok = tokenizer_tag_test(tokenizer)
    serialize_example(tokenizer)
    if not args.no_generate:
        one_generation(tokenizer)

    print("\n== Summary ==")
    print(f"  tag markers tokenize cleanly: {tag_ok}  "
          f"({'keep ⟨NAME⟩' if tag_ok else 'consider @@..## fallback'})")


if __name__ == "__main__":
    main()
