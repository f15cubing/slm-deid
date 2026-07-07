"""Inference interface: turn a passage into a tagged passage (Day 2).

`Tagger` is the small contract the eval harness and demo depend on: ``tag(passage) -> tagged``.
Two implementations:

- :class:`HFTagger` — wraps a HuggingFace/unsloth model + tokenizer, generates with the
  non-thinking chat template. Heavy deps (torch) are imported lazily so this module stays
  importable (and unit-testable) on a machine with no GPU.
- :class:`FunctionTagger` — wraps any ``str -> str`` callable (fakes/mocks in tests, or a
  rules baseline).

Model loading lives in :func:`load_hf_tagger` (Colab/GPU); it is not imported at module load.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from src.common import prompts


@runtime_checkable
class Tagger(Protocol):
    def tag(self, passage: str) -> str: ...


class FunctionTagger:
    """Adapt a plain ``str -> str`` function to the Tagger protocol."""

    def __init__(self, fn: Callable[[str], str], name: str = "function"):
        self._fn = fn
        self.name = name

    def tag(self, passage: str) -> str:
        return self._fn(passage)


class HFTagger:
    """Prompted tagging via a HF/unsloth causal LM (runs on GPU/Colab).

    The same class serves the **base** model (no adapter) and the **tuned** model (LoRA adapter
    already attached to ``model``); the eval harness just passes whichever it loaded.
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 256, name: str = "hf"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.name = name

    def tag(self, passage: str) -> str:
        import torch  # lazy

        enc = self.tokenizer.apply_chat_template(
            prompts.build_messages(passage),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return completion.strip()


def tag_all(tagger: Tagger, passages: list[str]) -> list[str]:
    """Tag a list of passages (sequential; batching is a later optimization)."""
    return [tagger.tag(p) for p in passages]


def load_hf_tagger(
    model_name: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    adapter: str | None = None,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
) -> HFTagger:
    """Load a base (or LoRA-adapted) model via unsloth and wrap it as an HFTagger. GPU only."""
    from unsloth import FastLanguageModel  # lazy, GPU-only

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter or model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    name = "tuned" if adapter else "base"
    return HFTagger(model, tokenizer, max_new_tokens=max_new_tokens, name=name)
