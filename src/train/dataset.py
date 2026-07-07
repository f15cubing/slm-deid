"""Turn our JSONL into an SFT dataset (Day 2/3).

Each example becomes a chat sequence [system, user=input, assistant=tagged target], serialized with
the Qwen3 chat template in **non-thinking** mode. Completion-only masking (loss on the assistant
tagged output only, not the prompt) is applied at train time via TRL's
``DataCollatorForCompletionOnlyLM`` keyed on :data:`RESPONSE_TEMPLATE`.

The pure helpers (`example_to_messages`, `RESPONSE_TEMPLATE`) are unit-tested; the dataset build
needs a tokenizer and runs on Colab.
"""

from __future__ import annotations

from src.common import prompts
from src.common.schema import Example

# Marker that begins the assistant turn in the Qwen3 chat template. The completion-only collator
# masks everything up to (and including) this so loss is computed only on the tagged output.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def example_to_messages(ex: Example) -> list[dict[str, str]]:
    """[system, user(input), assistant(target)] for one training example."""
    return prompts.build_training_messages(ex.input, ex.target)


def build_sft_dataset(path: str, tokenizer):
    """Load JSONL -> a HuggingFace Dataset with a serialized 'text' field. GPU/Colab (datasets)."""
    from datasets import Dataset

    from src.common.schema import read_jsonl

    examples = [ex.validate() for ex in read_jsonl(path)]

    def _text(ex: Example) -> str:
        return tokenizer.apply_chat_template(
            example_to_messages(ex),
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    return Dataset.from_dict({"text": [_text(ex) for ex in examples]})
