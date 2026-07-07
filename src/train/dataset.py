"""Turn our JSONL into an SFT dataset (Day 2/3).

Modern TRL (>=0.20) removed ``DataCollatorForCompletionOnlyLM``; completion-only masking is now a
dataset-preparation concern driven by ``SFTConfig(completion_only_loss=True)`` on a
**prompt-completion** dataset. So each example becomes::

    {"prompt": [system, user(input)], "completion": [assistant(target)]}

SFTTrainer applies the chat template and computes loss on the completion (the tagged output) only.
The pure helpers are unit-tested; the dataset build needs ``datasets`` and runs on Colab.
"""

from __future__ import annotations

from src.common import prompts
from src.common.schema import Example


def example_to_messages(ex: Example) -> list[dict[str, str]]:
    """[system, user(input), assistant(target)] — full conversation form."""
    return prompts.build_training_messages(ex.input, ex.target)


def example_to_prompt_completion(ex: Example) -> dict[str, list[dict[str, str]]]:
    """Prompt-completion form for SFTConfig(completion_only_loss=True)."""
    return {
        "prompt": prompts.build_messages(ex.input),
        "completion": [{"role": "assistant", "content": ex.target}],
    }


def build_sft_dataset(path: str):
    """Load JSONL -> a HuggingFace prompt-completion Dataset. GPU/Colab (needs `datasets`)."""
    from datasets import Dataset

    from src.common.schema import read_jsonl

    examples = [ex.validate() for ex in read_jsonl(path)]
    return Dataset.from_list([example_to_prompt_completion(ex) for ex in examples])
