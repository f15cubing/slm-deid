"""Prompt construction + chat serialization (Day 1, specs S1.1 / S1.3).

The system prompt encodes the locked behavior spec. ``build_messages`` produces the chat
message list; ``serialize`` renders it through a tokenizer's chat template in **non-thinking
mode** so we can eyeball prompt-vs-completion boundaries before training (completion-only
masking lands Day 3).

Keep the tag markers referenced from ``tags`` — never hard-code them here.
"""

from __future__ import annotations

from .tags import NAME_CLOSE, NAME_OPEN

SYSTEM_PROMPT = (
    "You de-identify educational text by tagging personal names, and nothing else.\n"
    "\n"
    "Rules:\n"
    f"1. Wrap every span that refers to a REAL PERSON'S NAME in {NAME_OPEN}…{NAME_CLOSE}.\n"
    "2. Tag EVERY mention of each personal name, including first-name-only mentions.\n"
    "3. Do NOT tag identically-spelled non-person uses: a method or unit named after "
    "someone (\"the Newton method\", \"Newton's laws\"), a place (\"Chelsea\", \"Darwin\", "
    "\"Florence\"), a common word (\"Grace\", \"Hope\", \"Bishop\"), a brand, or a course title.\n"
    "4. Do NOT tag emails, phone numbers, IDs, dates, or URLs — those are handled elsewhere.\n"
    "5. Change NOTHING else. Return the input text byte-for-byte identical except for the "
    "inserted tags. Do not add, remove, reword, reorder, or reformat any other character.\n"
    "\n"
    "Output ONLY the tagged text."
)

# A handful of canonical ambiguous passages for the Day-1 smoke check and the Day-5 demo.
# (input, note) — the note describes the intended judgment, not a gold label format.
SHOWCASE = [
    ("Newton was frustrated when the experiment failed.", "Newton = person → tag"),
    ("We applied the Newton method to approximate the root.", "Newton = method → don't tag"),
    ("Newton's laws describe classical motion.", "eponymous laws → don't tag"),
    ("I visited Chelsea last summer and loved it.", "Chelsea = place → don't tag"),
    ("Chelsea helped me revise my thesis statement.", "Chelsea = person → tag"),
    ("thanks, Sam — that explanation finally clicked", "first-name-only person → tag"),
    ("My teacher Ms. Rivera said Grace is a virtue worth practicing.",
     "Rivera = person → tag; Grace = concept → don't tag"),
]


def build_messages(passage: str, system: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Return the chat message list for tagging ``passage``."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": passage},
    ]


def build_training_messages(
    passage: str, tagged: str, system: str = SYSTEM_PROMPT
) -> list[dict[str, str]]:
    """Chat messages for a *training* example: prompt + the gold tagged assistant turn."""
    return build_messages(passage, system) + [{"role": "assistant", "content": tagged}]


def serialize(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Render ``messages`` via the tokenizer's chat template in non-thinking mode.

    Runs on CPU (tokenizer only) — no GPU needed. Used by S1.3 to eyeball the serialized
    example and confirm no ``<think>`` content leaks. Qwen3 exposes ``enable_thinking``; we
    pass it through and fall back gracefully for tokenizers that don't accept the kwarg.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        # Tokenizer without the enable_thinking kwarg.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
