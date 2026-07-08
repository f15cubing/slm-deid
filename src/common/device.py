"""Backend + device selection shared by training and inference.

Two runtimes are supported (see ``docs/plan.md`` and the Mac-MPS pivot):

- ``unsloth`` — CUDA/Colab: 4-bit QLoRA via unsloth + bitsandbytes.
- ``hf`` — Apple Silicon / CPU: plain LoRA via transformers + PEFT on the ``mps`` device.

``detect_backend`` picks ``unsloth`` only when a CUDA GPU is actually present, so existing
Colab commands keep working while a Mac auto-selects the ``hf`` path. torch is imported lazily
so this module stays importable on a machine without torch (it just resolves to ``hf``/``cpu``).
"""

from __future__ import annotations


def detect_backend(explicit: str | None = None) -> str:
    """Return the training/inference backend: ``"unsloth"`` (CUDA) or ``"hf"`` (Mac/CPU).

    An explicit value other than ``"auto"`` is honored verbatim; otherwise we probe for CUDA.
    """
    if explicit and explicit != "auto":
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "unsloth"
    except Exception:
        pass
    return "hf"


def pick_device() -> str:
    """Return the torch device string: ``"cuda"`` > ``"mps"`` > ``"cpu"``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
