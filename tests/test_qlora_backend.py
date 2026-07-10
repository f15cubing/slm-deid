"""Pure backend-selection logic for QLoRA training (src/train/qlora.py).

The model-loading/training path needs a GPU + heavy deps and is exercised by the smoke loop,
but the backend *decisions* are pure and tested here.
"""

import dataclasses

import pytest

from src.train.qlora import (
    coerce_sft_eos_token,
    ensure_hf_base,
    select_amp_flags,
    training_backend_settings,
)


@dataclasses.dataclass
class _CfgWithEos:
    eos_token: str = "<EOS_TOKEN>"  # TRL/Unsloth placeholder default


@dataclasses.dataclass
class _CfgNoEos:
    max_length: int = 10


class _Tok:
    eos_token = "<|im_end|>"


def test_coerce_overwrites_placeholder_on_built_config():
    # The out-of-vocab '<EOS_TOKEN>' placeholder is replaced in place with the real eos, and the
    # same object is returned (so SFTTrainer reads the corrected value).
    cfg = _CfgWithEos()
    out = coerce_sft_eos_token(cfg, _Tok())
    assert out is cfg
    assert cfg.eos_token == "<|im_end|>"


def test_coerce_noop_when_field_absent():
    # Older TRL without the field: don't add an attribute (keeps the hf/MPS path working).
    cfg = _CfgNoEos()
    coerce_sft_eos_token(cfg, _Tok())
    assert not hasattr(cfg, "eos_token")


def test_coerce_noop_when_tokenizer_has_no_eos():
    class _NoEosTok:
        eos_token = None

    cfg = _CfgWithEos()
    coerce_sft_eos_token(cfg, _NoEosTok())
    assert cfg.eos_token == "<EOS_TOKEN>"  # left untouched; nothing to pin


def test_unsloth_settings_use_8bit_adam():
    s = training_backend_settings("unsloth")
    assert s["optim"] == "adamw_8bit"
    assert s["sft_extra"] == {}


def test_amp_flags_t4_falls_back_to_fp16():
    # Turing (T4) has no bf16 — must train in fp16, never bf16.
    assert select_amp_flags(False) == {"bf16": False, "fp16": True}


def test_amp_flags_ampere_uses_bf16():
    # Ampere+ (A100) supports bf16 — prefer it, disable fp16.
    assert select_amp_flags(True) == {"bf16": True, "fp16": False}


def test_hf_settings_use_torch_adam_and_disable_amp():
    s = training_backend_settings("hf")
    assert s["optim"] == "adamw_torch"
    # MPS has no CUDA-style AMP; dtype is controlled at model load, so AMP flags stay off.
    assert s["sft_extra"]["fp16"] is False
    assert s["sft_extra"]["bf16"] is False
    assert s["sft_extra"]["dataloader_pin_memory"] is False


def test_ensure_hf_base_accepts_full_precision_repo():
    ensure_hf_base("Qwen/Qwen3-1.7B")  # must not raise


def test_ensure_hf_base_rejects_bnb_4bit_checkpoint():
    with pytest.raises(ValueError):
        ensure_hf_base("unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
