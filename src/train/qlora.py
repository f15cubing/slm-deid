"""QLoRA/LoRA SFT of Qwen3-1.7B for the NAME-judgment core (Day 3 real run; Day 2 smoke).

Two backends, auto-selected by hardware (see ``src/common/device.py``):

- ``unsloth`` (CUDA/Colab): 4-bit QLoRA via unsloth + bitsandbytes, ``adamw_8bit``.
- ``hf`` (Apple Silicon/CPU): plain LoRA via transformers + PEFT on ``mps``, ``adamw_torch``.

Both share the TRL ``SFTTrainer`` completion-only path. Reads ``configs/train.yaml`` (Colab) or
``configs/train.mps.yaml`` (Mac). Completion-only, non-thinking, r=32/a=32, lr=2e-4, seq 2048 by
default. ``--smoke`` shrinks to a 1-epoch pass over a tiny slice to prove the generate -> train ->
eval loop end to end (spec S2.11).

    python -m src.train.qlora --config configs/train.yaml           # Colab / CUDA
    python -m src.train.qlora --config configs/train.mps.yaml       # Mac / MPS
    python -m src.train.qlora --config configs/train.mps.yaml --smoke
"""

from __future__ import annotations

import argparse


def load_config(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def training_backend_settings(backend: str) -> dict:
    """Backend-specific optimizer + SFTConfig extras (pure; no heavy imports).

    unsloth uses bitsandbytes' 8-bit Adam; the hf/MPS path uses torch's Adam and keeps the
    CUDA-style AMP flags off (dtype is set at model load, and MPS has no such AMP path).
    """
    if backend == "unsloth":
        return {"optim": "adamw_8bit", "sft_extra": {}}
    return {
        "optim": "adamw_torch",
        "sft_extra": {"fp16": False, "bf16": False, "dataloader_pin_memory": False},
    }


def select_amp_flags(bf16_supported: bool) -> dict:
    """AMP precision for the CUDA/unsloth path: bf16 on Ampere+, else fp16.

    A Tesla T4 (Turing, CUDA cap 7.5) has no bf16 support, and current transformers/TRL default
    the trainer to bf16 on CUDA — which `TrainingArguments` then rejects ("needs Ampere+ GPU").
    Setting the flag explicitly from the hardware keeps the run portable across T4 and A100 without
    touching any LoRA hyperparameter (the frozen Day-4 config is unchanged).
    """
    return {"bf16": bool(bf16_supported), "fp16": not bf16_supported}


def coerce_sft_eos_token(sft_args, tokenizer):
    """Force the tokenizer's real EOS onto a *built* ``SFTConfig``; return the same object.

    Unsloth patches ``SFTConfig`` to inject the placeholder ``'<EOS_TOKEN>'`` (not in Qwen's
    vocab) as its ``eos_token`` default; newer TRL's ``SFTTrainer.__init__`` then rejects it as
    out-of-vocab. Passing ``eos_token=`` as a *constructor kwarg* does not stick — Unsloth's
    patched ``__init__`` re-injects the placeholder — so we overwrite the attribute *after*
    construction, right before ``SFTTrainer`` reads it. No-ops when the field is absent (older TRL
    / the hf-MPS path) or the tokenizer has no eos, leaving those backends untouched.
    """
    eos = getattr(tokenizer, "eos_token", None)
    if eos and hasattr(sft_args, "eos_token"):
        sft_args.eos_token = eos
    return sft_args


def ensure_hf_base(base_model: str) -> None:
    """Reject a bitsandbytes 4-bit checkpoint on the hf backend (it needs CUDA to load)."""
    lowered = base_model.lower()
    if "bnb" in lowered or "4bit" in lowered:
        raise ValueError(
            f"hf backend needs a full-precision base (e.g. Qwen/Qwen3-1.7B), got {base_model!r}. "
            "Use configs/train.mps.yaml or set base_model to a non-quantized repo."
        )


def train(cfg: dict, smoke: bool = False, output_dir: str | None = None) -> str:
    """Run QLoRA/LoRA training; returns the output adapter dir.

    ``output_dir`` overrides ``cfg["output_dir"]`` without editing the config, so successive
    data iterations (v1 -> v2) can be saved side by side while the training config stays
    byte-identical (Day-4 rule S4.4: only the data changes).
    """
    from trl import SFTConfig, SFTTrainer

    from src.common.device import detect_backend, pick_device
    from src.train.dataset import build_sft_dataset

    backend = detect_backend(cfg.get("backend"))
    peft_config = None

    if backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["base_model"],
            max_seq_length=cfg["max_seq_len"],
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=cfg["seed"],
        )
    else:  # hf / MPS
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ensure_hf_base(cfg["base_model"])
        dtype = getattr(torch, cfg.get("dtype", "float16"))
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype)
        model.to(pick_device())
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required for grad-checkpointing + PEFT
        peft_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    ds = build_sft_dataset(cfg["train_path"])  # prompt-completion format
    epochs = 1 if smoke else cfg["num_train_epochs"]
    if smoke:
        ds = ds.select(range(min(len(ds), 50)))
    output_dir = (output_dir or cfg["output_dir"]) + ("-smoke" if smoke else "")

    settings = training_backend_settings(backend)
    if backend == "unsloth":
        import torch

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        # Explicit fp16/bf16 from hardware so a T4 (no bf16) dodges the trainer's CUDA bf16 default.
        settings["sft_extra"] = {**settings["sft_extra"], **select_amp_flags(bf16_ok)}

    # Modern TRL: completion-only masking is built into SFTConfig (no data collator).
    sft_args = SFTConfig(
        completion_only_loss=True,  # loss on the tagged completion only
        max_length=cfg["max_seq_len"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=epochs,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=cfg["weight_decay"],
        seed=cfg["seed"],
        logging_steps=5,
        optim=settings["optim"],
        output_dir=output_dir,
        report_to="none",
        **settings["sft_extra"],
    )
    # Beat Unsloth's out-of-vocab '<EOS_TOKEN>' placeholder (rejected by newer SFTTrainer).
    coerce_sft_eos_token(sft_args, tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        peft_config=peft_config,  # None on unsloth (already wrapped); LoraConfig on hf
        args=sft_args,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)  # saves the LoRA adapter (both backends)
    tokenizer.save_pretrained(output_dir)
    print(f"[{backend}] saved adapter -> {output_dir}")
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--smoke", action="store_true", help="1-epoch tiny run to test the loop")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="override cfg output_dir (keeps the config byte-identical across data iterations)",
    )
    args = ap.parse_args()
    train(load_config(args.config), smoke=args.smoke, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
