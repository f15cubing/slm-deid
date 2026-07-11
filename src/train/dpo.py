"""DPO on top of the SFT adapter — Day-5 stretch rung 1.

Preference optimization (TRL ``DPOTrainer``) initialized from the canonical SFT LoRA (gpt551),
trained on the hybrid preference pairs from :mod:`src.train.prefs`. The question it answers:
*does DPO sharpen spec-adherence (precision / over-tag / consistency) BEYOND SFT alone?*

Two backends, same auto-selection as SFT (:mod:`src.train.qlora`), and the shared backend helpers
(optimizer, T4/A100 AMP flags, EOS-token coercion) are reused from there so the DPO path stays
byte-consistent with the SFT path:

- ``unsloth`` (CUDA/Colab): 4-bit QLoRA-DPO. ``PatchDPOTrainer()`` MUST run before the trainer is
  built (Unsloth patches TRL's DPO memory path). The SFT adapter is loaded as the trainable
  policy; the **reference model is the same weights with the adapter disabled** (``ref_model=None``
  + a PEFT policy → TRL uses the base as the reference). That regularizes toward the base model,
  not toward SFT — the standard, memory-cheap QLoRA-DPO choice. Documented in ``configs/dpo.yaml``.
- ``hf`` (Apple Silicon/CPU): plain LoRA-DPO for a local smoke of the loop.

These new knobs (``beta``, the DPO ``learning_rate``, ``num_train_epochs``) are genuinely new —
there is no prior DPO config to freeze — and must not be tuned to paper over a data problem
(the Day-4 ceiling applies to the *SFT* config, which is untouched here).

    python -m src.train.dpo --config configs/dpo.yaml                 # Colab / CUDA
    python -m src.train.dpo --config configs/dpo.yaml --smoke         # 1-epoch loop test

The heavy model/trainer path needs a GPU; the config + kwargs assembly below is pure and tested in
``tests/test_dpo_backend.py``.
"""

from __future__ import annotations

import argparse

# Reuse the SFT backend decisions verbatim so DPO stays consistent with the frozen SFT path.
from src.train.qlora import (
    coerce_sft_eos_token,
    ensure_hf_base,
    load_config,
    select_amp_flags,
    training_backend_settings,
)


def dpo_config_kwargs(
    cfg: dict, *, epochs: int, optim: str, amp_extra: dict, output_dir: str
) -> dict:
    """Assemble the DPOConfig kwargs from the config (pure; no TRL import).

    Kept separate from construction so the mapping (beta / lr / epochs / AMP flags) is unit-tested
    without a GPU or the trl dependency.
    """
    return {
        "beta": cfg["beta"],
        "max_length": cfg["max_seq_len"],
        "max_prompt_length": cfg.get("max_prompt_len", cfg["max_seq_len"] // 2),
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "warmup_ratio": cfg["warmup_ratio"],
        "num_train_epochs": epochs,
        "learning_rate": cfg["learning_rate"],
        "lr_scheduler_type": cfg["lr_scheduler_type"],
        "weight_decay": cfg["weight_decay"],
        "seed": cfg["seed"],
        "logging_steps": 5,
        "optim": optim,
        "output_dir": output_dir,
        "report_to": "none",
        **amp_extra,
    }


def train_dpo(cfg: dict, smoke: bool = False, output_dir: str | None = None) -> str:
    """Run DPO on top of the SFT adapter; return the output adapter dir.

    ``cfg['sft_adapter']`` is the SFT LoRA to initialize the policy from (gpt551 on Colab, loaded
    from Drive). ``output_dir`` overrides ``cfg['output_dir']`` without editing the config.
    """
    from datasets import Dataset  # noqa: F401  (ensures the dep is present early on Colab)
    from trl import DPOConfig, DPOTrainer

    from src.common.device import detect_backend, pick_device
    from src.train.prefs import build_trl_dataset

    backend = detect_backend(cfg.get("backend"))
    sft_adapter = cfg["sft_adapter"]
    peft_config = None
    ref_model = None  # PEFT policy + ref_model=None → reference is the adapter-disabled base

    if backend == "unsloth":
        from unsloth import FastLanguageModel, PatchDPOTrainer

        PatchDPOTrainer()  # MUST precede DPOTrainer construction (patches TRL's DPO path)
        # Loading an adapter dir attaches the SFT LoRA to the 4-bit base as the trainable policy.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_adapter,
            max_seq_length=cfg["max_seq_len"],
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_training(model)
    else:  # hf / MPS
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ensure_hf_base(cfg["base_model"])
        dtype = getattr(torch, cfg.get("dtype", "float16"))
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype)
        # Continue the SFT adapter as a trainable policy; adapter-disabled = the reference.
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
        model.to(pick_device())

    # Build the preference dataset from the pairs JSONL the notebook wrote (id/input/chosen/...).
    pairs = _load_pairs(cfg["pairs_path"])
    if smoke:
        pairs = pairs[: min(len(pairs), 50)]
    ds: Dataset = build_trl_dataset(pairs)
    epochs = 1 if smoke else cfg["num_train_epochs"]
    output_dir = (output_dir or cfg["output_dir"]) + ("-smoke" if smoke else "")

    settings = training_backend_settings(backend)
    amp_extra = dict(settings["sft_extra"])
    if backend == "unsloth":
        import torch

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_extra = {**amp_extra, **select_amp_flags(bf16_ok)}

    dpo_args = DPOConfig(
        **dpo_config_kwargs(
            cfg,
            epochs=epochs,
            optim=settings["optim"],
            amp_extra=amp_extra,
            output_dir=output_dir,
        )
    )
    coerce_sft_eos_token(dpo_args, tokenizer)  # no-op unless the DPOConfig carries an eos field

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{backend}] saved DPO adapter -> {output_dir}")
    return output_dir


def _load_pairs(path: str):
    """Read the pairs JSONL (as written by :func:`src.train.prefs.dump_pairs`) back into Pairs."""
    import json

    from src.train.prefs import Pair

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pairs.append(
                Pair(
                    id=d["id"],
                    category=d["category"],
                    input=d["input"],
                    chosen=d["chosen"],
                    rejected=d["rejected"],
                    strategy=d["strategy"],
                )
            )
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--smoke", action="store_true", help="1-epoch tiny run to test the loop")
    ap.add_argument("--output-dir", default=None, help="override cfg output_dir")
    args = ap.parse_args()
    train_dpo(load_config(args.config), smoke=args.smoke, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
