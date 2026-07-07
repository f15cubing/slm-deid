"""QLoRA SFT of Qwen3-1.7B for the NAME-judgment core (Day 3 real run; Day 2 smoke).

GPU/Colab only (unsloth + trl). Reads configs/train.yaml. Completion-only, non-thinking, r=32/a=32,
lr=2e-4, seq 2048 by default. ``--smoke`` shrinks to a 1-epoch pass over a tiny slice to prove the
generate -> train -> eval loop end to end (spec S2.11).

    python -m src.train.qlora --config configs/train.yaml
    python -m src.train.qlora --config configs/train.yaml --smoke
"""

from __future__ import annotations

import argparse


def load_config(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cfg: dict, smoke: bool = False) -> str:
    """Run QLoRA training; returns the output adapter dir."""
    from datasets import Dataset
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from src.train.dataset import RESPONSE_TEMPLATE, build_sft_dataset

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

    ds: Dataset = build_sft_dataset(cfg["train_path"], tokenizer)
    epochs = 1 if smoke else cfg["num_train_epochs"]
    if smoke:
        ds = ds.select(range(min(len(ds), 50)))
    output_dir = cfg["output_dir"] + ("-smoke" if smoke else "")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=collator,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_len"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_ratio=cfg["warmup_ratio"],
            num_train_epochs=epochs,
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            weight_decay=cfg["weight_decay"],
            seed=cfg["seed"],
            logging_steps=5,
            optim="adamw_8bit",
            output_dir=output_dir,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved adapter -> {output_dir}")
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--smoke", action="store_true", help="1-epoch tiny run to test the loop")
    args = ap.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
