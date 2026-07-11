"""Pure config/kwargs logic for DPO (src/train/dpo.py).

The model + DPOTrainer path needs a GPU and trl; the *decisions* — how the config maps to
DPOConfig kwargs, the epochs override, the AMP-flag merge, and the max_prompt_len default — are
pure and tested here (mirrors tests/test_qlora_backend.py).
"""

from src.train.dpo import dpo_config_kwargs

_CFG = {
    "beta": 0.1,
    "max_seq_len": 2048,
    "max_prompt_len": 1024,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.05,
    "learning_rate": 5.0e-6,
    "num_train_epochs": 1,
    "lr_scheduler_type": "linear",
    "weight_decay": 0.01,
    "seed": 0,
}


def _kwargs(cfg=None, **over):
    base = dict(
        epochs=cfg.get("num_train_epochs") if cfg else 1,
        optim="adamw_8bit",
        amp_extra={"bf16": True, "fp16": False},
        output_dir="outputs/dpo-test",
    )
    base.update(over)
    return dpo_config_kwargs(cfg or _CFG, **base)


def test_maps_beta_and_dpo_lr():
    k = _kwargs()
    assert k["beta"] == 0.1
    assert k["learning_rate"] == 5.0e-6  # the LOW DPO lr, not SFT's 2e-4
    assert k["optim"] == "adamw_8bit"
    assert k["output_dir"] == "outputs/dpo-test"


def test_epochs_argument_overrides_config():
    # --smoke passes epochs=1 even if the config asks for more.
    cfg = {**_CFG, "num_train_epochs": 3}
    assert _kwargs(cfg, epochs=1)["num_train_epochs"] == 1


def test_amp_flags_are_merged_in():
    k = _kwargs(amp_extra={"fp16": True, "bf16": False})
    assert k["fp16"] is True and k["bf16"] is False


def test_max_prompt_len_defaults_to_half_seq_when_absent():
    cfg = {k: v for k, v in _CFG.items() if k != "max_prompt_len"}
    assert _kwargs(cfg)["max_prompt_length"] == 1024  # 2048 // 2


def test_max_prompt_len_respected_when_present():
    assert _kwargs()["max_prompt_length"] == 1024
