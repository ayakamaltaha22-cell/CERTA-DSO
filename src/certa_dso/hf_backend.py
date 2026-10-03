"""Hugging Face/PEFT integration for CERTA-DSO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .core import check_isolation, isolation_commit, make_body, seal_ssc, state_dict_commit, trainable_names


def load_lora(model_name: str, target_modules: list[str], r: int = 8, alpha: int = 16, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, config), tokenizer, config


def _snapshot(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def run_certa_dso(model: Any, train_fn: Callable[[Any], None], commitments: dict, private_key, output_dir: str):
    """Run the selected backend, check the certified update region, and seal the SSC."""
    required = {"base_model", "dataset", "curriculum", "optimizer", "isolation_backend", "randomness", "environment", "replay_policy"}
    missing = sorted(required.difference(commitments))
    if missing:
        raise ValueError("missing commitment(s): " + ", ".join(missing))

    authorized = set(trainable_names(model))
    before = _snapshot(model)
    train_fn(model)
    after = _snapshot(model)

    violations = check_isolation(before, after, authorized)
    if violations:
        names = ", ".join(violations[:10])
        raise RuntimeError(f"ISOLATION_VIOLATION: {names}")

    body = make_body(
        base_model=commitments["base_model"],
        dataset=commitments["dataset"],
        curriculum=commitments["curriculum"],
        optimizer=commitments["optimizer"],
        isolation=isolation_commit(authorized, commitments["isolation_backend"]),
        randomness=commitments["randomness"],
        environment=commitments["environment"],
        replay_policy=commitments["replay_policy"],
        final_model={"commitment": state_dict_commit(model)},
    )
    ssc = seal_ssc(body, private_key)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "model")
    (output / "ssc.json").write_text(json.dumps(ssc, indent=2) + "\n", encoding="utf-8")
    return model, ssc
