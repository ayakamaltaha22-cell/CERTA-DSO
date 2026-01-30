# CERTA-DSO (Reference Implementation)

This repo implements the CERTA-DSO *framework* as defined in the manuscript: a **bi-output domain specialization operator**
that produces a **specialized model + a Specialization Certificate (SSC)**.

## What’s included

- `certa_dso.py`
  - `CERTA_DSO.specialize(...)` → `(model_specialized, SSC)`
  - `CERTA_DSO.verify_and_replay(...)` → deterministic replay reconstruction from `(base_model, SSC)`
  - SSC structure includes: base model identity, dataset hashes/Merkle root, curriculum spec, optimization spec,
    parameter-isolation specification, randomness control, execution environment fingerprint, and integrity bindings.
  - Optional Ed25519 signature (requires `cryptography`).

## Install

```bash
pip install torch transformers datasets accelerate
# optional (for signing SSC)
pip install cryptography
```

## Minimal usage (template)

You must provide a **tokenized** HF dataset compatible with `Trainer` (i.e., dicts with `input_ids` and `attention_mask`).

```python
from certa_dso import CERTA_DSO, CertaDSOConfig, DatasetCommitment, CurriculumSpec, OptimizationSpec
from certa_dso import isolation_lora_default, DeterminismSpec

# 1) Prepare tokenized train_dataset (HF datasets or torch dataset)
train_dataset = ...

# 2) Dataset commitment (hashes + optional Merkle root)
dc = DatasetCommitment.from_files(
    dataset_id="your-domain-dataset",
    files=["/path/to/train.jsonl"],
    use_merkle=True,
)

# 3) Operator config
cfg = CertaDSOConfig(
    base_model_id="meta-llama/Llama-2-7b-hf",
    output_dir="./out_certa_dso",
)

dso = CERTA_DSO(cfg)

# 4) Program specs
curr = CurriculumSpec(description="single-stage", stages=[])
opt = OptimizationSpec(max_steps=100, lr=1e-4, per_device_train_batch_size=1)
iso = isolation_lora_default()  # or isolation_adapters_default(), or isolation_full_ft()
det = DeterminismSpec(mode="epsilon", epsilon=0.0, seed=42)

model, ssc = dso.specialize(
    train_dataset=train_dataset,
    dataset_commitment=dc,
    curriculum=curr,
    optimization=opt,
    isolation=iso,
    determinism=det,
)

# 5) Deterministic replay (reconstruct from SSC)
replayed_model = dso.verify_and_replay("./out_certa_dso/ssc.json", train_dataset=train_dataset)
```

## Notes

- CERTA-DSO is designed to be **backend-agnostic**. `Trainer` is used here as one concrete backend.
- Adjust `IsolationSpec` patterns to match your actual LoRA/adapter parameter names.
- If you want exact replay, set `DeterminismSpec(mode="exact")` and ensure fully deterministic kernels and fixed environments.
