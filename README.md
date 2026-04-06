# CERTA-DSO Reference Implementation

This repository provides a full **reference implementation** of **CERTA-DSO**:
**A Certified Bi-Output Transformation Framework for Domain-Specialized LLMs**.

It operationalizes the manuscript concepts and algorithms as runnable software:

- bi-output specialization operator
- certificate-coupled optimization
- certified parameter isolation
- Specialization Support Certificate (SSC)
- cryptographic binding and verification
- deterministic replay
- certificate composition
- rollback / inversion
- audit logging and environment capture

## Important scope note

This codebase is a **complete framework implementation** of CERTA-DSO, including
the end-to-end pipeline, certificate machinery, replay, composition, rollback,
and a runnable PyTorch training path. It is designed to work with:

1. **toy/demo models out of the box**, and
2. **real Hugging Face / PyTorch models** through the provided integration hooks.

Because pretrained checkpoints, domain corpora, and private keys are external
assets, they are **not bundled** in this repository.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick demo

```bash
python certa_dso/examples/demo_run.py
```

## CLI

```bash
python -m certa_dso.cli run --work-dir outputs/demo
python -m certa_dso.cli verify --work-dir outputs/demo
```
