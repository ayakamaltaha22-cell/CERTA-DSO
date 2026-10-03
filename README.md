# CERTA-DSO

Reference code for **CERTA-DSO: Certificate-Defined, Replayable Domain Specialization for Large Language Models**.

CERTA-DSO wraps a model-specialization backend with a certificate contract. A successful run produces the specialized model and a Specialization Support Certificate (SSC). The SSC commits the base model, data, curriculum, optimizer settings, authorized update region, randomness controls, execution environment, replay policy, and final model state.

## What is implemented

The package contains the certificate functions needed by the framework: canonical JSON encoding, SHA-256 commitments, the Transformation Hash Tree, Ed25519 signatures, model/SSC verification, parameter-isolation checks, and exact or tolerance-bounded replay acceptance. `hf_backend.py` shows how the same checks can be placed around a Hugging Face/PEFT training function.

The implementation follows the certificate scope described in the paper. It verifies committed artifacts and conditions; it does not claim to establish dataset quality, fairness, absence of poisoning, host security, or an optimizer trajectory that was not recorded as verifiable evidence.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

For the Hugging Face/PEFT example:

```bash
pip install -e '.[hf]'
```

For development and tests:

```bash
pip install -e '.[dev]'
pytest -q
```

## Repository layout

```text
configs/                 example SSC/backend settings
examples/minimal.py      certificate construction and verification
src/certa_dso/core.py    commitments, sealing, verification and replay rules
src/certa_dso/hf_backend.py  PEFT integration
 tests/                   unit tests for the certificate path
```

## Basic use

```python
from certa_dso import generate_keypair, make_body, seal_ssc, verify_pair

private_key, public_key = generate_keypair()
body = make_body(
    base_model={"model_id": "my-base-model"},
    dataset={"dataset_id": "domain-data-v1"},
    curriculum={"instruction_order": "fixed"},
    optimizer={"name": "AdamW", "learning_rate": 2e-4},
    isolation={"authorized_parameter_names": ["adapter.weight"]},
    randomness={"seed": 42},
    environment={"python": "3.10", "cuda": "12.1"},
    replay_policy={"mode": "exact"},
    final_model={"commitment": "MODEL_SHA256"},
)
ssc = seal_ssc(body, private_key)
print(verify_pair(ssc, public_key, "MODEL_SHA256"))
```

For an actual specialization run, use `run_certa_dso` in `src/certa_dso/hf_backend.py`. The supplied training function performs the backend optimization. CERTA-DSO records the authorized trainable region before training, checks the resulting state for changes outside that region, commits the final model, and seals the SSC.

## Experimental status

The numerical values currently shown in the manuscript's evaluation tables are reporting templates rather than executed measurements. This repository implements the proposed framework and should not be used to present those placeholder values as measured results.

## License

A software license has not yet been selected. Add the intended license before public release.
