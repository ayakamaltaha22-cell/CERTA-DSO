"""Build and verify a small SSC without loading a language model."""
from certa_dso import environment_fingerprint, generate_keypair, make_body, seal_ssc, verify_pair

private_key, public_key = generate_keypair()
body = make_body(
    base_model={"model_id": "example-base"},
    dataset={"dataset_id": "example-domain"},
    curriculum={"instruction_order": "fixed", "steps": 100},
    optimizer={"name": "AdamW", "learning_rate": 2e-4},
    isolation={"authorized_parameter_names": ["adapter.weight"]},
    randomness={"seed": 42},
    environment=environment_fingerprint(),
    replay_policy={"mode": "exact"},
    final_model={"commitment": "replace-with-model-commitment"},
)
certificate = seal_ssc(body, private_key)
result = verify_pair(certificate, public_key, "replace-with-model-commitment")
print(certificate["root"])
print(result.code)
