from copy import deepcopy

from certa_dso import generate_keypair, make_body, replay_accept, seal_ssc, verify_pair, verify_ssc_integrity


def sample_body():
    return make_body(
        base_model={"model_id": "base"},
        dataset={"dataset_id": "domain-v1"},
        curriculum={"order": "fixed"},
        optimizer={"name": "AdamW"},
        isolation={"authorized_parameter_names": ["adapter.weight"]},
        randomness={"seed": 7},
        environment={"python": "3.10"},
        replay_policy={"mode": "exact"},
        final_model={"commitment": "model-digest"},
    )


def test_sealed_certificate_verifies():
    private_key, public_key = generate_keypair()
    certificate = seal_ssc(sample_body(), private_key)
    assert verify_ssc_integrity(certificate, public_key)


def test_change_to_sealed_field_is_detected():
    private_key, public_key = generate_keypair()
    certificate = seal_ssc(sample_body(), private_key)
    tampered = deepcopy(certificate)
    tampered["body"]["dataset"]["dataset_id"] = "other-data"
    assert not verify_ssc_integrity(tampered, public_key)


def test_model_commitment_is_part_of_pair_validity():
    private_key, public_key = generate_keypair()
    certificate = seal_ssc(sample_body(), private_key)
    assert verify_pair(certificate, public_key, "model-digest").accepted
    assert verify_pair(certificate, public_key, "different-model").code == "MODEL_COMMITMENT_MISMATCH"


def test_exact_and_tolerance_replay():
    assert replay_accept("abc", "abc", {"mode": "exact"}).accepted
    result = replay_accept(1.0, 1.0001, {"mode": "tolerance", "epsilon": 0.001}, distance=lambda a, b: abs(a - b))
    assert result.accepted
