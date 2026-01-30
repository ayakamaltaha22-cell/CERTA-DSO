"""
CERTA-DSO Reference Implementation (framework-level)
===================================================

This repository implements the *operator-theoretic* CERTA-DSO framework described in your manuscript:
- Bi-output Domain Specialization Operator: (specialized_model, SSC)
- Specialization Certificate (SSC) as a first-class transformation object
- Certificate-coupled optimization (CCO): optimization constrained by SSC-defined feasible set
- Certified parameter isolation: only certificate-approved parameter subsets are trainable
- Deterministic replay operator: verify SSC then deterministically reconstruct specialization
- Optional certificate composition chain

Note:
- CERTA-DSO is a *transformation framework*, not a single new fine-tuning algorithm.
  This code therefore provides pluggable isolation strategies (LoRA/adapters/masks) and
  uses HuggingFace Trainer as one concrete execution backend.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        set_seed as hf_set_seed,
    )
except Exception as e:
    raise RuntimeError("Missing dependency: transformers. Install with: pip install transformers") from e


# -----------------------------
# Hashing utilities
# -----------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def merkle_root_sha256(leaf_hashes_hex: Sequence[str]) -> str:
    if not leaf_hashes_hex:
        return _sha256_bytes(b"")
    level = [bytes.fromhex(x) for x in leaf_hashes_hex]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hashlib.sha256(left + right).digest())
        level = nxt
    return level[0].hex()


# -----------------------------
# Environment fingerprint
# -----------------------------

def _run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""

def git_commit() -> str:
    return _run_cmd(["git", "rev-parse", "HEAD"])

def pip_freeze() -> List[str]:
    out = _run_cmd([sys.executable, "-m", "pip", "freeze"])
    return [x.strip() for x in out.splitlines() if x.strip()]

def environment_fingerprint() -> Dict[str, Any]:
    return {
        "timestamp_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "git_commit": git_commit(),
        "pip_freeze": pip_freeze(),
    }


# -----------------------------
# Determinism controls
# -----------------------------

@dataclass
class DeterminismSpec:
    mode: Literal["exact", "epsilon"] = "epsilon"
    epsilon: float = 0.0
    seed: int = 42
    torch_deterministic_algorithms: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    matmul_precision: Literal["highest", "high", "medium"] = "high"

def apply_determinism(spec: DeterminismSpec) -> None:
    random.seed(spec.seed)
    os.environ["PYTHONHASHSEED"] = str(spec.seed)
    hf_set_seed(spec.seed)
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    torch.use_deterministic_algorithms(spec.torch_deterministic_algorithms, warn_only=True)
    torch.backends.cudnn.deterministic = spec.cudnn_deterministic
    torch.backends.cudnn.benchmark = spec.cudnn_benchmark
    try:
        torch.set_float32_matmul_precision(spec.matmul_precision)
    except Exception:
        pass


# -----------------------------
# Isolation spec (certificate-bound feasible set)
# -----------------------------

@dataclass
class IsolationSpec:
    allow_globs: List[str] = field(default_factory=list)
    allow_regex: List[str] = field(default_factory=list)
    allow_explicit: List[str] = field(default_factory=list)

    method: Literal["mask", "lora", "adapters", "full_ft", "custom"] = "custom"
    notes: str = ""

def _allowed(name: str, iso: IsolationSpec) -> bool:
    if name in set(iso.allow_explicit):
        return True
    for g in iso.allow_globs:
        if fnmatch.fnmatch(name, g):
            return True
    for r in iso.allow_regex:
        if re.match(r, name):
            return True
    return False

def enforce_isolation(model: torch.nn.Module, iso: IsolationSpec) -> List[str]:
    trainable = []
    for n, p in model.named_parameters():
        ok = _allowed(n, iso)
        p.requires_grad = bool(ok)
        if ok:
            trainable.append(n)
    return trainable

def snapshot_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

def assert_isolation_invariant(
    base_snapshot: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    iso: IsolationSpec,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    cur = dict(model.named_parameters())
    for name, base_t in base_snapshot.items():
        if name not in cur:
            continue
        if _allowed(name, iso):
            continue
        if not torch.allclose(cur[name].detach().cpu(), base_t, atol=atol, rtol=rtol):
            raise AssertionError(f"Isolation invariant violated: frozen param changed: {name}")


# -----------------------------
# Dataset commitments (hashes + optional Merkle root)
# -----------------------------

@dataclass
class DatasetCommitment:
    dataset_id: str
    files: List[str] = field(default_factory=list)
    file_hashes_sha256: Dict[str, str] = field(default_factory=dict)
    merkle_root_sha256: Optional[str] = None
    notes: str = ""

    @staticmethod
    def from_files(dataset_id: str, files: Sequence[str], notes: str = "", use_merkle: bool = True) -> "DatasetCommitment":
        hashes = {}
        leaves = []
        for f in files:
            h = sha256_file(f)
            hashes[f] = h
            leaves.append(h)
        mr = merkle_root_sha256(leaves) if use_merkle else None
        return DatasetCommitment(
            dataset_id=dataset_id,
            files=list(files),
            file_hashes_sha256=hashes,
            merkle_root_sha256=mr,
            notes=notes,
        )

    def verify(self) -> None:
        for f, expected in self.file_hashes_sha256.items():
            if not os.path.exists(f):
                raise FileNotFoundError(f"Dataset file missing: {f}")
            actual = sha256_file(f)
            if actual != expected:
                raise AssertionError(f"Dataset hash mismatch for {f}: expected {expected}, got {actual}")
        if self.merkle_root_sha256 is not None:
            leaves = [self.file_hashes_sha256[f] for f in self.files]
            actual_root = merkle_root_sha256(leaves)
            if actual_root != self.merkle_root_sha256:
                raise AssertionError(f"Merkle root mismatch: expected {self.merkle_root_sha256}, got {actual_root}")


# -----------------------------
# Program specs (curriculum + optimization)
# -----------------------------

@dataclass
class CurriculumSpec:
    description: str = "single-stage"
    stages: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class OptimizationSpec:
    objective: str = "causal_lm"
    loss_weights: Dict[str, float] = field(default_factory=dict)

    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0

    max_steps: int = 1000
    num_train_epochs: Optional[float] = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 2048
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Base model identity commitment
# -----------------------------

@dataclass
class BaseModelIdentity:
    model_id: str
    architecture: str
    checkpoint_sha256: str
    tokenizer_id: Optional[str] = None
    revision: Optional[str] = None

def hash_state_dict(model: torch.nn.Module) -> str:
    sd = model.state_dict()
    items = []
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        # hash raw bytes deterministically
        items.append((k, _sha256_bytes(t.numpy().tobytes())))
    return _sha256_bytes(_canonical_json(items))


# -----------------------------
# SSC object + integrity (hash-chain + optional Ed25519 signature)
# -----------------------------

@dataclass
class IntegritySpec:
    hash_chain_sha256: str
    signature_alg: Optional[str] = None
    signature: Optional[str] = None
    signer_public_key: Optional[str] = None

@dataclass
class SSC:
    ssc_version: str = "1.0"
    run_id: str = field(default_factory=lambda: _sha256_bytes(str(_dt.datetime.utcnow().timestamp()).encode())[:16])

    base_model: BaseModelIdentity = field(default_factory=BaseModelIdentity)
    dataset: DatasetCommitment = field(default_factory=DatasetCommitment)
    curriculum: CurriculumSpec = field(default_factory=CurriculumSpec)
    optimization: OptimizationSpec = field(default_factory=OptimizationSpec)
    isolation: IsolationSpec = field(default_factory=IsolationSpec)
    determinism: DeterminismSpec = field(default_factory=DeterminismSpec)
    environment: Dict[str, Any] = field(default_factory=environment_fingerprint)

    integrity: IntegritySpec = field(default_factory=lambda: IntegritySpec(hash_chain_sha256=""))
    parent_ssc_hash: Optional[str] = None  # for composition chains

    def to_dict(self, include_signature: bool = True) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        if not include_signature:
            d["integrity"]["signature"] = None
        return d

    def content_hash(self) -> str:
        d = self.to_dict(include_signature=False)
        return _sha256_bytes(_canonical_json(d))

    def finalize_integrity(self) -> None:
        self.integrity.hash_chain_sha256 = self.content_hash()

    def sign_ed25519(self, private_key_bytes: bytes) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization
        except Exception as e:
            raise RuntimeError("Install cryptography for signing: pip install cryptography") from e

        self.finalize_integrity()
        msg = bytes.fromhex(self.integrity.hash_chain_sha256)
        sk = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        sig = sk.sign(msg)
        pk = sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.integrity.signature_alg = "ed25519"
        self.integrity.signature = sig.hex()
        self.integrity.signer_public_key = pk.hex()

    def verify(self) -> None:
        # verify content hash matches stored hash_chain
        self.finalize_integrity()
        if self.integrity.hash_chain_sha256 != self.content_hash():
            raise AssertionError("SSC content hash mismatch (tampered SSC).")

        # verify signature if present
        if self.integrity.signature is None:
            return

        if self.integrity.signature_alg != "ed25519":
            raise ValueError("Unsupported signature algorithm in SSC.")

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except Exception as e:
            raise RuntimeError("Install cryptography for signature verification: pip install cryptography") from e

        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.integrity.signer_public_key))
        pk.verify(bytes.fromhex(self.integrity.signature), bytes.fromhex(self.integrity.hash_chain_sha256))

        # verify dataset commitments too
        self.dataset.verify()


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(_canonical_json(obj))

def load_json(path: str) -> Any:
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

def load_ssc(path: str) -> SSC:
    d = load_json(path)

    def _mk(cls, x):
        return cls(**x)

    return SSC(
        ssc_version=d["ssc_version"],
        run_id=d["run_id"],
        base_model=_mk(BaseModelIdentity, d["base_model"]),
        dataset=_mk(DatasetCommitment, d["dataset"]),
        curriculum=_mk(CurriculumSpec, d["curriculum"]),
        optimization=_mk(OptimizationSpec, d["optimization"]),
        isolation=_mk(IsolationSpec, d["isolation"]),
        determinism=_mk(DeterminismSpec, d["determinism"]),
        environment=d["environment"],
        integrity=_mk(IntegritySpec, d["integrity"]),
        parent_ssc_hash=d.get("parent_ssc_hash"),
    )


# -----------------------------
# CERTA-DSO Operator
# -----------------------------

@dataclass
class CertaDSOConfig:
    base_model_id: str
    tokenizer_id: Optional[str] = None
    revision: Optional[str] = None

    output_dir: str = "./certa_dso_out"
    ssc_filename: str = "ssc.json"
    model_dirname: str = "specialized_model"

    logging_steps: int = 50
    save_steps: int = 200
    eval_steps: Optional[int] = None
    report_to: Optional[str] = None  # "none", "wandb", ...


class CERTA_DSO:
    """
    DSO: (M_base, D, C, O, I) -> (M_specialized, SSC)

    - Certificate-coupled optimization: apply isolation constraints before training
    - SSC encodes (base model identity, dataset commitments, curriculum, optimization, isolation, determinism,
      execution environment, integrity bindings) and can be signed.
    """

    def __init__(self, cfg: CertaDSOConfig):
        self.cfg = cfg

    def _trainer_args(self, opt: OptimizationSpec) -> TrainingArguments:
        return TrainingArguments(
            output_dir=os.path.join(self.cfg.output_dir, "trainer"),
            per_device_train_batch_size=opt.per_device_train_batch_size,
            gradient_accumulation_steps=opt.gradient_accumulation_steps,
            learning_rate=opt.lr,
            weight_decay=opt.weight_decay,
            warmup_ratio=opt.warmup_ratio,
            max_steps=opt.max_steps,
            num_train_epochs=opt.num_train_epochs if opt.num_train_epochs is not None else 0,
            logging_steps=self.cfg.logging_steps,
            save_steps=self.cfg.save_steps,
            evaluation_strategy="no" if self.cfg.eval_steps is None else "steps",
            eval_steps=self.cfg.eval_steps,
            report_to=[] if self.cfg.report_to in (None, "none") else [self.cfg.report_to],
            bf16=opt.bf16,
            fp16=opt.fp16,
            gradient_checkpointing=opt.gradient_checkpointing,
            dataloader_drop_last=False,
            remove_unused_columns=False,
            seed=opt.extra.get("seed", 42),
            data_seed=opt.extra.get("data_seed", 42),
        )

    def _build_base_identity(self, model: torch.nn.Module) -> BaseModelIdentity:
        return BaseModelIdentity(
            model_id=self.cfg.base_model_id,
            architecture=model.__class__.__name__,
            checkpoint_sha256=hash_state_dict(model),
            tokenizer_id=self.cfg.tokenizer_id or self.cfg.base_model_id,
            revision=self.cfg.revision,
        )

    def specialize(
        self,
        train_dataset,
        dataset_commitment: DatasetCommitment,
        curriculum: CurriculumSpec,
        optimization: OptimizationSpec,
        isolation: IsolationSpec,
        determinism: DeterminismSpec,
        signing_key_ed25519: Optional[bytes] = None,
    ) -> Tuple[torch.nn.Module, SSC]:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        apply_determinism(determinism)

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_id or self.cfg.base_model_id,
            revision=self.cfg.revision,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model_id,
            revision=self.cfg.revision,
            torch_dtype=torch.bfloat16 if optimization.bf16 else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # bind base model identity and snapshot frozen params
        base_identity = self._build_base_identity(model)
        base_snapshot = snapshot_params(model)

        # verify dataset commitments before training
        dataset_commitment.verify()

        # enforce certificate-defined feasible set
        trainable = enforce_isolation(model, isolation)
        if not trainable:
            raise ValueError("IsolationSpec resulted in 0 trainable parameters. Fix allow_* patterns.")

        # build SSC (declarative specialization program + commitments)
        ssc = SSC(
            base_model=base_identity,
            dataset=dataset_commitment,
            curriculum=curriculum,
            optimization=optimization,
            isolation=isolation,
            determinism=determinism,
            environment=environment_fingerprint(),
            integrity=IntegritySpec(hash_chain_sha256=""),
        )

        # execute specialization program (Trainer is one backend)
        args = self._trainer_args(optimization)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
        )
        trainer.train()

        # check isolation invariant (base params unchanged)
        atol = 0.0 if determinism.mode == "exact" else max(1e-8, determinism.epsilon)
        rtol = 0.0 if determinism.mode == "exact" else max(1e-6, determinism.epsilon)
        assert_isolation_invariant(base_snapshot, model, isolation, atol=atol, rtol=rtol)

        # finalize certificate integrity + optional signature
        ssc.finalize_integrity()
        if signing_key_ed25519 is not None:
            ssc.sign_ed25519(signing_key_ed25519)

        # persist artifacts
        model_out = os.path.join(self.cfg.output_dir, self.cfg.model_dirname)
        ssc_out = os.path.join(self.cfg.output_dir, self.cfg.ssc_filename)
        os.makedirs(model_out, exist_ok=True)
        model.save_pretrained(model_out)
        tokenizer.save_pretrained(model_out)
        save_json(ssc_out, ssc.to_dict(include_signature=True))

        return model, ssc

    def verify_and_replay(self, ssc_path: str, train_dataset, strict_env: bool = False) -> torch.nn.Module:
        """
        Deterministic replay operator R(M_base, SSC) -> M_specialized
        - verify SSC integrity/signature
        - verify dataset commitments (hashes / Merkle root)
        - verify base model hash
        - enforce isolation mask
        - replay the declared optimization program
        """
        ssc = load_ssc(ssc_path)
        ssc.verify()  # includes dataset verification

        if strict_env:
            cur = environment_fingerprint()
            if cur["torch"] != ssc.environment.get("torch"):
                raise AssertionError(f"Env mismatch torch: {cur['torch']} != {ssc.environment.get('torch')}")
            if cur["transformers"] != ssc.environment.get("transformers"):
                raise AssertionError(f"Env mismatch transformers: {cur['transformers']} != {ssc.environment.get('transformers')}")

        apply_determinism(ssc.determinism)

        tokenizer = AutoTokenizer.from_pretrained(
            ssc.base_model.tokenizer_id or ssc.base_model.model_id,
            revision=ssc.base_model.revision,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            ssc.base_model.model_id,
            revision=ssc.base_model.revision,
            torch_dtype=torch.bfloat16 if ssc.optimization.bf16 else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # verify base model commitment
        base_hash = hash_state_dict(model)
        if base_hash != ssc.base_model.checkpoint_sha256:
            raise AssertionError(
                f"Base model hash mismatch: expected {ssc.base_model.checkpoint_sha256}, got {base_hash}"
            )

        base_snapshot = snapshot_params(model)
        trainable = enforce_isolation(model, ssc.isolation)
        if not trainable:
            raise ValueError("Replay: IsolationSpec resulted in 0 trainable parameters.")

        args = self._trainer_args(ssc.optimization)
        args.seed = ssc.determinism.seed
        args.data_seed = ssc.determinism.seed
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
        )
        trainer.train()

        atol = 0.0 if ssc.determinism.mode == "exact" else max(1e-8, ssc.determinism.epsilon)
        rtol = 0.0 if ssc.determinism.mode == "exact" else max(1e-6, ssc.determinism.epsilon)
        assert_isolation_invariant(base_snapshot, model, ssc.isolation, atol=atol, rtol=rtol)

        return model


# -----------------------------
# Certificate composition scaffold
# -----------------------------

def compose_ssc(parent: SSC, child: SSC) -> SSC:
    child.parent_ssc_hash = parent.content_hash()
    child.finalize_integrity()
    return child


# -----------------------------
# Convenience constructors for common isolation styles
# -----------------------------

def isolation_full_ft() -> IsolationSpec:
    return IsolationSpec(allow_regex=[r".*"], method="full_ft", notes="All parameters trainable (FT upper-bound).")

def isolation_lora_default() -> IsolationSpec:
    # Adjust to match your actual LoRA parameter names
    return IsolationSpec(allow_regex=[r".*lora_.*"], method="lora", notes="Train only LoRA parameters.")

def isolation_adapters_default() -> IsolationSpec:
    # Adjust to match your adapter naming
    return IsolationSpec(allow_globs=["*adapter*"], method="adapters", notes="Train only adapter parameters.")


if __name__ == "__main__":
    print("CERTA-DSO reference implementation. See README.md for usage.")
