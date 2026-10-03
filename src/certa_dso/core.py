from __future__ import annotations
import base64
import hashlib
import json
import os
import platform
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

FAILURES = {
    "BASE_MODEL_MISMATCH", "DATASET_HASH_MISMATCH", "CURRICULUM_MISMATCH",
    "SSC_INTEGRITY_FAILURE", "ISOLATION_VIOLATION", "OPTIMIZER_CONTRACT_FAILURE",
    "RANDOMNESS_MISMATCH", "ENVIRONMENT_MISMATCH",
    "MODEL_COMMITMENT_MISMATCH", "REPLAY_FAILURE",
}

def canonical(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def commit_obj(x: Any) -> str:
    return sha256(canonical(x))
def commit_file(path: str|Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def merkle_root(leaves: list[str]) -> str:
    if not leaves: return sha256(b"")
    level=[bytes.fromhex(x) for x in leaves]
    while len(level)>1:
        if len(level)%2: level.append(level[-1])
        level=[hashlib.sha256(level[i]+level[i+1]).digest() for i in range(0,len(level),2)]
    return level[0].hex()

def environment_fingerprint(extra: dict|None=None) -> dict:
    d={"python":sys.version.split()[0],"platform":platform.platform(),"machine":platform.machine()}
    if extra: d.update(extra)
    return d

def set_seed(seed:int, deterministic:bool=True):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except ImportError: pass

def generate_keypair():
    sk=Ed25519PrivateKey.generate(); return sk, sk.public_key()

def public_key_b64(pk:Ed25519PublicKey)->str:
    from cryptography.hazmat.primitives import serialization
    return base64.b64encode(pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()

def sign_root(sk:Ed25519PrivateKey, root:str)->str:
    return base64.b64encode(sk.sign(bytes.fromhex(root))).decode()
def verify_sig(pk:Ed25519PublicKey, root:str, sig:str)->bool:
    try: pk.verify(base64.b64decode(sig),bytes.fromhex(root)); return True
    except Exception: return False

FIELD_ORDER=["metadata","base_model","dataset","curriculum","optimizer","isolation","randomness","environment","replay_policy","final_model"]

def seal_ssc(body:dict, sk:Ed25519PrivateKey, key_id:str="local-ed25519") -> dict:
    missing=[k for k in FIELD_ORDER if k not in body]
    if missing: raise ValueError(f"missing SSC fields: {missing}")
    leaves={k:commit_obj(body[k]) for k in FIELD_ORDER}
    root=merkle_root([leaves[k] for k in FIELD_ORDER])
    return {"schema":"CERTA-DSO-SSC/1.0","body":body,"leaf_hashes":leaves,"root":root,
            "signature":{"algorithm":"Ed25519","key_id":key_id,"value":sign_root(sk,root)}}

def verify_ssc_integrity(ssc:dict, pk:Ed25519PublicKey)->bool:
    try:
        leaves={k:commit_obj(ssc["body"][k]) for k in FIELD_ORDER}
        root=merkle_root([leaves[k] for k in FIELD_ORDER])
        return leaves==ssc["leaf_hashes"] and root==ssc["root"] and verify_sig(pk,root,ssc["signature"]["value"])
    except Exception: return False

def state_dict_commit(model)->str:
    import torch
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode()); a=t.detach().cpu().contiguous(); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.numpy().tobytes())
    return h.hexdigest()

def trainable_names(model) -> list[str]:
    return sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
def isolation_commit(authorized:Iterable[str], backend:dict)->dict:
    names=sorted(authorized); return {"authorized_parameter_names":names,"mask_hash":commit_obj(names),"backend":backend}
def check_isolation(before:dict, after:dict, authorized:set[str], atol:float=0.0)->list[str]:
    bad=[]
    for k,b in before.items():
        if k not in after: bad.append(k); continue
        if k not in authorized:
            a=after[k]
            try:
                import torch
                if not torch.allclose(b.detach().cpu(),a.detach().cpu(),atol=atol,rtol=0): bad.append(k)
            except Exception:
                if np.max(np.abs(np.asarray(b)-np.asarray(a)))>atol: bad.append(k)
    return bad

@dataclass
class VerificationResult:
    accepted: bool
    code: str
    details: dict

def verify_pair(ssc:dict, pk:Ed25519PublicKey, model_commitment:str, artifacts:dict[str,Any]|None=None)->VerificationResult:
    if not verify_ssc_integrity(ssc,pk): return VerificationResult(False,"SSC_INTEGRITY_FAILURE",{})
    b=ssc["body"]
    if model_commitment != b["final_model"]["commitment"]: return VerificationResult(False,"MODEL_COMMITMENT_MISMATCH",{})
    artifacts=artifacts or {}
    mapping={"base_model":"BASE_MODEL_MISMATCH","dataset":"DATASET_HASH_MISMATCH","curriculum":"CURRICULUM_MISMATCH","optimizer":"OPTIMIZER_CONTRACT_FAILURE","randomness":"RANDOMNESS_MISMATCH","environment":"ENVIRONMENT_MISMATCH"}
    for k,code in mapping.items():
        if k in artifacts and commit_obj(artifacts[k]) != b[k].get("artifact_commitment", commit_obj(b[k])):
            return VerificationResult(False,code,{"field":k})
    return VerificationResult(True,"ACCEPT",{})

def make_body(*,base_model,dataset,curriculum,optimizer,isolation,randomness,environment,replay_policy,final_model,issuer="CERTA-DSO"):
    return {"metadata":{"certificate_id":str(uuid.uuid4()),"schema_version":"1.0","issuer":issuer,"created_at":datetime.now(timezone.utc).isoformat()},
            "base_model":base_model,"dataset":dataset,"curriculum":curriculum,"optimizer":optimizer,"isolation":isolation,"randomness":randomness,"environment":environment,"replay_policy":replay_policy,"final_model":final_model}

def replay_accept(reference, replayed, policy:dict, distance:Callable[[Any,Any],float]|None=None)->VerificationResult:
    mode=policy["mode"]
    if mode=="exact": ok=reference==replayed
    elif mode=="tolerance":
        if distance is None: raise ValueError("distance required for tolerance replay")
        d=float(distance(reference,replayed)); ok=d<=float(policy["epsilon"])
        if not ok:return VerificationResult(False,"REPLAY_FAILURE",{"distance":d,"epsilon":policy["epsilon"]})
    else: raise ValueError("mode must be exact or tolerance")
    return VerificationResult(ok,"ACCEPT" if ok else "REPLAY_FAILURE",{})
