"""Optional metadata-only full auxiliary checkpoint/model coverage audit.

This does not reload a model or execute repository code. It compares every
local safetensors key, shape, dtype and byte extent with the already loaded
auxiliary state_dict, including both embedding/head names when they alias.
Value-byte preservation is a separate live-hash/raw-matrix audit.
"""
from __future__ import annotations
import math
import hashlib


def validate_metadata(raw, loaded, *, expected_tied):
    if set(raw) != set(loaded):
        raise ValueError("Auxiliary state key coverage differs: missing="+str(sorted(set(raw)-set(loaded)))+
                         "; unexpected="+str(sorted(set(loaded)-set(raw))))
    if not raw:
        raise ValueError("Empty auxiliary state inventory")
    elements=0
    for name,source in raw.items():
        value=loaded[name];shape=tuple(source["shape"])
        if (tuple(value["shape"]) != shape or source["dtype"] != "BF16" or value["dtype"] != "BF16" or
                source["bytes"] != 2*math.prod(shape)):
            raise ValueError("Auxiliary state shape/dtype/byte extent differs: "+name)
        elements+=math.prod(shape)
    a,b="model.embed_tokens.weight","lm_head.weight"
    if a not in loaded or b not in loaded:
        raise ValueError("Both original auxiliary embedding/head state keys are required")
    aliases=loaded[a]["storage"]==loaded[b]["storage"]
    if aliases != expected_tied:
        raise ValueError("Loaded auxiliary state aliases disagree with original raw-matrix tying audit")
    return {"all_original_auxiliary_state_keys_shapes_dtypes_and_extents_covered":True,
        "stored_state_keys":len(raw),"loaded_state_keys":len(loaded),"stored_elements_with_alias_names":elements,
        "embedding_head_names_both_covered":True,"loaded_embedding_head_storage_alias":aliases,
        "scope":"Metadata coverage only; tensor values are checked by separate live SHA/raw-matrix guards"}


def audit_loaded_auxiliary(proposer):
    from pard2_proposer3b_v1 import matrix_inventory
    raw=matrix_inventory(proposer._directory/"model.safetensors")
    state=proposer.model.state_dict(keep_vars=True)
    loaded={name:{"shape":tuple(value.shape),"dtype":"BF16" if str(value.dtype)=="torch.bfloat16" else str(value.dtype),
        "storage":(str(value.device),value.untyped_storage().data_ptr(),value.storage_offset())}
        for name,value in state.items()}
    expected=proposer.metadata["checkpoint_tying"]["effective_tie_word_embeddings"]
    return validate_metadata(raw,loaded,expected_tied=expected)


def validate_tensor_hashes(raw_hashes,loaded_hashes):
    if set(raw_hashes)!=set(loaded_hashes) or not raw_hashes:
        raise ValueError("Auxiliary tensor byte audits do not cover identical nonempty key sets")
    different=[name for name in raw_hashes if raw_hashes[name]!=loaded_hashes[name]]
    if different:
        raise ValueError("Loaded auxiliary tensor bytes differ from the original checkpoint: "+str(different))
    return {"all_original_auxiliary_checkpoint_tensor_bytes_loaded_exactly":True,
        "tensor_keys_checked":len(raw_hashes),"tensor_sha256":{
            name:{"stored_sha256":raw_hashes[name],"loaded_sha256":loaded_hashes[name],"bits_equal":True}
            for name in sorted(raw_hashes)}}


def audit_loaded_checkpoint_weights(proposer):
    """Every original auxiliary tensor: coverage plus raw-vs-loaded byte SHA.

    Reads only the checked local auxiliary file and the already loaded model.
    No original target, weight reload, remote code or float conversion occurs.
    This is a setup/final-proof audit, never per-request timed work.
    """
    from pard2_proposer3b_v1 import matrix_inventory,tensor_sha
    coverage=audit_loaded_auxiliary(proposer)
    raw=matrix_inventory(proposer._directory/"model.safetensors")
    state=proposer.model.state_dict(keep_vars=True)
    raw_hashes,loaded_hashes={},{}
    total_bytes=0
    for name in sorted(raw):
        entry=raw[name];digest=hashlib.sha256()
        with entry["path"].open("rb") as stream:
            stream.seek(entry["offset"]);remaining=entry["bytes"]
            while remaining:
                block=stream.read(min(8*1024*1024,remaining))
                if not block:raise ValueError("Truncated original auxiliary tensor: "+name)
                digest.update(block);remaining-=len(block)
        raw_hashes[name]=digest.hexdigest()
        loaded_hashes[name]=tensor_sha(state[name])
        total_bytes+=entry["bytes"]
    return {**coverage,**validate_tensor_hashes(raw_hashes,loaded_hashes),
        "stored_tensor_bytes_with_alias_names_checked":total_bytes,
        "scope":"Every original stored auxiliary tensor and loaded state tensor, raw BF16 byte SHA; no tensor values converted"}


def self_test():
    names=("model.embed_tokens.weight","lm_head.weight","layer.weight")
    raw={name:{"shape":[2,3],"dtype":"BF16","bytes":12} for name in names}
    loaded={name:{"shape":[2,3],"dtype":"BF16","storage":1 if name!="layer.weight" else 2} for name in names}
    assert validate_metadata(raw,loaded,expected_tied=True)["loaded_embedding_head_storage_alias"]
    loaded["lm_head.weight"]["storage"]=3
    assert not validate_metadata(raw,loaded,expected_tied=False)["loaded_embedding_head_storage_alias"]
    from copy import deepcopy
    bad=deepcopy(loaded);bad.pop("layer.weight")
    cases=[(raw,bad,False),(raw,loaded,True)]
    bad=deepcopy(loaded);bad["layer.weight"]["shape"]=[3,3];cases.append((raw,bad,False))
    bad=deepcopy(loaded);bad["layer.weight"]["dtype"]="FP32";cases.append((raw,bad,False))
    bad=deepcopy(raw);bad["layer.weight"]["bytes"]=13;cases.append((bad,loaded,False))
    for a,b,tied in cases:
        try:validate_metadata(a,b,expected_tied=tied)
        except ValueError:pass
        else:raise AssertionError("Invalid full-state metadata accepted")
    hashes={name:"a"*64 for name in names}
    assert validate_tensor_hashes(hashes,hashes)["tensor_keys_checked"]==3
    changed=dict(hashes);changed["layer.weight"]="b"*64
    missing=dict(hashes);missing.pop("layer.weight")
    for bad in (changed,missing):
        try:validate_tensor_hashes(hashes,bad)
        except ValueError:pass
        else:raise AssertionError("Changed or unverified auxiliary tensor values accepted")
    return {"CPU_state_keys_shapes_dtypes_extents_and_tied_alias_tests_passed":True,"GPU_work_performed":False}


if __name__=="__main__":
    import json
    print(json.dumps(self_test()))
