"""Measure actual-size Qwen expert transfers with deterministic synthetic routing."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from safetensors import safe_open
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

from edge35b.expert_stream import MODES, ExpertPool, patch_experts
from edge35b.download import REVISION, file_hash


def load_expert_layer(layer, root, checkpoint_files, layer_index):
    with checkpoint_files.open() as file:
        metadata = json.load(file)
    if metadata["sha"] != REVISION:
        raise ValueError("Use shard metadata for the pinned revision")
    entries = {item["rfilename"]: item for item in metadata["siblings"]}
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    verified = set()
    for part in ("gate_up_proj", "down_proj"):
        key = f"model.language_model.layers.{layer_index}.mlp.experts.{part}"
        name = index[key]
        entry, path = entries[name], root / name
        if name not in verified:
            size, expected = entry["size"], entry["lfs"]["sha256"]
            if path.stat().st_size != size or file_hash(path, size, True).hexdigest() != expected:
                raise ValueError(f"Official shard verification failed: {name}")
            verified.add(name)
        with safe_open(path, framework="pt", device="cpu") as source:
            weight = source.get_tensor(key)
        reference = getattr(layer, part)
        if weight.dtype != torch.bfloat16 or weight.shape != reference.shape:
            raise ValueError(f"Invalid expert tensor: {key}")
        setattr(layer, part, torch.nn.Parameter(weight, requires_grad=False))
    return sorted(verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint-files", type=Path, help="Pinned Hugging Face files metadata JSON")
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh --out")
    if bool(args.model_dir) != bool(args.checkpoint_files):
        parser.error("Specify both --model-dir and --checkpoint-files for original weights")
    torch.set_num_threads(1)
    torch.manual_seed(8)
    config = Qwen3_5MoeTextConfig(hidden_size=2048, moe_intermediate_size=512, num_experts=256)
    config._experts_implementation = "eager"
    with torch.device("meta"):
        cpu = Qwen3_5MoeExperts(config).eval().requires_grad_(False).bfloat16()
    if args.model_dir:
        files = load_expert_layer(cpu, args.model_dir, args.checkpoint_files, args.layer)
        identity = {"layer": args.layer, "revision": REVISION, "verified_shards": files}
    else:
        cpu.to_empty(device="cpu")
        with torch.no_grad():
            for parameter in cpu.parameters():
                parameter.normal_(std=0.02)
        identity = {"weights": "synthetic"}
    gpu = deepcopy(cpu).cuda()
    inputs = []
    for step in range(16):
        x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
        # Four experts repeat across steps. The other four are disjoint.
        ids = torch.tensor([[0, 1, 2, 3, *(range(4 + step * 4, 8 + step * 4))]], device="cuda")
        probabilities = torch.full((1, 8), 0.125, device="cuda", dtype=torch.bfloat16)
        inputs.append((x, ids, probabilities))
    pool = ExpertPool(capacity_bytes=1 << 29)
    result = {"scope": "one actual-size expert layer with synthetic inputs and routing", **identity, "trials": []}
    with torch.inference_mode():
        reference = [gpu(*item) for item in inputs]
        del gpu
        for order in (args.modes, reversed(args.modes)):
            for mode in order:
                pool.clear()
                pool.reset_stats()
                torch.cuda.synchronize()
                with patch_experts(cpu, mode=mode, pool=pool):
                    start = time.perf_counter()
                    outputs = [cpu(*item) for item in inputs]
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                matches = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(outputs, reference))
                result["trials"].append({"mode": mode, "steps": len(inputs), "elapsed_s": elapsed,
                                         "bitwise_upstream_cuda_match": matches, **pool.snapshot()})
                if not matches:
                    raise RuntimeError(f"Bitwise mismatch for {mode}")
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
