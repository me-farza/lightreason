"""Fit a continuous projection from saved memory codes to Qwen 35B input embeddings."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn.functional import normalize
from safetensors import safe_open
from safetensors.torch import load_file

from edge35b.download import MODEL as DESTINATION_MODEL, REVISION as DESTINATION_REVISION, file_hash

from .neural import MAX_VALUE_TOKENS, NeuralModel, VALUE_DIM, _atomic_write, load_qwen


FORMAT = "lightreason-latent-bridge-v1"
DESTINATION_WIDTH = 2048
DESTINATION_TENSOR = "model.language_model.embed_tokens.weight"
_LEXICAL = re.compile(r" ?[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_destination(model_path):
    """Read only the pinned Qwen 35B tokenizer and original BF16 input matrix."""
    from transformers import AutoTokenizer

    path = Path(model_path)
    manifest = json.loads((path / "lightreason-download.json").read_text())
    if (manifest.get("model"), manifest.get("revision")) != (DESTINATION_MODEL, DESTINATION_REVISION):
        raise ValueError("Use the original pinned Qwen3.6-35B-A3B checkpoint.")
    files = {row["rfilename"]: row for row in manifest["files"]}

    def verify(name):
        entry = files[name]
        asset = path / name
        lfs = entry.get("lfs")
        expected = lfs["sha256"] if lfs else entry["blobId"]
        if asset.stat().st_size != entry["size"] or file_hash(asset, entry["size"], lfs).hexdigest() != expected:
            raise ValueError(f"Destination model file failed its pinned hash: {name}")
        return asset

    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        verify(name)
    index = json.loads(verify("model.safetensors.index.json").read_text())
    with safe_open(verify(index["weight_map"][DESTINATION_TENSOR]), framework="pt", device="cpu") as shard:
        embeddings = shard.get_tensor(DESTINATION_TENSOR)
    config = json.loads((path / "config.json").read_text())["text_config"]
    if embeddings.shape != (config["vocab_size"], DESTINATION_WIDTH) or embeddings.dtype != torch.bfloat16:
        raise ValueError("Unexpected 35B input embeddings.")
    return AutoTokenizer.from_pretrained(path, local_files_only=True), embeddings


def _excluded_tokens(tokenizer):
    from .neural_demo import _WORDS, _fixture

    words = set(_WORDS)
    for split in ("acceptance", "alternate", "development"):
        for _, _, cases in _fixture(0, split):
            for _, _, answer in cases:
                if answer:
                    words.update(answer.split())
    return words, {
        token
        for word in words
        for variant in (word, word.capitalize(), word.upper())
        for form in (variant, " " + variant)
        for token in tokenizer.encode(form, add_special_tokens=False)
    }


def _aligned_ids(source, destination, vocab_rows, generator, maximum):
    excluded_words, excluded_ids = _excluded_tokens(source)
    source_special = set(source.all_special_ids)
    destination_special = set(destination.all_special_ids)
    src, dst, used = [], [], set()
    ids = torch.randperm(min(vocab_rows, len(source)), generator=generator).tolist()
    for offset in range(0, len(ids), 4096):
        selected = [i for i in ids[offset:offset + 4096] if i not in source_special and i not in excluded_ids]
        forms = source.batch_decode([[i] for i in selected], clean_up_tokenization_spaces=False)
        candidates = [(i, text) for i, text in zip(selected, forms, strict=True)
                      if _LEXICAL.fullmatch(text) and text.strip().casefold() not in excluded_words]
        if not candidates:
            continue
        strings = [text for _, text in candidates]
        source_roundtrip = source(strings, add_special_tokens=False)["input_ids"]
        destination_ids = destination(strings, add_special_tokens=False)["input_ids"]
        for (i, _), first, second in zip(candidates, source_roundtrip, destination_ids, strict=True):
            if first == [i] and len(second) == 1 and second[0] not in destination_special and second[0] not in used:
                src.append(i)
                dst.append(second[0])
                used.add(second[0])
                if len(src) == maximum:
                    return torch.tensor(src), torch.tensor(dst), len(excluded_ids)
    return torch.tensor(src), torch.tensor(dst), len(excluded_ids)


class LatentBridge:
    """A CPU float projection. No tokenizer or output head participates in forward."""

    def __init__(self, weight, bias, target_norm):
        if (weight.shape != (VALUE_DIM, DESTINATION_WIDTH) or bias.shape != (DESTINATION_WIDTH,)
                or target_norm.shape != () or target_norm.item() <= 0
                or any(x.dtype != torch.float32 or not torch.isfinite(x).all()
                       for x in (weight, bias, target_norm))):
            raise ValueError("Invalid projection dimensions, norms, or dtype.")
        self.weight, self.bias, self.target_norm = weight, bias, target_norm

    @classmethod
    def load(cls, path, *, memory_checkpoint=None, model_path=None):
        with safe_open(path, framework="pt", device="cpu") as file:
            meta = file.metadata() or {}
            if (meta.get("format"), meta.get("destination_revision")) != (FORMAT, DESTINATION_REVISION):
                raise ValueError("Unsupported latent bridge or destination revision.")
            if memory_checkpoint is not None and meta.get("memory_checkpoint_sha256") != _sha256(memory_checkpoint):
                raise ValueError("The latent bridge was trained for a different memory checkpoint.")
            if model_path is not None:
                manifest = json.loads((Path(model_path) / "lightreason-download.json").read_text())
                if (manifest.get("model"), manifest.get("revision")) != (DESTINATION_MODEL, DESTINATION_REVISION):
                    raise ValueError("The latent bridge was trained for a different destination model.")
            return cls(*(file.get_tensor(key) for key in ("weight", "bias", "target_norm")))

    @torch.inference_mode()
    def __call__(self, codes):
        if (not isinstance(codes, torch.Tensor) or codes.ndim != 2 or codes.shape[1] != VALUE_DIM
                or not 1 <= codes.shape[0] <= MAX_VALUE_TOKENS or not codes.is_floating_point()
                or not torch.isfinite(codes).all()):
            raise ValueError("Expected a finite bank of selected continuous memory codes.")
        projected = normalize(codes.to(device="cpu", dtype=torch.float32) @ self.weight + self.bias, dim=-1)
        return (projected * self.target_norm).to(torch.bfloat16).contiguous()


def fit(args):
    started = perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    source, embeddings, head = load_qwen(args.model_3b, "cpu")
    memory = NeuralModel(embeddings, head).eval()
    memory.load_state_dict(load_file(args.memory_checkpoint, device="cpu"), strict=True)
    destination, target = load_destination(args.model_35b)
    source_ids, target_ids, excluded = _aligned_ids(source, destination, len(embeddings), generator, args.max_pairs)
    if len(source_ids) < 1024:
        raise ValueError("Too few aligned lexical forms for the embedding projection.")
    heldout = min(1024, len(source_ids) // 10)
    train_ids, test_ids = source_ids[heldout:], source_ids[:heldout]
    targets, test_targets = normalize(target[target_ids[heldout:]].float(), dim=-1), target[target_ids[:heldout]].float()
    target_norm = target[target_ids[heldout:]].float().norm(dim=-1).median()
    with torch.inference_mode():
        # Fit against the same FP16 precision restored from persisted state.
        codes = torch.cat([memory.encode(batch).half().float() for batch in train_ids.split(2048)])
        code_center, target_center = codes.mean(0), targets.mean(0)
        codes.sub_(code_center)
        targets.sub_(target_center)
        gram = codes.T @ codes
        gram.diagonal().add_(args.ridge * gram.diagonal().mean())
        weight = torch.linalg.solve(gram, codes.T @ targets)
        bias = target_center - code_center @ weight
        bridge = LatentBridge(weight, bias, target_norm)
        tests = memory.encode(test_ids).half()
        predicted = torch.cat([bridge(chunk).float() for chunk in tests.split(MAX_VALUE_TOKENS)])
        true_units = normalize(test_targets, dim=-1)
        similarities = normalize(predicted, dim=-1) @ true_units.T
        heldout_cosine = float(similarities.diagonal().mean())
        heldout_nearest = float((similarities.argmax(-1) == torch.arange(heldout)).float().mean())
        _atomic_write({"weight": weight, "bias": bias, "target_norm": target_norm}, args.out, {
            "format": FORMAT,
            "destination_revision": DESTINATION_REVISION,
            "memory_checkpoint_sha256": _sha256(args.memory_checkpoint),
        })
    report = {
        "format": FORMAT,
        "destination_revision": DESTINATION_REVISION,
        "seed": args.seed,
        "training_pairs": len(train_ids),
        "heldout_pairs": heldout,
        "excluded_source_token_ids": excluded,
        "heldout_target_mean_cosine": heldout_cosine,
        "heldout_top1_within_heldout_rows": heldout_nearest,
        "elapsed_s": perf_counter() - started,
    }
    if args.report is not None:
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    verbs = parser.add_subparsers(dest="verb", required=True)
    train = verbs.add_parser("fit", help="fit and measure a CPU projection offline")
    train.add_argument("--model-3b", required=True, type=Path)
    train.add_argument("--memory-checkpoint", required=True, type=Path)
    train.add_argument("--model-35b", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--report", type=Path)
    train.add_argument("--seed", type=int, default=1729)
    train.add_argument("--max-pairs", type=int, default=24000)
    train.add_argument("--threads", type=int, default=8)
    train.add_argument("--ridge", type=float, default=0.03)
    args = parser.parse_args()
    if args.out.exists() or (args.report is not None and args.report.exists()):
        parser.error("Choose fresh output paths.")
    if args.max_pairs < 1024 or args.threads < 1 or not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("Require at least 1024 pairs, positive threads and finite ridge regularization.")
    print(json.dumps(fit(args), indent=2))


if __name__ == "__main__":
    main()
