"""One-request soft input rows for the experimental local FreeToken worker."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


MARKER = "<|fim_pad|>"
MARKER_ID = 248063
MAX_ROWS = 12
HIDDEN_SIZE = 2048
INGRESS_ENV = "LIGHTREASON_LATENT_INGRESS"


def embedding_digest(rows):
    return sha256(rows.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


@lru_cache(maxsize=1)
def _read_sidecar(path, identity):
    # The inode is part of the cache key because a new request atomically replaces
    # this file. Nothing persists in the worker after all rows have been read.
    with safe_open(path, framework="pt", device="cpu") as handle:
        info = handle.metadata()
        rows = handle.get_tensor("embeddings")
    prompt_ids = json.loads(info["prompt_ids"])
    positions = [index for index, token in enumerate(prompt_ids) if token == MARKER_ID]
    if (rows.dtype != torch.bfloat16 or rows.shape != (len(positions), HIDDEN_SIZE)
            or not 1 <= len(positions) <= MAX_ROWS
            or not torch.isfinite(rows).all()):
        raise RuntimeError("Invalid Lightreason memory embedding payload.")
    return info["request_id"], prompt_ids, positions, rows


def inject_memory_embeddings(embedded, batch):
    """Replace native sentinel token embeddings during uncached prefill."""
    if not batch.is_prefill:
        return embedded
    chunk = []
    total_rows = 0
    for req in batch.reqs:
        start, stop = req.cached_len, req.device_len
        ids = req.input_ids[start:stop]
        if MARKER_ID in ids:
            chunk.append((req, total_rows, start, stop))
        total_rows += stop - start
    if not chunk:
        return embedded
    if len(batch.reqs) != 1 or len(chunk) != 1 or total_rows != embedded.shape[0]:
        raise RuntimeError("Unexpected Lightreason memory prefill batch.")
    req, batch_offset, start, stop = chunk[0]
    base = Path(os.environ[INGRESS_ENV])
    pending = base / "pending.safetensors"
    receipt = base / "injected.json"
    request_id, prompt_ids, positions, rows = _read_sidecar(str(pending), pending.stat().st_ino)
    if req.input_ids[:stop].tolist() != prompt_ids[:stop] or req.prompt_len != len(prompt_ids):
        raise RuntimeError("Lightreason memory payload does not match this request.")
    previous = {"request_id": request_id, "uid": req.uid, "positions": []}
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["request_id"] != request_id or previous["uid"] != req.uid:
            raise RuntimeError("Lightreason memory payload belongs to a different request.")
    indexes = [index for index, position in enumerate(positions) if start <= position < stop]
    injected = previous["positions"] + [positions[index] for index in indexes]
    if not indexes or injected != positions[:len(injected)]:
        raise RuntimeError("Lightreason memory embedding rows were replayed or skipped.")
    local_positions = [batch_offset + positions[index] - start for index in indexes]
    y = rows[indexes].to(device=embedded.device, dtype=embedded.dtype)
    embedded.index_copy_(0, torch.tensor(local_positions, device=embedded.device), y)
    staged = receipt.with_suffix(".part")
    staged.write_text(json.dumps({
        "request_id": request_id, "uid": req.uid, "positions": injected,
        "sha256": embedding_digest(rows),
    }))
    staged.replace(receipt)
    if len(injected) == len(positions):
        pending.unlink()
        _read_sidecar.cache_clear()
    return embedded
