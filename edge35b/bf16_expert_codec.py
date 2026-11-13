"""Losslessly pack BF16 expert rows for transfer into ordinary BF16 cache slots."""

from struct import unpack
import json

import numpy as np


LAYERS = (2, 1, 3, 4, 11, 5, 10, 39, 6, 23, 22, 34)
PROJECTIONS = ("gate_up_proj", "down_proj")


def checkpoint_bank(root, index, layer, projection):
    key = f"model.language_model.layers.{layer}.mlp.experts.{projection}"
    path = root / index[key]
    with path.open("rb") as stream:
        size, = unpack("<Q", stream.read(8))
        info = json.loads(stream.read(size))[key]
    shape = tuple(info["shape"])
    start, end = info["data_offsets"]
    if info["dtype"] != "BF16" or len(shape) != 3 or shape[0] != 256 or end - start != 2 * np.prod(shape):
        raise ValueError(f"Unexpected Qwen BF16 bank layout for {key}")
    return np.memmap(path, mode="r", dtype="<u2", offset=8 + size + start, shape=shape)


def exponent_window(bits):
    if bits.ndim != 1 or bits.dtype != np.dtype("<u2") or bits.size % 8 or not 0 < bits.size < 2**31:
        raise ValueError("Expected a flat, aligned BF16 bit array")
    exponents = (bits >> 7) & 255
    counts = np.bincount(exponents, minlength=256)
    windows = np.convolve(counts, np.ones(16, dtype=np.int64), mode="valid")
    base = int(windows.argmax())
    return base, int(bits.size - windows[base])


def row_bytes(n, escapes):
    return 3 * n // 2 + 6 * escapes


def pack_row(bits, base, escapes, output):
    n = bits.size
    delta = ((bits >> 7) & 255).astype(np.int16) - base
    indices = np.flatnonzero((delta < 0) | (delta >= 16))
    if indices.size != escapes or output.size != row_bytes(n, escapes):
        raise ValueError("BF16 row disagrees with its packed layout")
    nibble = (delta & 15).astype(np.uint8)
    split = 3 * n // 2
    output[:n] = (bits & 127) | ((bits >> 8) & 128)
    output[n:split] = nibble[::2] | (nibble[1::2] << 4)
    output[split:split + 4 * escapes].view("<i4")[:] = indices
    output[split + 4 * escapes:].view("<u2")[:] = bits[indices]


def restore_row(packed, n, base, escapes):
    split = 3 * n // 2
    sign_fraction = packed[:n].astype(np.uint16)
    exponent = packed[n:split].astype(np.uint16)
    bits = ((sign_fraction & 128) << 8) | (sign_fraction & 127)
    bits[::2] |= ((exponent & 15) + base) << 7
    bits[1::2] |= ((exponent >> 4) + base) << 7
    indices = packed[split:split + 4 * escapes].view("<i4")
    bits[indices] = packed[split + 4 * escapes:].view("<u2")
    return bits


def plan_bank(bits, *, check=None):
    """Lay out aligned packed rows. A dense escape list uses the raw bank."""
    rows, size = [], 0
    for row in bits:
        if check is not None:
            check()
        base, escapes = exponent_window(row)
        if escapes * 100 > row.size:
            rows.append(None)
            continue
        rows.append((size, base, escapes))
        size = (size + row_bytes(row.size, escapes) + 15) & ~15
    return rows, size


def pack_verified_row(row, output, base, escapes):
    pack_row(row, base, escapes, output)
    if not np.array_equal(row, restore_row(output, row.size, base, escapes)):
        raise RuntimeError("Packed expert did not round-trip to the original BF16 bits")
