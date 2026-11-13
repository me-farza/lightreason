"""Tune Ada's lossless gather on fetched rows from the eight-position trace.

This diagnoses transfer and reconstruction only. There is no proposer, GEMM,
CPU overflow, or request timing in this benchmark.
"""

import argparse
from functools import partial
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch

from edge35b.bf16_expert_codec import LAYERS, PROJECTIONS, checkpoint_bank, pack_verified_row, plan_bank, row_bytes
from edge35b.download import REVISION
from edge35b.freetoken_packed_transfer import _copy_rows, _repair_rows


DEFAULT = (4096, 128, 8)
TUNED = (4096, 128, 4)
GEOMETRIES = (
    DEFAULT, (1024, 32, 4), (1024, 64, 4), (1024, 128, 4), (2048, 32, 4), (2048, 64, 4),
    (2048, 128, 4), (4096, 32, 4), (4096, 64, 4), (4096, 128, 4), (4096, 32, 8),
    (4096, 64, 8), (8192, 64, 8),
)


def name(geometry):
    return "packed-" + "-".join(map(str, geometry))


def trace_shapes(report, *, include_fallback=False, include_unpacked=False):
    if report["revision"] != REVISION or not report["transport_summary"]["exact"]:
        raise ValueError("Expected the pinned, exact eight-position transfer report.")
    row = next(r for r in report["rows"] if r["case"] == "studio" and r["scenario"] == "raw-profile")
    passes = [r["expert_fetch_ids"] for r in row["diagnostics"]["passes"]]
    layers = sorted(LAYERS, key=lambda layer: -sum(len(ids[layer]) for ids in passes))[:4]
    if include_unpacked:
        layers.append(max((layer for layer in range(1, len(passes[0])) if layer not in LAYERS),
                          key=lambda layer: sum(len(ids[layer]) for ids in passes)))
    if include_fallback:
        layers.append(0)
    for layer in layers:
        fetches = sorted((ids[layer] for ids in passes if ids[layer]), key=len)
        indices = sorted({int((len(fetches) - 1) * q) for q in (0.25, 0.5, 0.9)})
        yield layer, [fetches[i] for i in indices]


def measure_layer(model, index, layer, shapes, repeats, native=None, *, shared_layout=False):
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
    from freetoken.kernel.pinned import device_ptr

    ids = sorted({e for shape in shapes for e in shape})
    id_map = {e: i for i, e in enumerate(ids)}
    raw, packed, out, descriptors, payloads = [], [], [], [], []
    shared_storage, shared_descriptors, shared_payloads = [], [], []
    have_escapes = False
    # Use contiguous host banks just as the raw production gather does. Remap
    # the trace IDs into this smaller bank; the rows and request order are exact.
    for projection in PROJECTIONS:
        weights = checkpoint_bank(model, index, layer, projection)[ids].reshape(len(ids), -1)
        host = torch.from_numpy(weights.view(np.int16)).pin_memory()
        raw.append(host)
        plans, total = plan_bank(weights)
        payload = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        packed.append(payload)
        payload_np = payload.numpy()
        raw_base, packed_base = device_ptr(host), device_ptr(payload) if total else 0
        desc, sizes = [(0, -1, 0)] * 256, []
        for i, (bits, plan) in enumerate(zip(weights, plans, strict=True)):
            if plan is None:
                desc[i] = (raw_base + i * bits.nbytes, -1, 0)
                sizes.append(bits.nbytes)
            else:
                offset, base, escapes = plan
                size = row_bytes(bits.size, escapes)
                pack_verified_row(bits, payload_np[offset:offset + size], base, escapes)
                desc[i] = (packed_base + offset, base, escapes)
                sizes.append(size)
                have_escapes |= escapes > 0
        descriptors.append(desc)
        payloads.append(sizes)
        if shared_layout:
            from freetoken.kernel import _cpu_moe

            row_values = weights.shape[1]
            row_tiles = row_values // 256
            packed_row_bytes = row_values * 3 // 2
            tiles = torch.empty(len(ids) * packed_row_bytes, dtype=torch.uint8, pin_memory=True)
            bases = torch.empty((len(ids), row_tiles), dtype=torch.uint8)
            _cpu_moe.pack_cpu_read_tiles(host.data_ptr(), tiles.data_ptr(), bases.data_ptr(), bases.numel())
            gpu_bases = bases.cuda()
            shared_storage.extend((tiles, gpu_bases))
            base_addr = gpu_bases.data_ptr()
            tile_addr = device_ptr(tiles)
            table = [(0, 0, 0)] * 256
            encoded_tiles = (bases != 255).sum(dim=1).tolist()
            for row, encoded in enumerate(encoded_tiles):
                raw_address = raw_base + row * row_values * 2
                table[row] = ((tile_addr + row * packed_row_bytes, base_addr + row * row_tiles, raw_address)
                              if encoded else (0, 0, raw_address))
            shared_descriptors.append(table)
            shared_payloads.append([row_values * 2 - encoded * 128 for encoded in encoded_tiles])
        out.append(torch.empty((64, weights.shape[1]), device="cuda", dtype=torch.int16))

    device = "cuda"
    src_ptrs = torch.tensor([device_ptr(b) for b in raw], device=device, dtype=torch.int64)
    dest_ptrs = torch.tensor([b.data_ptr() for b in out], device=device, dtype=torch.int64)
    sizes = torch.tensor([b.shape[1] * 2 for b in raw], device=device, dtype=torch.int64)
    table = torch.tensor(descriptors, device=device, dtype=torch.int64)
    shared_table = torch.tensor(shared_descriptors, device=device, dtype=torch.int64) if shared_layout else None
    rng = np.random.default_rng(layer + 8)
    rows = []
    for shape in shapes:
        source = [id_map[e] for e in shape]
        dest = rng.choice(64, len(source), replace=False).tolist()
        src = torch.tensor(source + [-1] * (64 - len(source)), device=device, dtype=torch.int32)
        dst = torch.tensor(dest + [-1] * (64 - len(source)), device=device, dtype=torch.int32)
        count = torch.tensor([len(source)], device=device, dtype=torch.int64)
        args = (table, dest_ptrs, sizes, src, dst, count)

        def transfer(geometry):
            pairs, blocks, warps = geometry
            _copy_rows[(blocks, 2)](*args, 256, pairs, blocks, num_warps=warps)
            if have_escapes:
                _repair_rows[(64, 2)](*args, 256, 256, num_warps=4)

        transfers = {"raw": partial(fast_index_copy_multi_jit, dest_ptrs, src_ptrs, sizes,
                                    dst, src, count, blocks_per_bank=8)}
        transfers.update((name(geometry), partial(transfer, geometry))
                         for geometry in ((DEFAULT, TUNED) if native else GEOMETRIES))
        if native:
            from edge35b.native_bf16_gather import NATIVE_GEOMETRIES, name as native_name

            transfers.update((native_name(geometry), partial(native.copy, *args, max_copies=64,
                                                              experts=256, geometry=geometry))
                             for geometry in NATIVE_GEOMETRIES)
            if shared_layout:
                from edge35b.native_bf16_gather import CPU_TILE_GEOMETRIES

                transfers.update((native_name(geometry), partial(native.copy, shared_table, *args[1:], max_copies=64,
                                                                  experts=256, geometry=geometry))
                                 for geometry in CPU_TILE_GEOMETRIES)

        graphs = {}
        untouched = next(i for i in range(64) if i not in dest)
        for tag, copy in transfers.items():
            copy()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                copy()
            graphs[tag] = graph
            for bank in out:
                bank.fill_(0x5A5A)
            if native:
                count.zero_()
                graph.replay()
                if not all(bool((bank[dest] == 0x5A5A).all()) for bank in out):
                    raise RuntimeError(f"{tag} wrote to the cache with an empty plan")
                count.fill_(len(source))
            graph.replay()
            for bank, original in zip(out, raw, strict=True):
                if not torch.equal(bank[dest].cpu(), original[source]) or not bool((bank[untouched] == 0x5A5A).all()):
                    raise RuntimeError(f"{tag} changed BF16 values or an untouched slot on layer {layer}")
        rates = {tag: [] for tag in graphs}
        for rep in range(repeats):
            order = tuple(graphs) if rep % 2 == 0 else tuple(reversed(graphs))
            for tag in order:
                first = torch.cuda.Event(enable_timing=True)
                last = torch.cuda.Event(enable_timing=True)
                first.record()
                graphs[tag].replay()
                last.record()
                last.synchronize()
                rates[tag].append(first.elapsed_time(last))
        rows.append({
            "layer": layer, "fetched_ids": shape, "bitwise_match": True,
            "raw_bytes": sum(b.shape[1] * 2 for b in raw) * len(shape),
            "packed_bytes": sum(bank[i] for bank in payloads for i in source),
            "median_ms": {tag: median(samples) for tag, samples in rates.items()},
            "samples_ms": rates,
        })
        if shared_layout:
            rows[-1]["shared_pcie_bytes"] = sum(bank[i] for bank in shared_payloads for i in source)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--native-cuda", action="store_true", help="Compare native sm_89 kernels and the two Triton controls")
    p.add_argument("--shared-cpu-layout", action="store_true", help="Compare CUDA reads of the AVX2 packed bytes")
    args = p.parse_args()
    if args.out.exists() or args.repeats < 2:
        p.error("Use a fresh result path and at least two repetitions.")
    if args.shared_cpu_layout and not args.native_cuda:
        p.error("The shared CPU layout requires the native Ada benchmark.")
    manifest = json.loads((args.model / "lightreason-download.json").read_text())
    if manifest["revision"] != REVISION:
        p.error("Use the pinned Qwen BF16 checkpoint.")
    trace = json.loads(args.trace.read_text())
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    start = time.monotonic()
    native = None
    if args.native_cuda:
        from edge35b.native_bf16_gather import NATIVE_GEOMETRIES, NativeBf16Gather, name as native_name

        native = NativeBf16Gather()
    rows, layers = [], []
    for layer, shapes in trace_shapes(trace, include_fallback=args.native_cuda, include_unpacked=args.shared_cpu_layout):
        if layer != 0:
            layers.append(layer)
        measured = measure_layer(args.model, index, layer, shapes, args.repeats, native,
                                 shared_layout=args.shared_cpu_layout)
        rows.extend(measured)
        print("layer", layer, "fetch counts", [len(shape) for shape in shapes], flush=True)
    dev = [r for r in rows if r["layer"] in layers[:2]]
    holdout = [r for r in rows if r["layer"] in layers[2:]]

    def score(data, tag):
        return sum(r["median_ms"]["raw"] for r in data) / sum(r["median_ms"][tag] for r in data)

    candidates = (native_name(g) for g in NATIVE_GEOMETRIES) if native else (name(g) for g in GEOMETRIES)
    candidate = max(candidates, key=lambda tag: score(dev, tag))
    report = {
        "revision": REVISION, "gpu": torch.cuda.get_device_name(), "repeats": args.repeats,
        "development_layers": layers[:2], "held_out_layers": layers[2:], "candidate": candidate,
        "development_speedup": score(dev, candidate), "held_out_speedup": score(holdout, candidate),
        "held_out_default_speedup": score(holdout, name(DEFAULT)),
        "scope": "Traced standalone GPU gather and unpack, with no concurrent CPU experts",
        "elapsed_s": time.monotonic() - start, "rows": rows,
    }
    if native:
        report.update(native_cuda_binary_version=native.binary_version,
                      held_out_triton_tuned_speedup=score(holdout, name(TUNED)), fallback_layer=0)
    if args.shared_cpu_layout:
        from edge35b.native_bf16_gather import CPU_TILE_GEOMETRIES

        shared = max(map(native_name, CPU_TILE_GEOMETRIES), key=lambda tag: score(dev, tag))
        control_time = sum(r["median_ms"]["cuda-8-128-default" if r["layer"] in LAYERS else "raw"] for r in holdout)
        shared_time = sum(r["median_ms"][shared] for r in holdout)
        report["shared_cpu_layout"] = {"selected": shared, "development_speedup_vs_raw": score(dev, shared),
                                       "held_out_hybrid_control_ms": control_time, "held_out_shared_ms": shared_time,
                                       "held_out_speedup_vs_hybrid": control_time / shared_time}
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print({k: v for k, v in report.items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    main()
