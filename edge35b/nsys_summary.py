"""Summarize NVTX and CUDA interval unions exported by Nsight Systems."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def union(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(ranges):
    return sum(end - start for start, end in union(ranges))


def inside(ranges, bounds):
    return [(max(start, left), min(end, right)) for left, right in bounds
            for start, end in ranges if start < right and end > left]


def report(path):
    with sqlite3.connect(path) as db:
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        marks = {}
        for start, end, label, text_id in db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
            marks.setdefault(label or strings.get(text_id), []).append((start, end))
        kernels = list(db.execute("SELECT start,end,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
        copies = list(db.execute("SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
        runtime = list(db.execute("SELECT start,end,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"))

    results = {}
    for phase in ("prefill", "decode-token"):
        bounds = marks[phase]
        windows = [(bounds[0][0], bounds[-1][1])]
        count = len(bounds)
        wall = duration(windows)
        gpu_kernels = inside([(a, b) for a, b, _ in kernels], windows)
        gpu_copies = inside([(a, b) for a, b, _, _ in copies], windows)
        busy = gpu_kernels + gpu_copies
        cpu = inside(marks.get("cpu-experts", []), windows)
        fences = inside(marks.get("route-and-hidden-fence", []), windows)
        weight_copy_host = inside(marks.get("expert-weight-copy", []), windows)
        actual = duration(busy)
        overlap = duration(cpu) + actual - duration(cpu + busy)
        work = Counter()
        for a, b, name in kernels:
            if a >= windows[0][0] and b <= windows[0][1]:
                work[strings[name]] += b - a
        host = Counter()
        for a, b, name in runtime:
            if a >= windows[0][0] and b <= windows[0][1]:
                host[strings[name]] += b - a
        direction = Counter()
        for a, b, kind, size in copies:
            if a >= windows[0][0] and b <= windows[0][1]:
                direction[kind] += size
        factor = count * 1e6
        results[phase] = {
            "steps": count, "wall_ms_per_step": wall / factor,
            "gpu_kernel_union_ms_per_step": duration(gpu_kernels) / factor,
            "gpu_copy_union_ms_per_step": duration(gpu_copies) / factor,
            "gpu_busy_union_ms_per_step": actual / factor,
            "cpu_expert_ms_per_step": duration(cpu) / factor,
            "cpu_experts_overlap_gpu_ms_per_step": overlap / factor,
            "route_fence_ms_per_step": duration(fences) / factor,
            "weight_copy_host_ms_per_step": duration(weight_copy_host) / factor,
            "gpu_or_cpu_experts_idle_ms_per_step": (wall - duration(cpu + busy)) / factor,
            "copy_bytes_per_step_by_cupti_kind": {k: v / count for k, v in direction.items()},
            "kernel_duration_ms_per_step_top": {k: v / factor for k, v in work.most_common(12)},
            "runtime_duration_ms_per_step_top": {k: v / factor for k, v in host.most_common(12)},
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.sqlite), indent=2))


if __name__ == "__main__":
    main()
