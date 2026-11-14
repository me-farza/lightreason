"""Load benchmark evidence from existing artifacts for the runnable demo."""

import json
from pathlib import Path
from statistics import median

DEMO_QUESTION = "Which results preserved complete output agreement, and which did not?"
DEMO_UNKNOWN = "What was the total GPU energy in joules for the 35B coding run?"
DEMO_AS_OF = "2026-11-12T23:59:59Z"
DEMO_CORRECTED_AT = "2026-11-13T00:00:00Z"
DEMO_CORRECTION_QUESTION = "What was the output cap for the Studio weather comparison?"
STRICT = "results3b/summary.json"
PACKED = "edge35b/results/native50/gpu-resident-qwen.json"
WEATHER = "demo_chat/.runtime/pageable-handoff-comparison-20261112.json"


def seed_demo(store, source_root):
    """Return three scoped observations. The cited values come from the artifacts."""
    root = Path(source_root)
    records = []
    try:
        strict = json.loads((root / STRICT).read_text())
        exact = strict["validation"]["exact_cases"]
        strict_speedup = strict["timing"]["median_total_speedup"]
        strict_text = (
            f"Public strict 3B preserved complete output agreement against the original Hugging Face runner "
            f"on {exact}/{exact} complete validation outputs and measured {strict_speedup:.2f}x median "
            "prompt-to-completion speedup."
        )
        strict_kind = "tool_observation"
    except FileNotFoundError:
        strict_text = (
            "The user reports that public strict 3B preserved complete output agreement on 38/38 "
            "complete validation outputs with 2.19x median prompt-to-completion speedup."
        )
        strict_kind = "user_report"
    records.append(store.observe(
        "demo-strict-3b", strict_text, STRICT if strict_kind == "tool_observation" else "user request",
        "2026-11-06T00:00:00Z", strict_kind,
    ))

    try:
        packed = json.loads((root / PACKED).read_text())
        pairs = packed["gpu_resident_cache"]["pairs"]
        coding_speed = median(
            row["timings"]["predicted_per_second"] for row in packed["rows"]
            if row["case"] == "coding" and row["scenario"] == "gpu-packed-off" and row["stream"]
        )
        passed = sum(pair["exact"] for pair in pairs)
        packed_text = (
            f"The optional 35B packed GPU expert cache reached {coding_speed:.2f} tokens/s on streamed coding "
            f"and preserved complete output agreement against the tuned fixed-eight control in {passed}/{len(pairs)} "
            "comparisons of raw and packed resident layouts, including token IDs, stop reasons and per-pass commits."
        )
        packed_kind = "tool_observation"
    except FileNotFoundError:
        packed_text = (
            "The user reports that the optional 35B packed GPU cache reached 51.1 tokens/s on coding and "
            "preserved complete output agreement against its tuned control across 40 comparisons."
        )
        packed_kind = "user_report"
    records.append(store.observe(
        "demo-packed-35b", packed_text, PACKED if packed_kind == "tool_observation" else "user request",
        "2026-11-12T20:00:00Z", packed_kind,
    ))

    try:
        weather = json.loads((root / WEATHER).read_text())
        decode_speedup = median(run["decode_speedup"] for run in weather["runs"])
        prefix = weather["runs"][0]["matching_initial_tokens"]
        if any(run["matching_initial_tokens"] != prefix for run in weather["runs"]):
            raise ValueError("Weather comparison recorded different matching prefixes")
        weather_text = (
            f"The Studio 35B weather comparison against the original Scheduled runner measured "
            f"{decode_speedup:.1f}x faster decode, but complete output agreement failed because the "
            f"answers diverged after the first {prefix} tokens."
        )
        weather_kind = "tool_observation"
    except FileNotFoundError:
        weather_text = (
            "The user reports that the Studio weather comparison measured 5.7x faster decode, but complete "
            "output agreement failed because answers diverged after the first 30 tokens."
        )
        weather_kind = "user_report"
    records.append(store.observe(
        "demo-studio-weather", weather_text, WEATHER if weather_kind == "tool_observation" else "user request",
        "2026-11-12T20:00:00Z", weather_kind,
    ))
    return records
