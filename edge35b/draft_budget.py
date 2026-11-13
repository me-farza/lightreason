"""Calibrate a DFlash2 prefix budget on real verifier widths and expert traffic."""

from collections import defaultdict
from math import isfinite
from statistics import median

from edge35b.analyze_pass_diagnostics import EXPERT_BYTES, profile_pass


WIDTHS = (4, 6, 8)
CONFIDENCE_BINS = (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.97)
COLD_IDS = 24
MIN_GAIN = 0.03


def confidence_bin(p):
    return sum(p >= threshold for threshold in CONFIDENCE_BINS)


def forecast_commits(width, confidence, probabilities, remaining):
    expected = survival = 1.0
    for p, row in zip(confidence[:min(width, remaining) - 1], probabilities):
        survival *= row[confidence_bin(p)]
        expected += survival
    return expected


def select_width(model, confidence, emitted, remaining):
    phase = "cold" if emitted < COLD_IDS else "steady"
    costs = model["costs"][phase]
    predictions = {
        width: forecast_commits(width, confidence, model["probabilities"], remaining)
        for width in WIDTHS
    }
    rates = {width: predictions[width] / costs[str(width)]["pass_ms"] for width in WIDTHS}
    best = max(WIDTHS, key=lambda width: (
        rates[width], predictions[width] / costs[str(width)]["host_bytes"], width,
    ))
    return best if rates[best] > rates[8] * (1 + MIN_GAIN) else 8


def validate(model):
    n = len(CONFIDENCE_BINS) + 1
    probs = model["probabilities"]
    if (len(probs) != 7 or any(len(row) != n or any(not isfinite(p) or not 0 < p < 1 for p in row)
                              for row in probs)):
        raise ValueError("Expected seven calibrated DFlash2 draft positions")
    for phase in ("cold", "steady"):
        for width in WIDTHS:
            cost = model["costs"][phase][str(width)]
            if any(not isfinite(cost[k]) or cost[k] <= 0 for k in ("pass_ms", "host_bytes")):
                raise ValueError("Expected positive verifier costs for every width")


def memory_metrics(record, packed_layers):
    """Return measured PCIe payload and logical CPU weight bytes at one read per expert."""
    pcie = sum(record["expert_pcie_payload_bytes"])
    cpu = 0
    for layer, row in enumerate(record["gpu_layer_stamps"]):
        distinct = row[8] - row[7]
        cpu += distinct * (3 * EXPERT_BYTES // 4 if layer in packed_layers else EXPERT_BYTES)
    return pcie, cpu


def fit(runs, packed_layers):
    outcomes = defaultdict(list)
    costs = defaultdict(list)
    for run in runs:
        emitted = 0
        for step, (_, committed, ms) in zip(run["diagnostics"]["passes"], run["passes"], strict=True):
            width, accepted = step["width"], step["accepted_drafts"]
            confidence = step["draft_confidence"]
            if len(confidence) != 7 or not all(isfinite(p) and 0 <= p <= 1 for p in confidence):
                raise ValueError("Invalid confidence from the live DFlash2 selector")
            for i in range(min(width - 1, accepted + 1, committed)):
                outcomes[i, confidence_bin(confidence[i])].append(i < accepted)
            pcie, cpu = memory_metrics(step, packed_layers)
            measured = profile_pass(step, ms)
            phase = "cold" if emitted < COLD_IDS else "steady"
            costs[phase, width].append((ms, pcie, cpu, measured["pcie_gather_ms"]))
            emitted += committed

    probabilities, counts = [], []
    n = len(CONFIDENCE_BINS) + 1
    for i in range(7):
        rows = [outcomes[i, b] for b in range(n)]
        if not any(rows):
            raise ValueError(f"No calibration labels at draft position {i + 1}")
        average = (1 + sum(map(sum, rows))) / (2 + sum(map(len, rows)))
        probabilities.append([(sum(row) + 8 * average) / (len(row) + 8) for row in rows])
        counts.append(list(map(len, rows)))

    estimates = {phase: {} for phase in ("cold", "steady")}
    for phase, by_width in estimates.items():
        for width in WIDTHS:
            rows = costs[phase, width]
            if len(rows) < 3:
                raise ValueError(f"Missing measured {phase} costs at width {width}")
            pcie = median(row[1] for row in rows)
            cpu = median(row[2] for row in rows)
            # The gather rate was measured with the CPU workers active. Include
            # exposed CPU wait, the proposer and all other work in the residual.
            bw = median(p / gather for _, p, _, gather in rows if p > 0 and gather > 0)
            residual = median(ms - gather for ms, _, _, gather in rows)
            by_width[str(width)] = {"pass_ms": residual + pcie / bw, "host_bytes": pcie + cpu,
                                    "pcie_bytes": pcie, "cpu_min_bytes": cpu, "samples": len(rows)}
    model = {"probabilities": probabilities, "counts": counts, "costs": estimates}
    validate(model)
    return model
