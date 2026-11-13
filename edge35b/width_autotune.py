"""A small verifier-width policy for the single-request BF16 benchmark."""

from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from time import perf_counter


WIDTHS = (4, 6, 8)
PACKED_TRANSFER_WIDTHS = (4, 8)
STATES = ("first", "short", "middle", "long")
DEFAULT_CPU_PAIR_TUNING = (0, 1, 64)
NATIVE_PACKED_TRANSPORT = "native-packed-hot"
SHARED_CPU_TRANSPORT = "native-shared-cpu"
SHARED_CPU_CONTROL_TRANSPORT = "native-shared-cpu-control"
PACKED_PRIMARY_TRANSPORT = "packed-primary"
SHARED_CPU_TRANSPORTS = (SHARED_CPU_CONTROL_TRANSPORT, SHARED_CPU_TRANSPORT)
NATIVE_TRANSPORTS = (NATIVE_PACKED_TRANSPORT, *SHARED_CPU_TRANSPORTS, PACKED_PRIMARY_TRANSPORT)
GATHER_GEOMETRIES = {f"hot-tiles-{stripes}-{tile}": (stripes, tile)
                     for stripes in (2, 4, 8) for tile in (2048, 4096)}


def validate_cpu_pair_tuning(values):
    if (not isinstance(values, (tuple, list)) or len(values) != 3
            or any(type(value) is not int for value in values)):
        raise ValueError("CPU pair tuning needs three integers.")
    widen, chunk, max_routes = values
    if widen not in (0, 1, 2, 3) or chunk not in (1, 2, 4) or not 1 <= max_routes <= 64:
        raise ValueError("Use widening 0–3, chunk size 1, 2, or 4, and a route limit of 1–64.")
    return widen, chunk, max_routes


def state(previous):
    if previous is None:
        return "first"
    width, committed, _ = previous
    if committed >= width - 1:
        return "long"
    if committed * 2 <= width:
        return "short"
    return "middle"


def fit_table(episodes):
    """Fit from fixed-width development runs using only pre-decision features."""
    samples = defaultdict(list)
    for passes in episodes:
        previous = None
        for width, committed, ms in passes:
            samples[state(previous), width].append((committed, ms))
            previous = (width, committed, ms)
    decision = {}
    for bucket in STATES:
        rates = {}
        for width in WIDTHS:
            rows = samples[bucket, width]
            minimum = 2 if bucket == "first" else 5
            if len(rows) >= minimum:
                rates[width] = sum(n for n, _ in rows) / sum(ms for _, ms in rows)
        decision[bucket] = max(rates, key=lambda w: (rates[w], w)) if rates else 8
    return decision


@dataclass
class WidthTrial:
    path: Path
    request_id: int | None = None
    mode: str = "8"
    min_pairs: int = 0
    cpu_pair_tuning: tuple[int, int, int] = DEFAULT_CPU_PAIR_TUNING
    diagnostics: str = "off"
    transport: str = "raw"
    cpu_read_policy: str = "stock"
    hot_expert_limit: int | None = None
    selector_calibration: dict | None = None
    draft_budget: dict | None = None
    capture_draft_confidence: bool = False
    three_route_reuse: bool = False
    gpu_resident_layout: str | None = None
    draft_prefix_selection: bool = False
    draft_teacher: list[int] | None = None
    emitted: int = 0
    table: dict = field(default_factory=dict)
    passes: list = field(default_factory=list)
    pass_start: float = 0.0

    @property
    def log(self):
        return self.path.with_suffix(".passes.jsonl")

    @property
    def diagnostic_log(self):
        return self.path.with_suffix(".details.jsonl")

    @property
    def uses_hot_source(self):
        return self.transport in ("hot", *NATIVE_TRANSPORTS) or self.transport in GATHER_GEOMETRIES

    def begin(self, request_id):
        if request_id == self.request_id:
            return False
        setup = json.loads(self.path.read_text())
        self.mode = str(setup["width"])
        self.min_pairs = int(setup.get("min_pairs", 0))
        self.cpu_pair_tuning = validate_cpu_pair_tuning(setup.get("cpu_pair_tuning", DEFAULT_CPU_PAIR_TUNING))
        self.diagnostics = setup.get("diagnostics", "off")
        self.transport = setup.get("transport", "raw")
        self.cpu_read_policy = setup.get("cpu_read_policy", "stock")
        self.hot_expert_limit = setup.get("hot_expert_limit")
        self.selector_calibration = setup.get("selector")
        self.draft_budget = setup.get("draft_budget")
        self.capture_draft_confidence = setup.get("draft_confidence", False)
        self.three_route_reuse = setup.get("three_route_reuse", False)
        self.gpu_resident_layout = setup.get("gpu_resident_layout")
        self.draft_prefix_selection = setup.get("draft_prefix_selection", False)
        self.draft_teacher = setup.get("draft_teacher")
        budget_enabled = os.getenv("LIGHTREASON_DRAFT_BUDGET") == "1"
        budget_trial = budget_enabled and self.transport == SHARED_CPU_TRANSPORT
        resident_trial = bool(os.getenv("LIGHTREASON_GPU_RESIDENT_PLAN")) and self.transport in (
            *SHARED_CPU_TRANSPORTS, PACKED_PRIMARY_TRANSPORT,
        )
        joint_trial = (os.getenv("LIGHTREASON_DRAFT_PREFIX") == "1" and budget_trial and resident_trial
                       and self.gpu_resident_layout == "packed")
        if type(self.draft_prefix_selection) is not bool or (self.draft_prefix_selection and not joint_trial):
            raise ValueError("Prefix selection requires the paired DFlash2 and packed-resident benchmark")
        if self.draft_teacher is not None and (
                not joint_trial or self.mode != "8" or self.draft_prefix_selection
                or self.diagnostics != "profile" or not isinstance(self.draft_teacher, list)
                or not self.draft_teacher or len(self.draft_teacher) > 4096
                or any(type(token) is not int or not 0 <= token < 248320 for token in self.draft_teacher)):
            raise ValueError("Teacher continuations are only available to the profiled Qwen benchmark")
        if self.three_route_reuse and (os.getenv("LIGHTREASON_CPU_THREE_ROUTES") != "1"
                                       or not (budget_trial or resident_trial) or (self.mode != "8" and not joint_trial)):
            raise ValueError("Three-route reuse needs a prepared paired graph")
        if self.gpu_resident_layout is not None and (
                not resident_trial or self.transport not in {SHARED_CPU_TRANSPORT, PACKED_PRIMARY_TRANSPORT}
                or (self.mode != "8" and not joint_trial)
                or self.gpu_resident_layout not in {"raw", "packed"}):
            raise ValueError("GPU residency needs the prepared eight-position shared graph")
        if self.transport == PACKED_PRIMARY_TRANSPORT and (not os.getenv("LIGHTREASON_PACKED_PRIMARY_REPORT")
                                                           or self.gpu_resident_layout != "packed"):
            raise ValueError("The packed primary graph requires a prepared packed GPU source")
        self.table = setup.get("table", {})
        if self.mode not in {"4", "6", "8", "rule", "learned", "draft-budget"} or not 0 <= self.min_pairs <= 32:
            raise ValueError("Invalid verifier width or CPU expert pair threshold")
        if self.mode == "learned" and (set(self.table) != set(STATES)
                                       or not all(w in WIDTHS for w in self.table.values())):
            raise ValueError("Invalid verifier decision table")
        if (self.diagnostics not in {"off", "audit", "profile", "train"}
                or (self.diagnostics in {"audit", "train"} and self.mode != "8")
                or (self.diagnostics == "profile" and self.mode not in {"4", "8"}
                    and not (budget_trial and self.mode in {"6", "draft-budget"}))):
            raise ValueError("Invalid verifier diagnostics mode or width")
        if self.capture_draft_confidence and (not budget_trial or self.diagnostics != "profile"):
            raise ValueError("Confidence collection requires the profiled draft budget trial")
        if self.mode == "draft-budget":
            from edge35b.draft_budget import validate

            if not budget_trial or self.draft_budget is None:
                raise ValueError("Draft budgeting needs prepared shared graphs and calibration")
            validate(self.draft_budget)
        elif self.draft_budget is not None:
            raise ValueError("Calibration is only used by the draft budget policy")
        packed_modes = {str(width) for width in PACKED_TRANSFER_WIDTHS}
        if budget_trial:
            packed_modes.update(("6", "draft-budget"))
        if self.transport not in {"raw", "packed", "hot", *NATIVE_TRANSPORTS, *GATHER_GEOMETRIES} or (
                self.transport != "raw" and (
                    self.mode not in packed_modes
                    or self.diagnostics not in {"off", "profile"}
                )) or (self.uses_hot_source and self.mode != "8" and not budget_trial):
            raise ValueError("Packed BF16 transport requires a fixed width; hot sources require eight positions")
        if self.transport in GATHER_GEOMETRIES and self.diagnostics != "off":
            raise ValueError("Tiled gathers use unprofiled verifier graphs.")
        if self.cpu_read_policy != "stock":
            from edge35b.cpu_read_packing import CPU_READ_POLICIES

            if (self.cpu_read_policy not in CPU_READ_POLICIES or (self.mode != "8" and not budget_trial)
                    or self.min_pairs != 1
                    or self.transport not in NATIVE_TRANSPORTS):
                raise ValueError("CPU read trials require the paired eight-position native CUDA verifier.")
        if self.transport in SHARED_CPU_TRANSPORTS and self.cpu_read_policy == "stock":
            raise ValueError("The shared CUDA gather requires the prepared packed CPU read policy.")
        if self.hot_expert_limit is not None and (
                not self.uses_hot_source or type(self.hot_expert_limit) is not int
                or not 1 <= self.hot_expert_limit <= 128 or self.transport in NATIVE_TRANSPORTS):
            raise ValueError("Invalid requested hot expert budget")
        if self.selector_calibration is not None:
            from math import isfinite

            scales = self.selector_calibration["scale"]
            biases = self.selector_calibration["bias"]
            if (self.mode != "8" or len(scales) != 7 or len(biases) != 7
                    or not all(isfinite(s) and 0.5 <= s <= 2.0 for s in scales)
                    or not all(len(row) == 16 and all(isfinite(b) and -2 <= b <= 2 for b in row) for row in biases)):
                raise ValueError("Invalid DFlash2 score adjustment")
        self.request_id = request_id
        self.emitted = 0
        self.passes.clear()
        return True

    @property
    def needs_draft_confidence(self):
        return self.capture_draft_confidence or self.mode == "draft-budget"

    def choose(self, *, started=None, confidence=None, remaining=None):
        self.pass_start = perf_counter() if started is None else started
        if self.mode == "draft-budget":
            from edge35b.draft_budget import select_width

            return select_width(self.draft_budget, confidence, self.emitted, remaining)
        previous = self.passes[-1] if self.passes else None
        bucket = state(previous)
        if self.mode == "rule":
            return {"first": 8, "short": 4, "middle": 6, "long": 8}[bucket]
        if self.mode == "learned":
            return self.table[bucket]
        return int(self.mode)

    def observe(self, width, committed, finished):
        self.emitted += committed
        self.passes.append((width, committed, (perf_counter() - self.pass_start) * 1000))
        if finished:
            record = {"width": self.mode, "min_pairs": self.min_pairs, "transport": self.transport,
                      "cpu_pair_tuning": self.cpu_pair_tuning, "passes": self.passes,
                      "hot_expert_limit": self.hot_expert_limit, "cpu_read_policy": self.cpu_read_policy}
            if self.three_route_reuse:
                record["three_route_reuse"] = True
            if self.gpu_resident_layout is not None:
                record["gpu_resident_layout"] = self.gpu_resident_layout
            if self.draft_prefix_selection:
                record["draft_prefix_selection"] = True
            if self.draft_teacher is not None:
                record["draft_teacher"] = True
            with self.log.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
