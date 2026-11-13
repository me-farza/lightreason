"""Read-only hardware telemetry, with optional bounded stock GPU workloads.

Default: inventory and two seconds of idle telemetry; no CUDA workload.
Explicit load probe: python hardware_exact10.py --run-load --out hardware.json
External benchmark: python hardware_exact10.py --out monitored.json \
    --command-timeout-s 300 --monitor-command .venv/bin/python benchmark.py
All wrapper options must precede --monitor-command. External stdout/stderr are
inherited unchanged. On normal completion the wrapper propagates the command's
exit status; guard failures return125 and command timeout returns124. The signed
original return code remains in the JSON report. CUDA_VISIBLE_DEVICES is set to
the monitored GPU UUID for the child process; no device settings change.

The sensor process samples NVML/CPU temperatures every250ms. A separate parent
watchdog stops the workload on GPU>=70C, CPU>=75C where available, stale/erroring
temperature telemetry, a changed450W power limit, excessive measured power, or
timeouts. Thresholds can only be lowered. These are sampled stop conditions,
not a guarantee that physical temperature cannot overshoot between samples.

No hardware setters exist here: no clock, power, persistence, CPU governor/EPP,
voltage, firmware, driver, boot or reset changes. Stock settings remain intact;
process/CUDA-context and NVML-session cleanup occurs in finally. No firmware
unlocker is downloaded or executed. GPU stages require explicit --run-load.

Inference timings are diagnostics collected under monitoring. This is not a
replacement for the separate exact-output gate or a clean latency benchmark.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import statistics
import subprocess
import sys
import time
import traceback


SOURCES = {
    "nvidia_smi": "https://docs.nvidia.com/deploy/nvidia-smi/index.html",
    "nvml_queries": "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html",
    "unofficial_report_read_only": "https://github.com/JustAResearcher/gpu-unlock-ada/blob/main/rtx4090-register-report.md",
}


class SensorError(RuntimeError):
    pass


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class ReadOnlyNVML:
    """Minimal ctypes bindings to initialization, shutdown and Get APIs only."""
    def __init__(self, index):
        library = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        self.lib = ctypes.CDLL(library)
        self.optional_errors = {}
        self.handle = ctypes.c_void_p()
        self.initialized = False
        self._call("nvmlInit_v2", [])
        self.initialized = True
        try:
            self._call("nvmlDeviceGetHandleByIndex_v2", [ctypes.c_uint(index), ctypes.byref(self.handle)])
        except Exception:
            self.close()
            raise

    def _call(self, name, arguments, required=True):
        function = getattr(self.lib, name, None)
        if function is None:
            message = f"NVML function unavailable: {name}"
            if required:
                raise SensorError(message)
            self.optional_errors[name] = message
            return False
        function.restype = ctypes.c_int
        status = function(*arguments)
        if status:
            message = f"{name} returned NVML status {status}"
            if required:
                raise SensorError(message)
            self.optional_errors[name] = message
            return False
        return True

    def uint(self, name, *arguments, required=False):
        result = ctypes.c_uint()
        ok = self._call(name, [self.handle, *arguments, ctypes.byref(result)], required)
        return result.value if ok else None

    def text(self, name):
        result = ctypes.create_string_buffer(128)
        ok = self._call(name, [self.handle, result, ctypes.c_uint(len(result))], False)
        return result.value.decode(errors="replace") if ok else None

    def clock(self, kind, maximum=False):
        return self.uint("nvmlDeviceGetMaxClockInfo" if maximum else "nvmlDeviceGetClockInfo", ctypes.c_uint(kind))

    def supported_clocks(self, name, *arguments):
        function = getattr(self.lib, name, None)
        if function is None:
            return None
        function.restype = ctypes.c_int
        count = ctypes.c_uint(0)
        status = function(self.handle, *arguments, ctypes.byref(count), None)
        # NVML_INSUFFICIENT_SIZE=7 is the documented count-query response.
        if status not in (0, 7) or not 0 < count.value <= 4096:
            self.optional_errors[name] = f"count query status={status}, count={count.value}"
            return None
        result = (ctypes.c_uint * count.value)()
        if not self._call(name, [self.handle, *arguments, ctypes.byref(count), result], False):
            return None
        return list(result[:count.value])

    def inventory(self):
        low, high = ctypes.c_uint(), ctypes.c_uint()
        constraints = self._call("nvmlDeviceGetPowerManagementLimitConstraints",
                                 [self.handle, ctypes.byref(low), ctypes.byref(high)], False)
        memory = self.supported_clocks("nvmlDeviceGetSupportedMemoryClocks")
        selected_memory = max(memory) if memory else None
        graphics = self.supported_clocks("nvmlDeviceGetSupportedGraphicsClocks", ctypes.c_uint(selected_memory)) if selected_memory else None
        default = self.uint("nvmlDeviceGetPowerManagementDefaultLimit", required=True)
        count = ctypes.c_uint()
        self._call("nvmlDeviceGetCount_v2", [ctypes.byref(count)], True)
        return {
            "name": self.text("nvmlDeviceGetName"), "uuid": self.text("nvmlDeviceGetUUID"),
            "physical_gpu_count": count.value,
            "power_default_w": default / 1000,
            "power_constraints_w": [low.value / 1000, high.value / 1000] if constraints else None,
            "maximum_graphics_clock_mhz": self.clock(0, True),
            "maximum_memory_clock_mhz": self.clock(2, True),
            "maximum_pcie_generation": self.uint("nvmlDeviceGetMaxPcieLinkGeneration"),
            "maximum_pcie_width": self.uint("nvmlDeviceGetMaxPcieLinkWidth"),
            "supported_memory_clocks_mhz": memory,
            "supported_graphics_clocks_mhz_at_reported_memory": graphics,
            "supported_graphics_memory_clock_mhz": selected_memory,
            "effective_uid": os.geteuid(), "hardware_changes_applied": False,
            "permission_note": "Read queries do not prove permission/support for clock writes; no write is attempted",
            "optional_query_errors": dict(self.optional_errors),
        }

    def sample(self, *, safety_only=False):
        temperature = self.uint("nvmlDeviceGetTemperature", ctypes.c_uint(0), required=True)
        if not 0 <= temperature <= 125:
            raise SensorError(f"Implausible mandatory GPU temperature: {temperature}")
        power_limit = self.uint("nvmlDeviceGetPowerManagementLimit", required=True)
        power = self.uint("nvmlDeviceGetPowerUsage")
        sample = {"gpu_temperature_c": temperature, "power_limit_w": power_limit / 1000,
                  "power_draw_w": power / 1000 if power is not None else None}
        if safety_only:
            return sample
        state = self.uint("nvmlDeviceGetPerformanceState")
        utilization = Utilization()
        has_util = self._call("nvmlDeviceGetUtilizationRates", [self.handle, ctypes.byref(utilization)], False)
        reasons = ctypes.c_ulonglong()
        reason_api = "nvmlDeviceGetCurrentClocksEventReasons" if hasattr(self.lib, "nvmlDeviceGetCurrentClocksEventReasons") else "nvmlDeviceGetCurrentClocksThrottleReasons"
        has_reasons = self._call(reason_api, [self.handle, ctypes.byref(reasons)], False)
        return {**sample,
                "graphics_clock_mhz": self.clock(0), "sm_clock_mhz": self.clock(1),
                "memory_clock_mhz": self.clock(2), "pstate": f"P{state}" if state is not None else None,
                "pcie_generation": self.uint("nvmlDeviceGetCurrPcieLinkGeneration"),
                "pcie_width": self.uint("nvmlDeviceGetCurrPcieLinkWidth"),
                "gpu_utilization_percent": utilization.gpu if has_util else None,
                "memory_utilization_percent": utilization.memory if has_util else None,
                "clock_event_reasons_mask": hex(reasons.value) if has_reasons else None}

    def close(self):
        if self.initialized:
            self._call("nvmlShutdown", [], False)
            self.initialized = False


def read_text(path):
    return Path(path).read_text().strip()


def cpu_inventory(sysfs=Path("/sys")):
    policies = []
    for policy in sorted((sysfs / "devices/system/cpu/cpufreq").glob("policy*")):
        row = {"policy": policy.name}
        for name in ["scaling_driver", "scaling_governor", "energy_performance_preference",
                     "scaling_min_freq", "scaling_max_freq", "cpuinfo_min_freq", "cpuinfo_max_freq"]:
            path = policy / name
            if path.exists():
                try:
                    row[name] = read_text(path)
                except OSError as exc:
                    # Offline logical CPUs retain cpufreq policy directories;
                    # their diagnostic controls can return EBUSY. They are not
                    # thermal safety sensors and do not invalidate monitoring.
                    row[name] = None
                    row.setdefault("optional_read_errors", {})[name] = f"{type(exc).__name__}: {exc}"
        policies.append(row)
    flags = {}
    for name, relative in [("boost", "devices/system/cpu/cpufreq/boost"),
                           ("amd_pstate_status", "devices/system/cpu/amd_pstate/status")]:
        path = sysfs / relative
        if path.exists():
            try:
                flags[name] = read_text(path)
            except OSError as exc:
                flags.setdefault("optional_read_errors", {})[name] = f"{type(exc).__name__}: {exc}"
    sensors = []
    cpu_names = {"k10temp", "coretemp", "zenpower", "cpu_thermal"}
    for hwmon in sorted((sysfs / "class/hwmon").glob("hwmon*")):
        name_path = hwmon / "name"
        if not name_path.exists() or read_text(name_path).lower() not in cpu_names:
            continue
        for sensor in sorted(hwmon.glob("temp*_input")):
            label_path = sensor.with_name(sensor.name.replace("_input", "_label"))
            sensors.append({"path": str(sensor), "label": read_text(label_path) if label_path.exists() else sensor.stem,
                            "driver": read_text(name_path)})
    if not sensors:
        for zone in sorted((sysfs / "class/thermal").glob("thermal_zone*")):
            kind = zone / "type"
            if kind.exists() and read_text(kind).lower() in {"x86_pkg_temp", "cpu-thermal", "cpu_thermal"}:
                sensors.append({"path": str(zone / "temp"), "label": read_text(kind), "driver": "thermal_zone"})
    return {"policies": policies, **flags, "temperature_sensors": sensors,
            "cpu_temperature_available": bool(sensors), "configuration_changes_applied": False}


def cpu_sample(inventory, sysfs=Path("/sys"), *, safety_only=False):
    temperatures = []
    for sensor in inventory["temperature_sensors"]:
        try:
            value = float(read_text(sensor["path"])) / 1000
        except Exception as exc:
            raise SensorError(f"Previously available CPU temperature sensor failed: {sensor['path']}: {exc}") from exc
        if not math.isfinite(value) or not -10 <= value <= 125:
            raise SensorError(f"Implausible CPU temperature: {sensor['path']}={value}")
        temperatures.append({"label": sensor["label"], "temperature_c": value})
    sample = {"cpu_temperatures": temperatures,
              "cpu_temperature_max_c": max((row["temperature_c"] for row in temperatures), default=None)}
    if safety_only:
        return sample
    frequencies = []
    for path in (sysfs / "devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq"):
        try:
            frequencies.append(float(read_text(path)) / 1000)
        except (OSError, ValueError):
            pass  # Frequency is optional diagnostics, not a thermal safety sensor.
    return {**sample,
            "cpu_frequency_mhz_min": min(frequencies, default=None),
            "cpu_frequency_mhz_median": statistics.median(frequencies) if frequencies else None,
            "cpu_frequency_mhz_max": max(frequencies, default=None)}


def safety_violation(sample, gpu_stop_c=70, cpu_stop_c=75, power_limit_w=450):
    gpu = sample.get("gpu_temperature_c")
    if not isinstance(gpu, (int, float)) or not math.isfinite(gpu):
        return "Mandatory GPU temperature is unavailable or invalid"
    if gpu >= gpu_stop_c:
        return f"GPU temperature stop: {gpu}C >= {gpu_stop_c}C"
    cpu = sample.get("cpu_temperature_max_c")
    if cpu is not None and (not math.isfinite(cpu) or cpu >= cpu_stop_c):
        return f"CPU temperature stop: {cpu}C >= {cpu_stop_c}C"
    if sample.get("power_limit_w") != power_limit_w:
        return f"Power limit changed or is unverifiable: {sample.get('power_limit_w')}W; require {power_limit_w}W"
    draw = sample.get("power_draw_w")
    if draw is not None and draw > power_limit_w:
        return f"Measured GPU power exceeds requested ceiling: {draw}W > {power_limit_w}W"
    return None


def sensor_process(options, output, stop):
    nvml = None
    try:
        nvml = ReadOnlyNVML(options["gpu_index"])
        cpu = cpu_inventory()
        output.put({"type": "inventory", "gpu": nvml.inventory(), "cpu": cpu})
        while not stop.is_set():
            start = time.monotonic()
            safety_only = options.get("safety_sampling", False)
            sample = {"type": "sample", "monotonic_s": start, **nvml.sample(safety_only=safety_only),
                      **cpu_sample(cpu, safety_only=safety_only)}
            output.put(sample)
            stop.wait(max(0, 0.25 - (time.monotonic() - start)))
    except BaseException as exc:
        output.put({"type": "sensor_error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        if nvml is not None:
            nvml.close()


def load_process(options, output, stop):
    """Only spawned after --run-load and a healthy initial sensor sample."""
    try:
        output.put({"type": "phase", "name": "setup", "timeout_s": options["setup_timeout_s"]})
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from benchmark import Reference, PROMPTS
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(options["gpu_index"])
        properties = torch.cuda.get_device_properties(options["gpu_index"])
        cuda_uuid = getattr(properties, "uuid", None)
        normalize_uuid = lambda value: str(value).lower().replace("gpu-", "").replace("-", "")
        if cuda_uuid is not None:
            if normalize_uuid(cuda_uuid) != normalize_uuid(options["expected_gpu_uuid"]):
                raise RuntimeError("CUDA workload device differs from the NVML-monitored GPU")
        elif not (options["physical_gpu_count"] == torch.cuda.device_count() == 1 and options["gpu_index"] == 0
                  and not os.environ.get("CUDA_VISIBLE_DEVICES")):
            raise RuntimeError("Cannot prove CUDA/NVML device mapping; refusing unmonitored workload")
        output.put({"type": "cuda_device", "uuid": str(cuda_uuid), "name": properties.name})
        parent = mp.parent_process()
        def stopping():
            return stop.is_set() or (parent is not None and not parent.is_alive())
        model = AutoModelForCausalLM.from_pretrained(options["model_path"], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
        tokenizer = AutoTokenizer.from_pretrained(options["model_path"], local_files_only=True)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": PROMPTS[0]}],
            add_generation_prompt=True, return_tensors="pt").cuda()
        reference = Reference(model)
        expected, _ = reference.run(ids, options["tokens"])
        module, name = options["runner_factory"].rsplit(":", 1)
        args = argparse.Namespace(model_path=options["model_path"], max_len=options["max_len"])
        adapter = getattr(importlib.import_module(module), name)(model=model, tokenizer=tokenizer, args=args)
        exact = getattr(adapter, "runner", adapter)
        if not hasattr(exact, "run"):
            raise TypeError("Runner factory must expose run(ids,n) or .runner.run(ids,n)")
        extra = {"eos_ids": None} if "eos_ids" in inspect.signature(exact.run).parameters else {}
        tokens, _ = exact.run(ids, options["tokens"], **extra)
        if tokens != expected:
            raise RuntimeError("Exact runner failed the short reference output check")
        if stopping():
            output.put({"type": "stopped"})
            return
        for name, runner in [("reference", reference), ("exact", exact)]:
            output.put({"type": "phase", "name": name, "timeout_s": options["stage_seconds"] + 1})
            torch.cuda.synchronize()
            start, timings = time.perf_counter(), []
            while time.perf_counter() - start < options["stage_seconds"] and not stopping():
                current, metrics = runner.run(ids, options["tokens"], **(extra if name == "exact" else {}))
                if current != expected:
                    raise RuntimeError(f"{name} output changed during the monitored probe")
                timings.append(metrics)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            output.put({"type": "stage_result", "stage": name, "elapsed_s": elapsed,
                        "requests": len(timings), "fixed_tokens_per_request": options["tokens"],
                        "median_generation_s": statistics.median(row["total_s"] for row in timings) if timings else None,
                        "median_decode_tokens_s": statistics.median(row["decode_tokens_s"] for row in timings) if timings else None,
                        "all_short_probe_tokens_exact": True})
            if stopping():
                output.put({"type": "stopped"})
                return
        output.put({"type": "phase", "name": "gemm_setup", "timeout_s": 5})
        size = options["gemm_size"]
        a = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        destination = torch.empty_like(a)
        with torch.no_grad():
            for _ in range(3):
                torch.mm(a, b, out=destination)
            torch.cuda.synchronize()
            output.put({"type": "phase", "name": "stock_bf16_gemm", "timeout_s": options["gemm_seconds"] + 1})
            start, count = time.perf_counter(), 0
            while time.perf_counter() - start < options["gemm_seconds"] and not stopping():
                for _ in range(4):
                    torch.mm(a, b, out=destination)
                    count += 1
                # Bound queued GPU work instead of queuing kernels for seconds.
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        output.put({"type": "stage_result", "stage": "stock_bf16_gemm", "elapsed_s": elapsed,
                    "matrix_size": size, "iterations": count,
                    "diagnostic_dense_tflops_including_sync_overhead": count * 2 * size ** 3 / elapsed / 1e12,
                    "dispatch": "Stock torch.mm BF16; actual Tensor Core kernel use is not proved without profiling"})
        output.put({"type": "done"})
    except BaseException as exc:
        output.put({"type": "load_error", "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})


def stop_process(process, stop):
    stop.set()
    if process is None:
        return
    process.join(0.15)
    if process.is_alive():
        process.terminate()
        process.join(0.5)
    if process.is_alive():
        process.kill()
        process.join(0.5)


def start_monitored_command(command, gpu_uuid, stdout=None, stderr=None):
    """Launch an argument vector directly, inheriting normal logs, no shell."""
    if not command or not gpu_uuid:
        raise ValueError("External command and monitored GPU UUID are required")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return subprocess.Popen(command, env=environment, start_new_session=True,
                            stdout=stdout, stderr=stderr)


def stop_monitored_command(process, termination_signal=signal.SIGTERM):
    """End the whole child session on guard failure, including GPU descendants."""
    if process is None:
        return
    try:
        os.killpg(process.pid, termination_signal)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


def telemetry_summary(samples):
    result = {}
    for phase in dict.fromkeys(row["phase"] for row in samples):
        rows = [row for row in samples if row["phase"] == phase]
        fields = {}
        for name in ["gpu_temperature_c", "cpu_temperature_max_c", "power_draw_w", "graphics_clock_mhz",
                     "memory_clock_mhz", "gpu_utilization_percent", "pcie_generation", "pcie_width",
                     "cpu_frequency_mhz_median"]:
            values = [row[name] for row in rows if isinstance(row.get(name), (int, float))]
            fields[name] = {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else None
        result[phase] = {"samples": len(rows), "ranges": fields,
                         "pstates": sorted({row["pstate"] for row in rows if row.get("pstate")}),
                         "clock_event_masks": sorted({row["clock_event_reasons_mask"] for row in rows if row.get("clock_event_reasons_mask")})}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--runner-factory", default="exact_graph:make_quality_runner")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--idle-seconds", type=float, default=2)
    parser.add_argument("--stage-seconds", type=float, default=2)
    parser.add_argument("--gemm-seconds", type=float, default=1.5)
    parser.add_argument("--gemm-size", type=int, default=2048)
    parser.add_argument("--gpu-stop-c", type=float, default=70)
    parser.add_argument("--cpu-stop-c", type=float, default=75)
    parser.add_argument("--setup-timeout-s", type=float, default=45)
    parser.add_argument("--total-timeout-s", type=float, default=60)
    parser.add_argument("--command-timeout-s", type=float, default=300)
    parser.add_argument("--out", default="results20x/hardware-readonly.json")
    parser.add_argument("--final-report-only", action="store_true",
                        help="Write the report after the workload stops, keeping disk I/O out of the safety loop")
    parser.add_argument("--safety-sampling", action="store_true",
                        help="Sample temperatures and power without optional clock and utilization queries")
    parser.add_argument("--monitor-command", nargs=argparse.REMAINDER,
                        help="Run an external argument vector with inherited stdout/stderr under the safety watchdog; put all wrapper options first")
    args = parser.parse_args()
    if args.monitor_command is not None:
        if args.monitor_command[:1] == ["--"]:
            args.monitor_command = args.monitor_command[1:]
        if not args.monitor_command:
            parser.error("--monitor-command requires an executable and optional arguments")
        if args.run_load:
            parser.error("Choose either --run-load or --monitor-command")
    if not 1 <= args.gpu_stop_c <= 70 or not 1 <= args.cpu_stop_c <= 75:
        parser.error("Thermal thresholds may be lowered but cannot exceed GPU70C / CPU75C")
    if not 0.25 <= args.stage_seconds <= 3 or not 0.25 <= args.gemm_seconds <= 2 or not 0.25 <= args.idle_seconds <= 5:
        parser.error("Require stage0.25..3s, GEMM0.25..2s and idle0.25..5s")
    if not 1 <= args.setup_timeout_s <= 45 or not 5 <= args.total_timeout_s <= 60:
        parser.error("Setup must be<=45s and total timeout5..60s")
    if not 1 <= args.command_timeout_s <= 3600:
        parser.error("External command timeout must be1..3600s")
    if not 2 <= args.tokens <= 32 or not 512 <= args.gemm_size <= 4096 or args.gpu_index < 0:
        parser.error("Require tokens2..32, GEMM size512..4096 and nonnegative GPU index")
    context = mp.get_context("spawn")
    sensor_queue, load_queue = context.Queue(), context.Queue()
    sensor_stop, load_stop = context.Event(), context.Event()
    sensors = context.Process(target=sensor_process, args=(vars(args), sensor_queue, sensor_stop), daemon=True)
    workload = None
    external = None
    external_returncode = None
    wrapper_exit_code = 0
    interrupted_signal = None
    samples, stages, events = [], [], []
    inventory, outcome, reason = None, "running", None
    phase, phase_deadline = "idle", None
    start = time.monotonic()
    last_sample = None
    idle_start = None
    done_at = None
    total_timeout = (args.command_timeout_s + 2 * args.idle_seconds + 5
                     if args.monitor_command else args.total_timeout_s)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    def report():
        return {"arguments": vars(args), "outcome": outcome, "reason": reason,
                "hardware_changes_applied": False, "power_limit_required_w": 450,
                "sample_interval_s": 0.25, "sensor_stale_abort_s": 0.75,
                "effective_total_timeout_s": total_timeout,
                "external_command": args.monitor_command,
                "external_returncode": external_returncode,
                "wrapper_exit_code": wrapper_exit_code,
                "external_logs": "stdout/stderr inherited unchanged; wrapper completion goes to stderr" if args.monitor_command else None,
                "external_cuda_visible_devices": inventory["gpu"]["uuid"] if inventory and args.monitor_command else None,
                "inventory": inventory, "telemetry_by_phase": telemetry_summary(samples),
                "stages": stages, "events": events, "samples": samples, "sources": SOURCES,
                "limits": ["Sampled thermal stops cannot prevent every between-sample overshoot.",
                           "Idle P8/low clocks or PCIe Gen1 alone are not evidence of an inference bottleneck.",
                           "Resident-weight decode need not be limited by PCIe; compare under-load telemetry and kernel profiles.",
                           "SIMT GEMV decode is not accelerated by a Tensor Core instruction-rate change.",
                           "This probe changes no manufacturer controls; unofficial firmware/register unlocks are outside this experiment.",
                           "Monitoring overhead and short stages mean these are hardware diagnostics, not a10x performance proof."]}
    sensors.start()
    def interrupted(signum, frame):
        nonlocal interrupted_signal
        interrupted_signal = signum
        raise KeyboardInterrupt(f"Signal {signum}")
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        while True:
            now = time.monotonic()
            if now - start > total_timeout:
                raise TimeoutError("Total bounded probe timeout")
            try:
                event = sensor_queue.get(timeout=0.05)
            except queue.Empty:
                event = None
            if event:
                if event["type"] == "inventory":
                    inventory = event
                    if inventory["gpu"]["power_default_w"] != 450:
                        raise SensorError("Manufacturer default power is not the required450W; no change attempted")
                elif event["type"] == "sensor_error":
                    raise SensorError(event["message"])
                elif event["type"] == "sample":
                    last_sample = event["monotonic_s"]
                    event.update({"elapsed_s": now - start, "phase": phase})
                    samples.append(event)
                    failure = safety_violation(event, args.gpu_stop_c, args.cpu_stop_c)
                    if failure:
                        raise SensorError(failure)
                    if idle_start is None:
                        idle_start = now
            if last_sample is None:
                if now - start > 3:
                    raise SensorError("No healthy sensor sample within3s; no CUDA workload started")
            elif now - last_sample > 0.75:
                raise SensorError("Temperature telemetry stale for>750ms")
            if not sensors.is_alive() and event is None:
                raise SensorError("Sensor process exited")
            if idle_start is not None and workload is None and external is None and done_at is None and now - idle_start >= args.idle_seconds:
                if args.monitor_command:
                    try:
                        external = start_monitored_command(args.monitor_command, inventory["gpu"]["uuid"])
                    except (FileNotFoundError, PermissionError) as exc:
                        external_returncode = 127 if isinstance(exc, FileNotFoundError) else 126
                        reason = f"{type(exc).__name__}: {exc}"
                        events.append({"type": "external_launch_error", "message": reason,
                                       "returncode": external_returncode})
                        done_at = now
                        phase, phase_deadline = "post_command_idle", None
                    else:
                        events.append({"type": "external_start", "pid": external.pid,
                                       "command": args.monitor_command})
                        phase, phase_deadline = "external_command", now + args.command_timeout_s
                elif not args.run_load:
                    outcome = "completed_read_only_idle"
                    break
                else:
                    worker_options = {**vars(args), "expected_gpu_uuid": inventory["gpu"]["uuid"],
                                      "physical_gpu_count": inventory["gpu"]["physical_gpu_count"]}
                    workload = context.Process(target=load_process, args=(worker_options, load_queue, load_stop))
                    workload.start()
                    phase = "setup"
                    phase_deadline = now + args.setup_timeout_s
            while True:
                try:
                    message = load_queue.get_nowait()
                except queue.Empty:
                    break
                events.append(message)
                if message["type"] == "phase":
                    phase = message["name"]
                    phase_deadline = time.monotonic() + message["timeout_s"]
                elif message["type"] == "stage_result":
                    stages.append(message)
                elif message["type"] == "load_error":
                    raise SensorError(message["message"])
                elif message["type"] == "done":
                    done_at = time.monotonic()
                    phase, phase_deadline = "post_load_idle", None
            if external is not None and done_at is None and external.poll() is not None:
                external_returncode = external.returncode
                events.append({"type": "external_exit", "returncode": external_returncode})
                done_at = time.monotonic()
                phase, phase_deadline = "post_command_idle", None
            if phase_deadline is not None and now > phase_deadline:
                raise TimeoutError(f"Bounded stage timeout: {phase}")
            if workload is not None and not workload.is_alive() and done_at is None:
                # A queue feeder can briefly lag process exit; consume it on
                # the next poll before treating this as a missing completion.
                workload.join(0)
                if workload.exitcode not in (None, 0):
                    raise SensorError(f"Workload exited with status{workload.exitcode}")
            if done_at is not None and now - done_at >= args.idle_seconds:
                if args.monitor_command:
                    outcome = "completed_external_command" if external_returncode == 0 else "external_command_failed"
                    wrapper_exit_code = external_returncode if external_returncode >= 0 else 128 - external_returncode
                else:
                    outcome = "completed_stock_load_probe"
                break
            if not args.final_report_only:
                output.write_text(json.dumps(report(), indent=2))
    except BaseException as exc:
        outcome, reason = "aborted", f"{type(exc).__name__}: {exc}"
        if args.monitor_command:
            if isinstance(exc, KeyboardInterrupt):
                wrapper_exit_code = 128 + (interrupted_signal or signal.SIGINT)
            else:
                wrapper_exit_code = 124 if isinstance(exc, TimeoutError) else 125
        else:
            wrapper_exit_code = 2
    finally:
        # Stop CUDA work first, retaining thermal telemetry until its context
        # is gone. No device settings require restoration because none changed.
        try:
            stop_monitored_command(external, interrupted_signal or signal.SIGTERM)
        except subprocess.TimeoutExpired:
            events.append({"type": "external_shutdown_pending", "pid": external.pid})
        if external is not None:
            external_returncode = external.returncode
        stop_process(workload, load_stop)
        stop_process(sensors, sensor_stop)
        signal.signal(signal.SIGTERM, previous_sigterm)
        output.write_text(json.dumps(report(), indent=2))
    print(json.dumps({"outcome": outcome, "reason": reason, "out": str(output),
                      "external_returncode": external_returncode, "wrapper_exit_code": wrapper_exit_code,
                      "telemetry_by_phase": telemetry_summary(samples)}, indent=2),
          file=sys.stderr if args.monitor_command else sys.stdout)
    if wrapper_exit_code:
        raise SystemExit(wrapper_exit_code)


if __name__ == "__main__":
    main()
