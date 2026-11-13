"""Compile and capture the Ada-native lossless BF16 copy kernel."""

from ctypes import c_int32, c_void_p
from pathlib import Path


NATIVE_GEOMETRIES = (
    (4, 256, "default"), (8, 128, "default"), (8, 256, "default"), (16, 128, "default"),
    (8, 256, "l2"), (16, 128, "l2"),
)
CPU_TILE_GEOMETRIES = ((4, 256, "cpu-tiles"), (8, 128, "cpu-tiles"), (16, 128, "cpu-tiles"))


def name(geometry):
    return "cuda-" + "-".join(map(str, geometry))


def select_geometry(tag, *, shared=False):
    for geometry in CPU_TILE_GEOMETRIES if shared else NATIVE_GEOMETRIES:
        if name(geometry) == tag:
            return geometry
    raise ValueError(f"Unknown Ada gather geometry {tag!r}.")


def _checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA call failed: {status!r}")
    return values[0] if values else None


class NativeBf16Gather:
    def __init__(self):
        import torch
        from cuda.bindings import driver, nvrtc

        if torch.cuda.get_device_capability() != (8, 9):
            raise ValueError("The experimental native gather requires Ada CUDA capability 8.9.")
        torch.cuda.init()
        # Materialize PyTorch's primary context before loading a driver module.
        torch.empty(1, device="cuda")
        source = Path(__file__).with_name("bf16_gather_ada.cu").read_bytes()
        program = _checked(nvrtc.nvrtcCreateProgram(source, b"bf16_gather_ada.cu", 0, None, None))
        try:
            options = [b"--gpu-architecture=sm_89", b"--std=c++17"]
            status, = nvrtc.nvrtcCompileProgram(program, len(options), options)
            if int(status):
                log = bytearray(_checked(nvrtc.nvrtcGetProgramLogSize(program)))
                _checked(nvrtc.nvrtcGetProgramLog(program, log))
                raise RuntimeError(log.rstrip(b"\0").decode(errors="replace"))
            cubin = bytearray(_checked(nvrtc.nvrtcGetCUBINSize(program)))
            _checked(nvrtc.nvrtcGetCUBIN(program, cubin))
        finally:
            _checked(nvrtc.nvrtcDestroyProgram(program))

        self.driver = driver
        self.torch = torch
        self.module = _checked(driver.cuModuleLoadData(cubin))
        self.functions = {policy: _checked(driver.cuModuleGetFunction(self.module, symbol))
                          for policy, symbol in (("default", b"gather_bf16"), ("l2", b"gather_bf16_l2"),
                                                 ("cpu-tiles", b"gather_bf16_cpu_tiles"),
                                                 ("resident", b"gather_bf16_resident"))}
        attributes = driver.CUfunction_attribute
        self.binary_version = _checked(driver.cuFuncGetAttribute(attributes.CU_FUNC_ATTRIBUTE_BINARY_VERSION,
                                                                  self.functions["default"]))
        if self.binary_version != 89:
            raise RuntimeError(f"Unexpected CUDA binary architecture {self.binary_version}.")
        self.primary = _checked(driver.cuModuleGetFunction(self.module, b"stage_bf16_primary"))
        self.compact_tail = _checked(driver.cuModuleGetFunction(self.module, b"gather_bf16_compact_tail"))

    def copy(self, descriptors, destinations, sizes, src, dst, count, *, max_copies, experts, geometry):
        stripes, threads, policy = geometry
        pointers = tuple(t.data_ptr() for t in (descriptors, destinations, sizes, src, dst, count))
        arguments = (pointers + (experts,), (c_void_p,) * len(pointers) + (c_int32,))
        stream = self.driver.CUstream(self.torch.cuda.current_stream(destinations.device).cuda_stream)
        _checked(self.driver.cuLaunchKernel(self.functions[policy], stripes, destinations.numel(), max_copies,
                                           threads, 1, 1, 0, stream, arguments, 0))

    def stage_primary(self, sources, miss_sources, layouts, prefix_tables, plan, count, *, experts, packed_slots):
        pointers = tuple(t.data_ptr() for t in (sources, miss_sources, layouts, prefix_tables, plan, count))
        rows = plan.shape[1]
        arguments = (pointers + (experts, packed_slots, rows), (c_void_p,) * len(pointers) + (c_int32,) * 3)
        stream = self.driver.CUstream(self.torch.cuda.current_stream(plan.device).cuda_stream)
        _checked(self.driver.cuLaunchKernel(self.primary, 4, sources.shape[0], rows, 256, 1, 1, 0, stream, arguments, 0))

    def copy_compact_tail(self, sources, destinations, sizes, prefixes, tail, src, dst, count, entries, resident, experts):
        tensors = (sources, destinations, sizes, prefixes, tail, src, dst, count, entries, resident)
        pointers = tuple(t.data_ptr() for t in tensors)
        rows = entries.shape[1]
        arguments = (pointers + (experts, rows), (c_void_p,) * len(pointers) + (c_int32,) * 2)
        stream = self.driver.CUstream(self.torch.cuda.current_stream(src.device).cuda_stream)
        _checked(self.driver.cuLaunchKernel(self.compact_tail, 4, sources.shape[0], rows, 256, 1, 1, 0,
                                           stream, arguments, 0))
