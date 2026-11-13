"""Compile the Packed gate/activation experiment for Ada using NVRTC."""

import ctypes as C
from functools import lru_cache
from pathlib import Path

import torch


def checked(status, name):
    if status:
        raise RuntimeError(f"{name} returned CUDA status {status}")


@lru_cache(maxsize=1)
def load_gate():
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("The packed CUDA kernel is built for sm_89")
    candidates = [Path(torch.__file__).resolve().parents[1] / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.12"]
    try:
        compiler = C.CDLL("libnvrtc.so.12")
    except OSError:
        compiler = C.CDLL(str(candidates[0]))
    driver = C.CDLL("libcuda.so.1")
    pointer = C.c_void_p
    compiler.nvrtcCreateProgram.argtypes = [C.POINTER(pointer), C.c_char_p, C.c_char_p,
                                          C.c_int, pointer, pointer]
    compiler.nvrtcCompileProgram.argtypes = [pointer, C.c_int, C.POINTER(C.c_char_p)]
    compiler.nvrtcGetProgramLogSize.argtypes = [pointer, C.POINTER(C.c_size_t)]
    compiler.nvrtcGetProgramLog.argtypes = [pointer, pointer]
    compiler.nvrtcGetCUBINSize.argtypes = [pointer, C.POINTER(C.c_size_t)]
    compiler.nvrtcGetCUBIN.argtypes = [pointer, pointer]
    compiler.nvrtcDestroyProgram.argtypes = [C.POINTER(pointer)]
    driver.cuModuleLoadData.argtypes = [C.POINTER(pointer), pointer]
    driver.cuModuleGetFunction.argtypes = [C.POINTER(pointer), pointer, C.c_char_p]
    driver.cuLaunchKernel.argtypes = [pointer, *([C.c_uint] * 7), pointer, C.POINTER(pointer), pointer]
    program = pointer()
    source = Path(__file__).with_suffix(".cu").read_bytes()
    checked(compiler.nvrtcCreateProgram(C.byref(program), source, b"packed_ada.cu", 0, None, None), "nvrtcCreateProgram")
    try:
        options = (C.c_char_p * 3)(b"--gpu-architecture=sm_89", b"--fmad=false", b"--std=c++17")
        status = compiler.nvrtcCompileProgram(program, len(options), options)
        if status:
            length = C.c_size_t()
            compiler.nvrtcGetProgramLogSize(program, C.byref(length))
            log = C.create_string_buffer(length.value)
            compiler.nvrtcGetProgramLog(program, log)
            raise RuntimeError(log.value.decode(errors="replace"))
        length = C.c_size_t()
        checked(compiler.nvrtcGetCUBINSize(program, C.byref(length)), "nvrtcGetCUBINSize")
        binary = C.create_string_buffer(length.value)
        checked(compiler.nvrtcGetCUBIN(program, binary), "nvrtcGetCUBIN")
    finally:
        compiler.nvrtcDestroyProgram(C.byref(program))
    module, function = pointer(), pointer()
    checked(driver.cuModuleLoadData(C.byref(module), binary), "cuModuleLoadData")
    checked(driver.cuModuleGetFunction(C.byref(function), module, b"packed_gate_silu"), "cuModuleGetFunction")

    def launch(table, hidden, active):
        args = [pointer(t.data_ptr()) for t in (table, hidden, active)]
        references = (pointer * len(args))(*(C.addressof(value) for value in args))
        stream = torch.cuda.current_stream().cuda_stream
        checked(driver.cuLaunchKernel(function, 64, 8, 1, 256, 1, 1, 0, stream, references, None),
                "cuLaunchKernel")

    return launch
