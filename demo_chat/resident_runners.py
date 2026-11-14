"""Share audited BF16 experts and hand off CUDA memory between idle runners."""

from array import array
import ctypes
import json
import logging
import mmap
import os
from pathlib import Path
from shutil import disk_usage
import socket
from time import perf_counter
import warnings

from edge35b.download import GIB, available_memory


LAYERS = 40
EXPERT_SHAPES = {"gate_up": (256, 1024, 2048), "down": (256, 2048, 512)}
EXPERT_BYTES = {"gate_up": GIB, "down": GIB // 2}
SHARED_BYTES = LAYERS * sum(EXPERT_BYTES.values())
_bank_fd = None
# Linux uapi values. The workstation's Python build omits the seal constants.
F_ADD_SEALS = 1033
SEAL_SHARED_BANK = 0x01 | 0x02 | 0x04 | 0x10  # seal, shrink, grow, future write


def _layout():
    offset = 0
    for role, shape in EXPERT_SHAPES.items():
        for layer in range(LAYERS):
            yield role, layer, shape, offset
            offset += EXPERT_BYTES[role]


def shared_host_banks(host_bank, specs, layers):
    """Allocate the existing FreeToken bank layout in a single shareable RAM file."""
    import fcntl
    import torch

    global _bank_fd
    expected = {role: (shape, torch.bfloat16) for role, shape in EXPERT_SHAPES.items()}
    if _bank_fd is not None or specs != expected or layers != list(range(LAYERS)):
        raise RuntimeError("Shared banks require the pinned Qwen3.6 BF16 expert layout.")
    if available_memory() < SHARED_BYTES + 36 * GIB:
        raise MemoryError("Shared experts and parked CUDA allocations need 96 GiB of available RAM at startup.")
    fd = os.memfd_create("lightreason-bf16-experts", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(fd, SHARED_BYTES)
        banks = {role: {} for role in EXPERT_SHAPES}
        for role, layer, shape, offset in _layout():
            size = EXPERT_BYTES[role]
            bank = host_bank.__new__(host_bank)
            bank._buf = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE, offset=offset)
            bank.tensor = torch.frombuffer(bank._buf, dtype=torch.bfloat16).view(shape)
            bank.addr = bank.tensor.data_ptr()
            bank.nbytes = size
            bank._pinned = bank._locked = False
            banks[role][layer] = bank
        # Existing loader mappings remain writable while packing. Other processes
        # can map the completed bank only for reading.
        fcntl.fcntl(fd, F_ADD_SEALS, SEAL_SHARED_BANK)
        _bank_fd = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
        return banks
    finally:
        os.close(fd)


def allocator_env(root, cuda_major, python_env, *, disk_backup=None, preload_library=None):
    tools = root / "studio-resident-tools"
    library = preload_library or tools / f"torch_memory_saver_hook_mode_preload_cu{cuda_major}.abi3.so"
    runtimes = list(python_env.glob(f"lib/python*/site-packages/nvidia/*/lib/libcudart.so.{cuda_major}"))
    if not library.is_file() or len(runtimes) != 1:
        raise RuntimeError("The resident Studio runner needs the pinned Torch Memory Saver CUDA wheel.")
    libraries = os.pathsep.join(filter(None, (os.getenv("LD_LIBRARY_PATH"), str(runtimes[0].parent))))
    env = dict(
        LD_PRELOAD=str(library), LD_LIBRARY_PATH=libraries,
        TMS_INIT_ENABLE="1", TMS_INIT_ENABLE_CPU_BACKUP="1", TMS_RETAIN_CPU_BACKUP="1",
        TMS_INIT_ENABLE_DISK_BACKUP="0",
        PYTORCH_ALLOC_CONF="expandable_segments:False", NCCL_GRAPH_REGISTER="0",
    )
    if disk_backup:
        env.update(TMS_INIT_ENABLE_CPU_BACKUP="0", TMS_INIT_ENABLE_DISK_BACKUP="1",
                   TMS_DISK_BACKUP_DIR=str(disk_backup))
    return env


def resettable_expert_region():
    from torch_memory_saver import torch_memory_saver

    return torch_memory_saver.region("studio-expert-cache", enable_cpu_backup=False)


class Residency:
    def __init__(self, *, resettable_cache=None):
        self.parked = False
        self.released_bytes = 0
        self.backed_segments = set()
        self.resettable_cache = resettable_cache
        self.disk_backup = os.getenv("TMS_DISK_BACKUP_DIR") if os.getenv("TMS_INIT_ENABLE_DISK_BACKUP") == "1" else None
        self.pause_pageable = None
        self.can_use_pageable = False
        self.backup_sizes = {}
        if resettable_cache is not None and os.getenv("LIGHTREASON_STUDIO_PAGEABLE_HANDOFF") == "1":
            self.pause_pageable = ctypes.CDLL(os.environ["LD_PRELOAD"]).tms_pause_pageable
            self.pause_pageable.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.POINTER(ctypes.c_size_t)]
            self.pause_pageable.restype = ctypes.c_int

    def transition(self, command):
        import torch
        from torch_memory_saver import torch_memory_saver

        if command not in ("park", "activate"):
            raise ValueError("Invalid CUDA residency command.")
        started = perf_counter()
        before, _ = torch.cuda.mem_get_info()
        if command == "park" and not self.parked:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            segments = {(s["address"], s["total_size"]) for s in torch.cuda.memory_snapshot()}
            if self.resettable_cache is not None:
                cache_ptrs = [bank.data_ptr() for bank in self.resettable_cache.bank_caches.values()]
                segments = {(address, size) for address, size in segments
                            if not any(address <= ptr < address + size for ptr in cache_ptrs)}
            new_backup = sum(size for _, size in segments - self.backed_segments)
            if self.disk_backup:
                if disk_usage(self.disk_backup).free < new_backup + GIB:
                    raise MemoryError("Insufficient disk space to park this runner’s CUDA state.")
                new_backup = 0
            available = available_memory()
            if available < new_backup + 8 * GIB:
                raise MemoryError(f"Parking needs {new_backup / GIB:.2f} GiB of new backup memory and "
                                  f"an 8 GiB reserve. Host RAM available is {available / GIB:.2f} GiB.")
            before, _ = torch.cuda.mem_get_info()
            if self.can_use_pageable:
                ram_bytes, disk_bytes = ctypes.c_size_t(), ctypes.c_size_t()
                # Keep the same reserve used by the ordinary host backup.
                budget = available - 8 * GIB
                if self.pause_pageable(budget, ctypes.byref(ram_bytes), ctypes.byref(disk_bytes)):
                    raise RuntimeError("Lossless CUDA parking failed.")
                self.backup_sizes = dict(pageable_bytes=ram_bytes.value, disk_bytes=disk_bytes.value)
            else:
                torch_memory_saver.pause()
            free, _ = torch.cuda.mem_get_info()
            # The configured allocator keeps these host backups across resume.
            if not self.disk_backup and os.getenv("TMS_RETAIN_CPU_BACKUP") == "1":
                self.backed_segments = segments
            else:
                self.backed_segments.clear()
            self.released_bytes = free - before
            self.parked = True
            if free - before < GIB:
                raise RuntimeError("The CUDA allocator did not release this runner’s VRAM.")
        elif command == "activate" and self.parked:
            if before < self.released_bytes:
                raise MemoryError("Insufficient VRAM to resume this runner.")
            torch_memory_saver.resume()
            if self.resettable_cache is not None:
                # Every live request starts with a reset cache. The slot storage
                # can be reused after its mappings are restored.
                self.resettable_cache.reset()
            torch.cuda.synchronize()
            self.parked = False
            # The initial backup stays on disk while Scheduled audits the
            # checkpoint. Later handoffs may use RAM within the byte budget.
            self.can_use_pageable = self.pause_pageable is not None
        free, _ = torch.cuda.mem_get_info()
        return dict(parked=self.parked, free_bytes=free, seconds=perf_counter() - started, **self.backup_sizes)


class CheckpointResidency:
    """Park Scheduled's immutable parameters on their audited checkpoint views."""

    def __init__(self, model, gpu_weight_views):
        parameters = {name for name, weight in model.named_parameters() if weight.is_cuda}
        if not parameters or parameters != gpu_weight_views.keys():
            raise RuntimeError("Scheduled's checkpoint views do not cover its resident weights.")
        self.model = model
        self.weights = []
        for name, view in gpu_weight_views.items():
            module, _, field = name.rpartition(".")
            self.weights.append((model.get_submodule(module), field, view))
        self.buffers = []
        self.parked = False
        self.released_bytes = 0

    def transition(self, command):
        import torch

        if command not in ("park", "activate"):
            raise ValueError("Invalid CUDA residency command.")
        started = perf_counter()
        before, _ = torch.cuda.mem_get_info()
        if command == "park" and not self.parked:
            if available_memory() < 8 * GIB:
                raise MemoryError("Parking Scheduled needs an 8 GiB host-memory reserve.")
            torch.cuda.synchronize()
            # Scheduled owns no CUDA graphs or KV cache between requests.
            # Moving buffers preserves their current bytes, including RoPE state.
            for module in self.model.modules():
                for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
                    if buffer.is_cuda:
                        self.buffers.append((module, name, buffer.device))
                        setattr(module, name, buffer.cpu())
            for module, name, view in self.weights:
                setattr(module, name, torch.nn.Parameter(view, requires_grad=False))
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            self.released_bytes = free - before
            if self.released_bytes < GIB:
                raise RuntimeError("Scheduled did not release its resident CUDA weights.")
            self.parked = True
        elif command == "activate" and self.parked:
            if before < self.released_bytes:
                raise MemoryError("Insufficient VRAM to resume Scheduled.")
            for module, name, view in self.weights:
                setattr(module, name, torch.nn.Parameter(view.to("cuda:0"), requires_grad=False))
            for module, name, device in self.buffers:
                setattr(module, name, getattr(module, name).to(device))
            self.buffers.clear()
            torch.cuda.synchronize()
            self.parked = False
        free, _ = torch.cuda.mem_get_info()
        return dict(parked=self.parked, free_bytes=free, seconds=perf_counter() - started)


def attach_scheduler(scheduler, path):
    """Handle private commands on the scheduler thread only at its idle barrier."""
    import zmq

    if scheduler.config.tp_info.size != 1 or _bank_fd is None:
        raise RuntimeError("Resident Studio requires a single FreeToken rank with shared BF16 banks.")
    queue = scheduler._recv_from_tokenizer
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    listener.bind(path)
    listener.listen(1)
    poller = zmq.Poller()
    poller.register(listener.fileno(), zmq.POLLIN)
    poller.register(queue.socket, zmq.POLLIN)
    original = scheduler.receive_msg
    cache = scheduler.engine.moe_offload_cache if os.getenv("LIGHTREASON_STUDIO_CACHE_PARK") == "1" else None
    residency = Residency(resettable_cache=cache)

    def receive(blocking=False):
        if not blocking:
            return original(blocking=False)
        scheduler.run_when_idle()
        while True:
            events = dict(poller.poll())
            if listener.fileno() in events:
                with listener.accept()[0] as connection:
                    connection.settimeout(2)
                    command = connection.recv(32)
                    if command == b"experts":
                        connection.sendmsg([b"ok"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", [_bank_fd]))])
                    elif command in (b"park", b"activate"):
                        was_parked = residency.parked
                        try:
                            reply = residency.transition(command.decode())
                        except (MemoryError, RuntimeError) as exc:
                            logging.exception("FreeToken GPU handoff failed")
                            connection.sendall(json.dumps({"error": str(exc)}).encode())
                            continue
                        if reply["parked"] and not was_parked:
                            poller.unregister(queue.socket)
                        elif was_parked and not reply["parked"]:
                            poller.register(queue.socket, zmq.POLLIN)
                        connection.sendall(json.dumps(reply).encode())
                    else:
                        raise ValueError("Invalid FreeToken handoff command.")
            if not residency.parked and queue.socket in events:
                return original(blocking=False)

    scheduler.receive_msg = receive


def _connect(path, command):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        connection.settimeout(45)
        connection.connect(str(path))
        connection.sendall(command.encode())
        return connection
    except BaseException:
        connection.close()
        raise


def switch_freetoken(path, command):
    with _connect(path, command) as connection:
        data = connection.recv(1024)
    if not data:
        raise RuntimeError(f"FreeToken exited during GPU handoff ({command}).")
    reply = json.loads(data)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply


def attach_checkpoint_experts(model, path):
    """Use the shared CPU views before Packed Replay’s full checkpoint byte audit."""
    import torch

    with _connect(path, "experts") as connection:
        fds = array("i")
        message, ancillary, flags, _ = connection.recvmsg(32, socket.CMSG_SPACE(fds.itemsize),
                                                         socket.MSG_CMSG_CLOEXEC)
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds.frombytes(data)
    try:
        if message != b"ok" or flags & socket.MSG_CTRUNC or len(fds) != 1:
            raise RuntimeError("FreeToken did not provide its shared BF16 bank.")
        if os.fstat(fds[0]).st_size != SHARED_BYTES or len(model.model.layers) != LAYERS:
            raise RuntimeError("The shared BF16 bank does not match the reference model.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            for role, index, shape, offset in _layout():
                experts = model.model.layers[index].mlp.experts
                name = role + "_proj"
                parameter = getattr(experts, name)
                if tuple(parameter.shape) != shape or parameter.dtype != torch.bfloat16:
                    raise RuntimeError(f"Unexpected BF16 tensor geometry in layer {index} {name}.")
                view = mmap.mmap(fds[0], EXPERT_BYTES[role], access=mmap.ACCESS_READ, offset=offset)
                tensor = torch.frombuffer(view, dtype=torch.bfloat16).view(shape)
                setattr(experts, name, torch.nn.Parameter(tensor, requires_grad=False))
    finally:
        for fd in fds:
            os.close(fd)
