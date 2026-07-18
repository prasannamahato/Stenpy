# memory_manager.py
from __future__ import annotations
import collections
import gc
import math
import os
import pickle
import struct as _struct
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

try:
    import torch.distributed as dist
except ImportError:
    dist = None

if TYPE_CHECKING:
    import zict 

try:
    import zict as _zict_mod
except ImportError:
    _zict_mod = None

try:
    from mpi4py import MPI as _MPI
    _HAS_MPI = True
except ImportError:
    _MPI = None
    _HAS_MPI = False

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False

_DTYPE = torch.float64
_DEBUG = os.environ.get("OPS_DEBUG", "0") == "1"
_VRAM_HEADROOM_FRAC = float(os.environ.get("OPS_VRAM_HEADROOM", "0.15"))

_call_counter = __import__("itertools").count()


def _uid(prefix: str = "t") -> str:
    return f"{prefix}_{next(_call_counter)}"


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[memory_manager] {msg}")


def _assert_fp64(t: torch.Tensor, where: str) -> None:
    if t.dtype not in (_DTYPE, torch.complex128):
        raise TypeError(f"{where}: expected float64 or complex128, got {t.dtype}")


def _vram_headroom_ok(device: torch.device, needed_bytes: int = 0) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    free, total = torch.cuda.mem_get_info(device)
    headroom = total * _VRAM_HEADROOM_FRAC
    return free - needed_bytes > headroom


def _vram_free_bytes(device: torch.device) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 2 ** 62
    props = torch.cuda.get_device_properties(device)
    return props.total_memory - torch.cuda.memory_allocated(device)



_MPI_ACTIVE = False
_MPI_COMM = None
_MPI_RANK = 0
_MPI_WORLD = 1


def _detect_mpi() -> None:
    global _MPI_ACTIVE, _MPI_COMM, _MPI_RANK, _MPI_WORLD
    if _HAS_MPI:
        try:
            _MPI_COMM = _MPI.COMM_WORLD
            _MPI_RANK = _MPI_COMM.Get_rank()
            _MPI_WORLD = _MPI_COMM.Get_size()
            if _MPI_WORLD > 1:
                _MPI_ACTIVE = True
                _dbg(f"MPI detected: rank={_MPI_RANK}, world={_MPI_WORLD}")
        except Exception:
            _MPI_ACTIVE = False


_detect_mpi()


def _mpi_halo_exchange_1d_optimized(shard: torch.Tensor, radius: int) -> torch.Tensor:
    global _MPI_ACTIVE, _MPI_COMM, _MPI_RANK, _MPI_WORLD

    if not _MPI_ACTIVE or radius <= 0 or _MPI_WORLD <= 1:
        return shard

    try_cuda_aware = shard.is_cuda and _MPI.Get_library_version().find("CUDA") != -1

    if not try_cuda_aware and shard.is_cuda:
        device = shard.device
        shard = shard.cpu()

    n_local = shard.shape[0]
    left_rank = _MPI_RANK - 1 if _MPI_RANK > 0 else None
    right_rank = _MPI_RANK + 1 if _MPI_RANK < _MPI_WORLD - 1 else None

    if n_local < 2 * radius:
        return shard

    send_left = shard[radius:2 * radius].clone()
    send_right = shard[-2 * radius:-radius].clone()
    recv_left = torch.zeros_like(send_left)
    recv_right = torch.zeros_like(send_right)

    reqs = []

    if left_rank is not None:
        reqs.append(_MPI_COMM.Irecv(recv_left.numpy(), source=left_rank, tag=20))
        reqs.append(_MPI_COMM.Isend(send_left.numpy().copy(), dest=left_rank, tag=10))

    if right_rank is not None:
        reqs.append(_MPI_COMM.Irecv(recv_right.numpy(), source=right_rank, tag=10))
        reqs.append(_MPI_COMM.Isend(send_right.numpy().copy(), dest=right_rank, tag=20))

    if reqs:
        _MPI.Request.Waitall(reqs)
    if left_rank is not None:
        shard[:radius] = recv_left
    if right_rank is not None:
        shard[-radius:] = recv_right

    if not try_cuda_aware and shard.is_cuda:
        shard = shard.to(device)

    return shard


def is_distributed() -> bool:
    if _MPI_ACTIVE and _MPI_WORLD > 1:
        return True
    if dist is not None and dist.is_available() and dist.is_initialized():
        return True
    return False


class BufferState:

    __slots__ = (
        "key", "shape", "dtype", "device", "size_bytes",
        "last_access", "access_count", "remaining_consumers",
    )

    def __init__(self, key: str, shape: Tuple[int, ...], dtype: torch.dtype,
                 device: torch.device) -> None:
        self.key = key
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.size_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        self.last_access = time.monotonic()
        self.access_count = 1
        self.remaining_consumers = -1


def _tensor_weight(_key: str, value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    return len(value) if hasattr(value, "__len__") else 8


_DTYPE_CODES: Dict[torch.dtype, int] = {
    torch.float64: 0, torch.float32: 1, torch.complex128: 2,
    torch.complex64: 3, torch.int64: 4, torch.int32: 5, torch.bool: 6,
}
_CODE_DTYPES: Dict[int, torch.dtype] = {v: k for k, v in _DTYPE_CODES.items()}


def _dump_to_bytes(tensor: torch.Tensor) -> bytes:
    arr = tensor.detach().cpu().contiguous().numpy()
    code = _DTYPE_CODES.get(tensor.dtype)
    if code is None:
        raise TypeError(f"disk spill: unsupported dtype {tensor.dtype}")
    header = _struct.pack("<BB", code, arr.ndim) + _struct.pack(f"<{arr.ndim}q", *arr.shape)
    return header + arr.tobytes()


def _make_bytes_loader(state_getter: Callable[[], "BufferState"]):
    def loader(blob: bytes) -> torch.Tensor:
        state = state_getter()
        cpu_tensor = pickle.loads(blob) if blob[:1] == b"\x80" else _load_from_bytes(blob)
        return cpu_tensor.to(device=state.device, dtype=state.dtype)
    return loader


def _load_from_bytes(buf: bytes) -> torch.Tensor:
    code, ndim = _struct.unpack_from("<BB", buf, 0)
    offset = 2
    shape = _struct.unpack_from(f"<{ndim}q", buf, offset)
    offset += 8 * ndim
    dtype = _CODE_DTYPES[code]
    np_dtype = torch.empty((), dtype=dtype).numpy().dtype
    import numpy as np
    arr = np.frombuffer(buf, dtype=np_dtype, offset=offset).reshape(shape)
    return torch.from_numpy(arr.copy())


class MemoryManager:

    def __init__(
        self,
        gpu_fraction: float = float(os.environ.get("OPS_MM_GPU_FRACTION", "0.8")),
        ram_fraction: float = float(os.environ.get("OPS_MM_RAM_FRACTION", "0.75")),
        spill_dir: str = os.environ.get("OPS_MM_SPILL_DIR", "./spill"),
        admission_margin: float = float(os.environ.get("OPS_MM_ADMISSION_SAFETY_MARGIN", "0.90")),
        pool_depth: int = int(os.environ.get("OPS_POOL_DEPTH", "4")),
        pool_max_mb: float = float(os.environ.get("OPS_POOL_MAX_MB", "256")),
    ) -> None:
        if _zict_mod is None:
            raise ImportError(
                "MemoryManager requires the 'zict' package (pip install zict)."
            )
        self.gpu_fraction = gpu_fraction
        self.ram_fraction = ram_fraction
        self.admission_margin = admission_margin
        self._pool_depth = pool_depth
        self._pool_max_bytes = pool_max_mb * 1024 ** 2

        self._lock: threading.RLock = threading.RLock()
        self._meta: Dict[str, BufferState] = {}
        self._consumer_counts: Dict[str, int] = {}
        self._step: int = 0
        self._disk_raw = _zict_mod.File(spill_dir)
        self._disk = _zict_mod.Func(
            _dump_to_bytes,
            _make_bytes_loader(self._current_load_state),
            self._disk_raw,
        )
        self._host = _zict_mod.Buffer(
            fast={}, slow=self._disk,
            n=self._ram_budget_bytes(),
            weight=_tensor_weight,
        )
        self._gpu: Dict[torch.device, "zict.Buffer"] = {}
        self._pool: Dict[Tuple, List[torch.Tensor]] = collections.defaultdict(list)

        self._load_key_stack: List[str] = [] 

    def _current_load_state(self) -> "BufferState":
        key = self._load_key_stack[-1]
        return self._meta[key]

    def _ram_budget_bytes(self) -> int:
        if _HAS_PSUTIL:
            return int(_psutil.virtual_memory().total * self.ram_fraction)
        return int(4 * 1024 ** 3)

    def _gpu_buffer(self, device: torch.device) -> "zict.Buffer":
        if device not in self._gpu:
            total = torch.cuda.get_device_properties(device).total_memory
            budget = int(total * self.gpu_fraction)
            gpu_to_host = _zict_mod.Func(
                lambda t: t.detach().cpu(),
                lambda t: t.to(device, non_blocking=True),
                self._host,
            )
            self._gpu[device] = _zict_mod.Buffer(
                fast={}, slow=gpu_to_host, n=budget, weight=_tensor_weight,
            )
        return self._gpu[device]

    def _tier_for(self, device: torch.device) -> "zict.Buffer":
        return self._gpu_buffer(device) if device.type == "cuda" else self._host

    def _available_bytes(self, device: torch.device) -> int:
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                free, _total = torch.cuda.mem_get_info(device)
                return int(free)
            except Exception:
                total = torch.cuda.get_device_properties(device).total_memory
                return int(total - torch.cuda.memory_allocated(device))
        if _HAS_PSUTIL:
            return int(_psutil.virtual_memory().available)
        return 2 * 1024 ** 3

    def _admission_check(self, needed_bytes: int, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        avail = self._available_bytes(device)
        if needed_bytes > avail * self.admission_margin:
            raise MemoryError(
                f"Refusing to allocate {needed_bytes / 1e9:.3f} GB on {device}: "
                f"only {avail / 1e9:.3f} GB available "
                f"({self.admission_margin:.0%} safety margin). Refused up front "
                f"instead of risking an uncontrolled OOM."
            )

    @staticmethod
    def _bucket(numel: int) -> int:
        if numel <= 0:
            return 1
        return 1 << (numel - 1).bit_length()

    def get_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = _DTYPE,
        device: torch.device = torch.device("cpu"),
        key: Optional[str] = None,
        layout: torch.memory_format = torch.contiguous_format,
    ) -> torch.Tensor:
        shape = tuple(shape)
        key = key or _uid("buf")
        needed_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()

        with self._lock:
            state = self._meta.get(key)
            if state is not None:
                if state.shape != shape or state.dtype != dtype or state.device != device:
                    raise KeyError(
                        f"get_buffer: key '{key}' requested with shape={shape}, "
                        f"dtype={dtype}, device={device}, but a live buffer already "
                        f"exists under this key with shape={state.shape}, "
                        f"dtype={state.dtype}, device={state.device}. Derive keys "
                        f"from shape instead of reusing one across shapes."
                    )
                tier = self._tier_for(device)
                self._load_key_stack.append(key)
                try:
                    tensor = tier[key]
                finally:
                    self._load_key_stack.pop()
                state.last_access = time.monotonic()
                state.access_count += 1
                return tensor
        pool_key = (shape, str(device), dtype, layout)
        with self._lock:
            pool = self._pool.get(pool_key)
            if pool:
                tensor = pool.pop().zero_()
            else:
                tensor = None

        if tensor is None:
            self._admission_check(needed_bytes, device)
            tensor = torch.zeros(shape, dtype=dtype, device=device).to(memory_format=layout)

        with self._lock:
            self._meta[key] = BufferState(key, shape, dtype, device)
            self._meta[key].remaining_consumers = self._consumer_counts.get(key, -1)
            tier = self._tier_for(device)
            tier[key] = tensor
        tensor._mm_key = key
        return tensor

    def allocate(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        key: Optional[str] = None,
        layout: torch.memory_format = torch.contiguous_format,
        dtype: torch.dtype = _DTYPE,
    ) -> torch.Tensor:
        if device.type == "cuda":
            needed = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if not _vram_headroom_ok(device, needed):
                raise MemoryError(f"Insufficient VRAM headroom for shape {shape} on {device}")
        return self.get_buffer(shape=shape, dtype=dtype, device=device, key=key, layout=layout)

    def adopt(self, tensor: torch.Tensor, key: Optional[str] = None) -> torch.Tensor:
        if tensor is None:
            return tensor
        k = key or _uid("adopt")
        with self._lock:
            self._meta[k] = BufferState(k, tuple(tensor.shape), tensor.dtype, tensor.device)
            self._meta[k].remaining_consumers = self._consumer_counts.get(k, -1)
            self._tier_for(tensor.device)[k] = tensor
        tensor._mm_key = k
        return tensor

    def release(self, tensor: torch.Tensor, key: Optional[str] = None) -> None:
        if tensor is None:
            return
        key = getattr(tensor, "_mm_key", key)
        if key is None:
            return
        with self._lock:
            state = self._meta.pop(key, None)
            if state is None:
                return
            tier = self._tier_for(state.device)
            try:
                del tier[key]
            except KeyError:
                pass
            nbytes = tensor.numel() * tensor.element_size()
            if (
                tensor.is_contiguous()
                and not tensor.requires_grad
                and nbytes <= self._pool_max_bytes
            ):
                pool_key = (state.shape, str(state.device), state.dtype, torch.contiguous_format)
                pool = self._pool[pool_key]
                if len(pool) < self._pool_depth:
                    pool.append(tensor.detach())

    def spill_to_disk(self, key: str) -> None:
        with self._lock:
            state = self._meta.get(key)
            if state is None:
                return
            tier = self._tier_for(state.device)
            try:
                value = tier[key]
            except KeyError:
                return
            del tier[key]
            self._disk[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value

    def should_manage(self, shape: Tuple[int, ...]) -> bool:
        return math.prod(shape) >= 1024

    def owns(self, tensor: torch.Tensor) -> bool:
        key = getattr(tensor, "_mm_key", None)
        if key is None:
            return False
        with self._lock:
            return key in self._meta

    def make_field(
        self,
        tensor: torch.Tensor,
        spacing: Tuple[float, ...],
        origin: Optional[Tuple[float, ...]] = None,
        key: Optional[str] = None,
    ) -> Any:
        from stenpy_engine import Field  

        _assert_fp64(tensor, "MemoryManager.make_field")
        k = key or _uid("field")
        with self._lock:
            if k not in self._meta:
                self._meta[k] = BufferState(k, tuple(tensor.shape), tensor.dtype, tensor.device)
                self._tier_for(tensor.device)[k] = tensor
                tensor._mm_key = k
        return Field(tensor=tensor, spacing=spacing, origin=origin, mm=self, key=k)

    def allocate_field(
        self,
        shape: Tuple[int, ...],
        spacing: Tuple[float, ...],
        origin: Optional[Tuple[float, ...]] = None,
        device: torch.device = torch.device("cpu"),
        key: Optional[str] = None,
        layout: torch.memory_format = torch.contiguous_format,
    ) -> Any:
        from stenpy_engine import Field  

        k = key or _uid("field")
        t = self.allocate(shape, device, key=k, layout=layout)
        return Field(tensor=t, spacing=spacing, origin=origin, mm=self, key=k)

    def clear_pool(self, full: bool = False) -> None:
        with self._lock:
            self._pool.clear()
            if full:
                for key in list(self._meta.keys()):
                    state = self._meta.pop(key)
                    tier = self._tier_for(state.device)
                    try:
                        del tier[key]
                    except KeyError:
                        pass

    def halo_exchange(
        self, shard: torch.Tensor, radius: int, dims: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if radius <= 0:
            return shard
        if _MPI_ACTIVE and _MPI_WORLD > 1:
            if dims is None or 0 in dims:
                return _mpi_halo_exchange_1d_optimized(shard, radius)
            return shard
        return shard

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        if _MPI_ACTIVE and _MPI_WORLD > 1:
            was_cuda = t.is_cuda
            if was_cuda:
                device = t.device
                t = t.cpu()
            result_np = t.numpy().copy()
            _MPI_COMM.Allreduce(_MPI.IN_PLACE, result_np, op=_MPI.SUM)
            t_out = torch.from_numpy(result_np).to(_DTYPE)
            return t_out.to(device) if was_cuda else t_out
        if dist is not None and dist.is_available() and dist.is_initialized():
            out = t.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out
        return t

    def memory_pressure(self, device: Optional[torch.device] = None) -> float:
        if not torch.cuda.is_available():
            return 0.0
        dev = device or torch.device("cuda", 0)
        try:
            return torch.cuda.memory_allocated(dev) / torch.cuda.get_device_properties(dev).total_memory
        except Exception:
            return 0.0

    def cpu_memory_pressure(self) -> float:
        if not _HAS_PSUTIL:
            return 0.0
        return _psutil.virtual_memory().percent / 100.0

    def set_consumer_counts(self, counts: Dict[str, int]) -> None:
        with self._lock:
            self._consumer_counts.update(counts)
            for key, n in counts.items():
                state = self._meta.get(key)
                if state is not None:
                    state.remaining_consumers = n

    def decrement_consumer(self, key: str) -> None:
        with self._lock:
            n = self._consumer_counts.get(key)
            if n is not None and n > 0:
                self._consumer_counts[key] = n - 1
            state = self._meta.get(key)
            if state is not None and state.remaining_consumers > 0:
                state.remaining_consumers -= 1

    def advance_step(self) -> None:
        with self._lock:
            self._step += 1

    @property
    def step(self) -> int:
        return self._step

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            gpu_live = {str(dev): len(buf.fast) for dev, buf in self._gpu.items()}
            return {
                "live_keys": len(self._meta),
                "host_fast": len(self._host.fast),
                "host_slow": len(self._host.slow),
                "gpu_fast": gpu_live,
                "pooled": sum(len(v) for v in self._pool.values()),
                "step": self._step,
            }

    def __repr__(self) -> str:
        s = self.stats()
        return f"MemoryManager(live={s['live_keys']}, host_fast={s['host_fast']}, host_spilled={s['host_slow']}, step={s['step']})"


def use_advanced_mm() -> None:
    _dbg("MemoryManager is the zict-backed implementation")



_PRESETS: Dict[str, Dict[str, int]] = {
    "laptop":      {"total_ram_bytes": 8 * 1024 ** 3, "total_vram_bytes": 0},
    "workstation": {"total_ram_bytes": 64 * 1024 ** 3, "total_vram_bytes": 24 * 1024 ** 3},
    "hpc":         {"total_ram_bytes": 512 * 1024 ** 3, "total_vram_bytes": 80 * 1024 ** 3},
}


@dataclass
class HardwareProfile:
    name: str = "auto"
    total_ram_bytes: int = 0
    total_vram_bytes: int = 0

    @classmethod
    def detect(cls) -> "HardwareProfile":
        ram = 8 * 1024 ** 3
        if _HAS_PSUTIL:
            try:
                ram = int(_psutil.virtual_memory().total)
            except Exception:
                pass
        vram = 0
        try:
            if torch.cuda.is_available():
                _free, total = torch.cuda.mem_get_info()
                vram = int(total)
        except Exception:
            pass
        return cls(name="auto", total_ram_bytes=ram, total_vram_bytes=vram)

    @classmethod
    def preset(cls, name: str) -> "HardwareProfile":
        if name not in _PRESETS:
            raise ValueError(f"Unknown hardware profile '{name}'. Known: {list(_PRESETS)}")
        return cls(name=name, **_PRESETS[name])

    def scaled(self, ram_frac: float = 0.0, vram_frac: float = 0.0,
               floor_bytes: int = 0, ceiling_bytes: Optional[int] = None) -> int:
        candidate = max(int(self.total_ram_bytes * ram_frac),
                         int(self.total_vram_bytes * vram_frac))
        candidate = max(candidate, floor_bytes)
        if ceiling_bytes is not None:
            candidate = min(candidate, ceiling_bytes)
        return candidate


class AdaptiveThresholds:

    def __init__(self, profile: Optional[HardwareProfile] = None,
                 override: Optional[Dict[str, int]] = None) -> None:
        self.profile = profile or HardwareProfile.detect()
        self._override = dict(override or {})

    def _get(self, key: str, compute: Callable[[], int]) -> int:
        if key in self._override:
            return self._override[key]
        env_key = f"OPS_THRESHOLD_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                return int(env_val)
            except ValueError:
                pass
        return compute()

    def row_stream_threshold_bytes(self) -> int:
        return self._get(
            "row_stream",
            lambda: self.profile.scaled(ram_frac=0.001, vram_frac=0.002,
                                         floor_bytes=8 * 1024 * 1024),
        )

    def eager_load_ceiling_bytes(self) -> int:
        return self._get(
            "eager_load_ceiling",
            lambda: self.profile.scaled(ram_frac=0.25, vram_frac=0.5,
                                         floor_bytes=64 * 1024 * 1024),
        )

    def small_output_threshold_bytes(self) -> int:
        return self._get(
            "small_output",
            lambda: self.profile.scaled(ram_frac=0.01, vram_frac=0.02,
                                         floor_bytes=32 * 1024 * 1024),
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.name,
            "total_ram_gb": round(self.profile.total_ram_bytes / 1024 ** 3, 2),
            "total_vram_gb": round(self.profile.total_vram_bytes / 1024 ** 3, 2),
            "row_stream_threshold_mb": round(self.row_stream_threshold_bytes() / 1024 ** 2, 2),
            "eager_load_ceiling_mb": round(self.eager_load_ceiling_bytes() / 1024 ** 2, 2),
            "small_output_threshold_mb": round(self.small_output_threshold_bytes() / 1024 ** 2, 2),
        }

    def __repr__(self) -> str:
        return f"AdaptiveThresholds(profile={self.profile.name!r})"


_DEFAULT_THRESHOLDS: Optional[AdaptiveThresholds] = None


def get_default_thresholds() -> AdaptiveThresholds:
    global _DEFAULT_THRESHOLDS
    if _DEFAULT_THRESHOLDS is None:
        env_profile = os.environ.get("OPS_HARDWARE_PROFILE")
        profile = HardwareProfile.preset(env_profile) if env_profile else HardwareProfile.detect()
        _DEFAULT_THRESHOLDS = AdaptiveThresholds(profile=profile)
    return _DEFAULT_THRESHOLDS


def set_hardware_profile(name_or_profile: Union[str, HardwareProfile]) -> AdaptiveThresholds:
    global _DEFAULT_THRESHOLDS
    profile = (HardwareProfile.preset(name_or_profile)
               if isinstance(name_or_profile, str) else name_or_profile)
    _DEFAULT_THRESHOLDS = AdaptiveThresholds(profile=profile)
    return _DEFAULT_THRESHOLDS


__all__ = [
    "MemoryManager", "BufferState", "HardwareProfile", "AdaptiveThresholds",
    "get_default_thresholds", "set_hardware_profile", "use_advanced_mm",
    "is_distributed",
    "_uid", "_assert_fp64", "_vram_headroom_ok", "_vram_free_bytes",
    "_tensor_weight", "_dump_to_bytes", "_load_from_bytes",
]
