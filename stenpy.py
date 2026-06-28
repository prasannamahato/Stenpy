# stenpy.py

from __future__ import annotations

import collections
import contextlib
import inspect
import itertools
import math
import os
import threading
import warnings
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
import datetime
from functools import lru_cache
from collections import OrderedDict

try:
    from mpi4py import MPI as _MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_FUSION_ENABLED = os.environ.get("OPS_FUSION_ENABLED", "1").lower() in ("1", "true", "yes")
try:
    import fused_compiler as fc
except ImportError:
    fc = None
    _FUSION_ENABLED = False

_DTYPE = torch.float64

_DEBUG             = os.environ.get("OPS_DEBUG", "0") == "1"
_VALIDATION_SAMPLE = float(os.environ.get("OPS_VALIDATION_SAMPLE", "0.01"))
_VALIDATION_LEVEL  = os.environ.get("OPS_VALIDATION", "basic")
_VRAM_HEADROOM_FRAC = float(os.environ.get("OPS_VRAM_HEADROOM", "0.15"))
_MAX_LAZY_FIELD_HANDLES = int(os.environ.get("OPS_MAX_LAZY_FIELD_HANDLES", "8")) 

_call_counter = itertools.count()


def _uid(prefix: str = "t") -> str:
    return f"{prefix}_{next(_call_counter)}"


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[stenpy] {msg}")


def _assert_fp64(t: torch.Tensor, where: str) -> None:
    if t.dtype not in (_DTYPE, torch.complex128):
        raise TypeError(f"{where}: expected float64 or complex128, got {t.dtype}")


def _vram_headroom_ok(device: torch.device, needed_bytes: int = 0) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    props = torch.cuda.get_device_properties(device)
    free  = props.total_memory - torch.cuda.memory_allocated(device)
    headroom = props.total_memory * _VRAM_HEADROOM_FRAC
    return free - needed_bytes > headroom


def _vram_free_bytes(device: torch.device) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 2 ** 62
    props = torch.cuda.get_device_properties(device)
    return props.total_memory - torch.cuda.memory_allocated(device)


# ---------------------------------------------------------------------------
# Pinned staging buffer pool
# ---------------------------------------------------------------------------

class _PinnedStagingPool:
    def __init__(self, max_buffers: int = 4) -> None:
        self._lock    = threading.Lock()
        self._free:   List[torch.Tensor] = []
        self._max     = max_buffers
        self._enabled = torch.cuda.is_available()

    def get(self, nbytes: int) -> Optional[torch.Tensor]:
        if not self._enabled:
            return None
        with self._lock:
            for i, buf in enumerate(self._free):
                if buf.numel() >= nbytes:
                    self._free.pop(i)
                    return buf[:nbytes]
        rounded = 1 << (nbytes - 1).bit_length() if nbytes > 0 else 1
        try:
            return torch.empty(rounded, dtype=torch.uint8, pin_memory=True)[:nbytes]
        except Exception:
            return None

    def put(self, buf: torch.Tensor) -> None:
        if not self._enabled or buf is None:
            return
        flat = buf.reshape(-1) if not buf.is_contiguous() else buf
        with self._lock:
            if len(self._free) < self._max:
                self._free.append(flat)

    def clear(self) -> None:
        with self._lock:
            self._free.clear()


_PINNED_POOL = _PinnedStagingPool(max_buffers=4)


def _lazy_to_gpu(
    arr:    np.ndarray,
    device: torch.device,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    nbytes = arr.nbytes
    pinned = _PINNED_POOL.get(nbytes)
    if pinned is not None:
        np.copyto(pinned.numpy(), arr.ravel().view(np.uint8))
        src = pinned.view(torch.float64).reshape(arr.shape)
        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with ctx:
            gpu_t = src.to(device, non_blocking=True)
        return gpu_t, pinned
    t = torch.from_numpy(arr.astype(np.float64, copy=False)).to(_DTYPE)
    return t.to(device), None


# ---------------------------------------------------------------------------
# MPI detection and globals
# ---------------------------------------------------------------------------

_MPI_ACTIVE = False
_MPI_COMM = None
_MPI_RANK = 0
_MPI_WORLD = 1


def _detect_mpi():
    """Detect if we're running under MPI and set globals"""
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
    """
    Optimized idempotent halo exchange with overlapping communication.
    Uses non-blocking sends/receives for maximum throughput.
    """
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

    send_left = shard[radius:2*radius].clone()
    send_right = shard[-2*radius:-radius].clone()
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


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

class Field:
    __slots__ = ("tensor", "spacing", "origin", "_mm", "_key")

    def __init__(
        self,
        tensor:  torch.Tensor,
        spacing: Tuple[float, ...],
        origin:  Optional[Tuple[float, ...]] = None,
        mm:      Optional["MemoryManager"]   = None,
        key:     Optional[str]               = None,
    ) -> None:
        if tensor.dtype != _DTYPE:
            raise TypeError(f"Field requires float64 tensor, got {tensor.dtype}")
        self.tensor  = tensor
        self.spacing = tuple(float(s) for s in spacing)
        self.origin  = tuple(origin) if origin is not None else tuple(0.0 for _ in spacing)
        self._mm     = mm
        self._key    = key

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return _DTYPE

    @property
    def ndim(self) -> int:
        return len(self.spacing)

    def release(self) -> None:
        if self._mm is not None and self.tensor is not None and self._key is not None:
            self._mm.release(self.tensor, self._key)
        self.tensor = None
        self._key   = None

    def __enter__(self) -> "Field":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __repr__(self) -> str:
        sh = tuple(self.tensor.shape) if self.tensor is not None else None
        return f"Field(shape={sh}, spacing={self.spacing}, fp64)"


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    _DEFAULT_POOL_DEPTH = int(os.environ.get("OPS_POOL_DEPTH", "4"))

    def __init__(self, pool_depth: int = _DEFAULT_POOL_DEPTH) -> None:
        self._pool:       Dict[Tuple, List[torch.Tensor]] = collections.defaultdict(list)
        self._lock        = threading.Lock()
        self._pool_depth  = pool_depth
        self._live:       Dict[str, torch.Tensor] = {}
        self._pool_ptrs:  Dict[Tuple, set]        = {}

    @staticmethod
    def _pool_key(
        shape:  Tuple[int, ...],
        device: torch.device,
        layout: torch.memory_format = torch.contiguous_format,
    ) -> Tuple:
        return (shape, str(device), layout)

    def _tag(self, t: torch.Tensor, key: str) -> None:
        t._mm_key = key

    def allocate(
        self,
        shape:  Tuple[int, ...],
        device: torch.device,
        key:    Optional[str]               = None,
        layout: torch.memory_format         = torch.contiguous_format,
    ) -> torch.Tensor:
        shape = tuple(shape)
        if device.type == "cuda":
            needed = math.prod(shape) * 8
            if not _vram_headroom_ok(device, needed):
                raise MemoryError(
                    f"Insufficient VRAM headroom for shape {shape} on {device}"
                )
        pk = self._pool_key(shape, device, layout)
        with self._lock:
            pool = self._pool[pk]
            ptrs = self._pool_ptrs.setdefault(pk, set())
            if pool:
                t = pool.pop()
                ptrs.discard(t.data_ptr())
                t.zero_()
            else:
                t = torch.zeros(shape, dtype=_DTYPE, device=device).to(
                    memory_format=layout
                )
            if key:
                self._live[key] = t
                self._tag(t, key)
        return t

    def release(self, tensor: torch.Tensor, key: Optional[str] = None) -> None:
        if tensor is None:
            return
        with self._lock:
            if key is not None:
                self._live.pop(key, None)
            if not hasattr(tensor, "_mm_key"):
                return
            if not tensor.is_contiguous():
                return
            if tensor.requires_grad or tensor.grad_fn is not None:
                return
            tensor = tensor.detach()
            pk   = self._pool_key(tuple(tensor.shape), tensor.device)
            pool = self._pool[pk]
            ptrs = self._pool_ptrs.setdefault(pk, set())
            ptr  = tensor.data_ptr()
            if ptr in ptrs:
                return
            pool.append(tensor)
            ptrs.add(ptr)
            while len(pool) > self._pool_depth * 2:
                old = pool.pop(0)
                ptrs.discard(old.data_ptr())

    def should_manage(self, shape: Tuple[int, ...]) -> bool:
        return math.prod(shape) >= 1024

    def advance_step(self) -> None:
        """No-op for basic MemoryManager; advanced MM overrides this."""
        pass

    def make_field(
        self,
        tensor:  torch.Tensor,
        spacing: Tuple[float, ...],
        origin:  Optional[Tuple[float, ...]] = None,
        key:     Optional[str]               = None,
    ) -> Field:
        _assert_fp64(tensor, "MemoryManager.make_field")
        k = key or _uid("field")
        with self._lock:
            self._live[k] = tensor
            self._tag(tensor, k)
        return Field(tensor=tensor, spacing=spacing, origin=origin, mm=self, key=k)

    def allocate_field(
        self,
        shape:   Tuple[int, ...],
        spacing: Tuple[float, ...],
        origin:  Optional[Tuple[float, ...]] = None,
        device:  torch.device                = torch.device("cpu"),
        key:     Optional[str]               = None,
        layout:  torch.memory_format         = torch.contiguous_format,
    ) -> Field:
        k = key or _uid("field")
        t = self.allocate(shape, device, key=k, layout=layout)
        return Field(tensor=t, spacing=spacing, origin=origin, mm=self, key=k)

    def clear_pool(self) -> None:
        with self._lock:
            self._pool.clear()
            self._pool_ptrs.clear()

    def halo_exchange(
        self,
        shard:  torch.Tensor,
        radius: int,
        dims:   Optional[List[int]] = None,
    ) -> torch.Tensor:
        global _MPI_ACTIVE, _MPI_WORLD

        if radius <= 0:
            return shard

        if _MPI_ACTIVE and _MPI_WORLD > 1:
            if dims is None or 0 in dims:
                return _mpi_halo_exchange_1d_optimized(shard, radius)
            return shard

        return shard

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        global _MPI_ACTIVE, _MPI_COMM, _MPI_WORLD
        
        if _MPI_ACTIVE and _MPI_WORLD > 1:
            was_cuda = t.is_cuda
            if was_cuda:
                device = t.device
                t = t.cpu()
            result_np = t.numpy().copy()
            _MPI_COMM.Allreduce(_MPI.IN_PLACE, result_np, op=_MPI.SUM)
            t_out = torch.from_numpy(result_np).to(_DTYPE)
            if was_cuda:
                t_out = t_out.to(device)
            return t_out
        if is_distributed():
            out = t.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out
        return t

    def memory_pressure(self, device: Optional[torch.device] = None) -> float:
        if not torch.cuda.is_available():
            return 0.0
        dev = device or torch.device("cuda", 0)
        try:
            return (
                torch.cuda.memory_allocated(dev)
                / torch.cuda.get_device_properties(dev).total_memory
            )
        except Exception:
            return 0.0

    def owns(self, tensor: torch.Tensor) -> bool:
        key = getattr(tensor, "_mm_key", None)
        if key is None:
            return False
        with self._lock:
            return key in self._live

    def __repr__(self) -> str:
        n_live = len(self._live)
        n_pool = sum(len(v) for v in self._pool.values())
        return f"MemoryManager(live={n_live}, pooled={n_pool})"


def use_advanced_mm() -> None:
    global MemoryManager
    try:
        from memory_manager import MemoryManager as _AdvMM
        MemoryManager = _AdvMM
        _dbg("Switched to advanced MemoryManager from memory_manager.py")
    except ImportError as exc:
        warnings.warn(
            f"memory_manager.py not found; staying with built-in pool ({exc})"
        )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphNode:
    id:        str
    op_name:   str
    input_ids: Tuple[str, ...]  = dc_field(default_factory=tuple)
    params:    Dict[str, Any]   = dc_field(default_factory=dict)


class Graph:
    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._order: List[str]            = []
        self._dtypes: Dict[str, torch.dtype] = {} 
        self._shapes: Dict[str, Tuple[int, ...]] = {}

    def add(
        self,
        op_name:   str,
        input_ids: Tuple[str, ...]          = (),
        params:    Optional[Dict[str, Any]] = None,
        node_id:   Optional[str]            = None,
    ) -> str:
        nid  = node_id or _uid(op_name)
        node = GraphNode(
            id        = nid,
            op_name   = op_name,
            input_ids = tuple(input_ids),
            params    = dict(params or {}),
        )
        self._nodes[nid] = node
        self._order.append(nid)
        return nid

    def topological_sort(self) -> List[GraphNode]:
        in_deg:   Dict[str, int]       = {nid: 0  for nid in self._nodes}
        children: Dict[str, List[str]] = {nid: [] for nid in self._nodes}
        for nid, node in self._nodes.items():
            for dep in node.input_ids:
                if dep not in self._nodes:
                    raise ValueError(
                        f"Node '{nid}' references unknown input '{dep}'"
                    )
                in_deg[nid] += 1
                children[dep].append(nid)
        queue: collections.deque = collections.deque(
            nid for nid, d in in_deg.items() if d == 0
        )
        order: List[GraphNode] = []
        while queue:
            nid = queue.popleft()
            order.append(self._nodes[nid])
            for child in children[nid]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        if len(order) != len(self._nodes):
            raise ValueError("Graph contains a cycle")
        return order

    def clone_with_replacement(
        self, source_id: str, tensor: torch.Tensor
    ) -> "Graph":
        g = Graph()
        for nid in self._order:
            node = self._nodes[nid]
            if nid == source_id:
                g.add("_constant", (), {"value": tensor}, node_id=nid)
            else:
                g.add(node.op_name, node.input_ids, node.params, node_id=nid)
        return g

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"Graph({len(self._nodes)} nodes)"

    def set_node_shape(self, node_id: str, shape: Tuple[int, ...],
                       dtype: torch.dtype = torch.float64) -> None:
        self._shapes[node_id] = tuple(shape)
        self._dtypes[node_id] = dtype

    def get_node_shape(self, node_id: str) -> Optional[Tuple[int, ...]]:
        return self._shapes.get(node_id)

    def get_node_dtype(self, node_id: str) -> torch.dtype:
        return self._dtypes.get(node_id, torch.float64)

    def build_consumer_map(self) -> Dict[str, List[str]]:
        consumers = {nid: [] for nid in self._nodes}
        for nid, node in self._nodes.items():
            for dep in node.input_ids:
                consumers[dep].append(nid)
        return consumers

# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _validate_halo_contract(
    tensor:   torch.Tensor,
    radius:   int,
    dims:     Optional[List[int]],
    op_name:  str,
) -> None:
    if radius < 0:
        raise ValueError(f"{op_name}: invalid halo radius {radius}")
    if tensor.ndim == 0:
        return
    for d in (dims if dims is not None else range(tensor.ndim)):
        if tensor.shape[d] < 2 * radius + 1:
            raise ValueError(
                f"{op_name}: dim {d} size {tensor.shape[d]} "
                f"too small for halo radius {radius}"
            )


def _op_accepts_alloc(fn: Callable) -> bool:
    try:
        return "alloc" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False


_FUSIBLE_OPS = {"add", "sub", "mul", "div", "neg", "exp", "log", "sqrt", "sin", "tanh"}

def _try_fused_dispatch(
    expr_str: str,
    inputs:   Dict[str, torch.Tensor],
    out_buf:  torch.Tensor,
) -> bool:
    return False


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class Runtime:
    def __init__(
        self,
        mm:        MemoryManager,
        device:    Union[str, torch.device] = "cpu",
        skip_pool: bool                     = False,
        fusion_enabled: bool                = None,
    ) -> None:
        self.mm         = mm
        _set_dist_mm(mm)
        self.device     = torch.device(device)
        self._alloc_cache: Dict[str, bool] = {}
        self.skip_pool  = skip_pool
        if fusion_enabled is None:
            fusion_enabled = _FUSION_ENABLED
        if fusion_enabled and fc is None:
            warnings.warn("fusion_compiler not found; fusion disabled", ImportWarning)
            fusion_enabled = False
        self.fusion_enabled = fusion_enabled
        self._h2d_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )

    def _accepts_alloc(self, op_name: str, fn: Callable) -> bool:
        if op_name not in self._alloc_cache:
            self._alloc_cache[op_name] = _op_accepts_alloc(fn)
        return self._alloc_cache[op_name]

    def _ref_counts(self, graph: Graph) -> Dict[str, int]:
        counts: Dict[str, int] = {nid: 0 for nid in graph._nodes}
        for node in graph._nodes.values():
            for dep in node.input_ids:
                counts[dep] += 1
        return counts

    def flush_vram(self) -> None:
        live = getattr(self.mm, "_live", None)
        lock = (
            getattr(self.mm, "_lock", None)
            or getattr(self.mm, "_live_lock", None)
        )

        if live is not None:
            ctx = lock if lock is not None else contextlib.nullcontext()
            with ctx:
                dead_keys = []
                for k, state in list(live.items()):
                    tensor = (
                        state
                        if isinstance(state, torch.Tensor)
                        else getattr(state, "tensor", None)
                    )
                    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                        dead_keys.append(k)
                for k in dead_keys:
                    state = live.get(k)
                    if isinstance(state, torch.Tensor):
                        live.pop(k, None)
                    elif state is not None:
                        state.tensor     = None
                        state.is_free    = True
                        state.is_evicted = True
        pw_lock = getattr(self.mm, "_pending_writes_lock", None)
        pending = getattr(self.mm, "_pending_writes", None)
        cond    = getattr(self.mm, "_pending_writes_cond", None)

        if pw_lock is not None and cond is not None:
            with pw_lock:
                if pending is not None:
                    pending.clear()
                cond.notify_all()
        else:
            if pending is not None:
                pending.clear()
        pool_ptrs = getattr(self.mm, "_pool_ptrs", None)
        if pool_ptrs is not None:
            pool_ptrs.clear()

        _device_pools = getattr(self.mm, "_device_pools", None)
        _pool_lock    = getattr(self.mm, "_pool_lock", None)
        if _device_pools is not None:
            if _pool_lock is not None:
                with _pool_lock:
                    _device_pools.clear()
            else:
                _device_pools.clear()

        _PINNED_POOL.clear()
        if self.device.type == "cuda" and torch.cuda.is_available():
            _clear_k_grid_cache()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        _dbg("flush_vram: done")

    def _auto_chunk_size(
        self,
        lazy:        "LazyField",
        chunk_dim:   int,
        safety_factor: float = 8.0,
    ) -> int:
        shape          = lazy.shape
        slice_elements = math.prod(s for i, s in enumerate(shape) if i != chunk_dim)
        bytes_per_slice = slice_elements * 8
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            free   = _vram_free_bytes(self.device)
            usable = int(free * (1.0 - _VRAM_HEADROOM_FRAC * 2))
        else:
            try:
                usable = int(_psutil.virtual_memory().available * 0.30)
            except Exception:
                usable = 1 * 1024 ** 3
        chunk  = max(1, int(usable / (safety_factor * bytes_per_slice)))
        result = min(chunk, shape[chunk_dim])
        _dbg(
            f"_auto_chunk_size: free={usable/1024**3:.2f}GB "
            f"bytes/slice={bytes_per_slice/1024**2:.1f}MB chunk={result}"
        )
        return result


    def run_chunked(
            self,
            graph:      "Graph",
            sink_id:    str,
            chunk_dim:  int                = 0,
            chunk_size: Optional[int]      = None,
            out_path:   Optional[str]      = None,
            spacing:    Optional[Tuple]    = None,
            origin:     Optional[Tuple]    = None,
        ) -> Dict[str, Any]:
            import queue as _queue
            lazy_node_ids: List[str]           = []
            lazy_field:    Optional[LazyField] = None
            for nid in graph._order:
                node = graph._nodes[nid]
                if node.op_name == "_constant":
                    val = node.params.get("value")
                    if isinstance(val, LazyField):
                        lazy_node_ids.append(nid)
                        if lazy_field is None:
                            lazy_field = val

            if lazy_field is None:
                return self.run(graph)
            
            total = lazy_field.shape[chunk_dim]
            if chunk_size is None:
                chunk_size = self._auto_chunk_size(lazy_field, chunk_dim)

            idx0: List[Any] = [slice(None)] * len(lazy_field.shape)
            idx0[chunk_dim] = slice(0, min(chunk_size, total))
            arr0 = lazy_field[tuple(idx0)]
            t0_  = torch.from_numpy(arr0.astype(np.float64)).to(_DTYPE).to(self.device)
            g0   = graph
            for nid in lazy_node_ids:
                g0 = g0.clone_with_replacement(nid, t0_)
            with torch.no_grad():
                r0 = self.run(g0)
            c0 = r0[sink_id].cpu()

            out_shape      = list(c0.shape)
            out_shape[chunk_dim] = total
            out_shape      = tuple(out_shape)

            s_min = float(c0.min())
            s_max = float(c0.max())
            s_sum = float(c0.sum())
            s_n   = c0.numel()

            del t0_, g0, r0, arr0
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            h5f = h5_dset = None
            write_dim = chunk_dim
            if out_path is not None:
                chunk_h5 = tuple(min(128, s) for s in c0.shape)
                h5f      = h5py.File(out_path, "w")
                h5_dset  = h5f.create_dataset(
                    "field", shape=out_shape, dtype=np.float64, chunks=chunk_h5,
                )
                sp = spacing or tuple(1.0 for _ in range(len(out_shape)))
                og = origin  or tuple(0.0 for _ in range(len(out_shape)))
                h5f.create_dataset("spacing", data=np.array(sp, dtype=np.float64))
                h5f.create_dataset("origin",  data=np.array(og, dtype=np.float64))

            WRITE_Q_SIZE = 2
            write_q:   _queue.Queue      = _queue.Queue(maxsize=WRITE_Q_SIZE)
            write_errors: List[Exception] = []

            def _writer_thread() -> None:
                while True:
                    item = write_q.get()
                    if item is None:
                        write_q.task_done()
                        break
                    w_start, w_end, arr_cpu = item
                    try:
                        if h5_dset is not None:
                            sl            = [slice(None)] * len(out_shape)
                            sl[write_dim] = slice(w_start, w_end)
                            h5_dset[tuple(sl)] = arr_cpu
                    except Exception as e:
                        write_errors.append(e)
                    finally:
                        del arr_cpu
                        write_q.task_done()

            writer = threading.Thread(
                target=_writer_thread, daemon=True, name="chunk-writer"
            )
            writer.start()

            PREFETCH_SIZE = 1
            prefetch_q: _queue.Queue = _queue.Queue(maxsize=PREFETCH_SIZE)

            chunk_starts = [s for s in range(chunk_size, total, chunk_size) if s < total]

            def _reader_thread(starts: List[int]) -> None:
                for st in starts:
                    en = min(st + chunk_size, total)
                    if en <= st:
                        continue
                    idxr            = [slice(None)] * len(lazy_field.shape)
                    idxr[chunk_dim] = slice(st, en)
                    arr = lazy_field[tuple(idxr)]
                    prefetch_q.put((st, en, arr))
                prefetch_q.put(None)

            reader = threading.Thread(
                target=_reader_thread, args=(chunk_starts,),
                daemon=True, name="chunk-reader",
            )
            reader.start()

            write_q.put((0, min(chunk_size, total), c0.numpy()))
            del c0

            pinned_bufs: List = []
            pinned_bufs_lock  = threading.Lock()

            while True:
                item = prefetch_q.get()
                if item is None:
                    break
                p_start, p_end, arr = item

                if self.device.type == "cuda" and self._h2d_stream is not None:
                    gpu_t, pinned = _lazy_to_gpu(arr, self.device, self._h2d_stream)
                    torch.cuda.current_stream(self.device).wait_stream(self._h2d_stream)
                    if pinned is not None:
                        with pinned_bufs_lock:
                            pinned_bufs.append(pinned)
                else:
                    gpu_t  = torch.from_numpy(arr.astype(np.float64, copy=False)).to(_DTYPE).to(self.device)
                    pinned = None

                del arr

                cg = graph
                for nid in lazy_node_ids:
                    cg = cg.clone_with_replacement(nid, gpu_t)

                with torch.no_grad():
                    cr = self.run(cg)
                cout = cr[sink_id]

                s_min = min(s_min, float(cout.min()))
                s_max = max(s_max, float(cout.max()))
                s_sum += float(cout.sum())
                s_n   += cout.numel()

                cout_np = cout.cpu().numpy()
                if self.mm.owns(cout):
                    self.mm.release(cout)
                del gpu_t, cg, cr, cout

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                with pinned_bufs_lock:
                    for pb in pinned_bufs:
                        _PINNED_POOL.put(pb)
                    pinned_bufs.clear()

                write_q.put((p_start, p_end, cout_np))
                if write_errors:
                    break

            write_q.put(None)
            write_q.join()
            reader.join(timeout=5.0)
            if h5f is not None:
                h5f.close()
            if write_errors:
                raise RuntimeError(f"HDF5 write error: {write_errors[0]}")

            return {
                sink_id: {
                    "shape_out": out_shape,
                    "min":       s_min,
                    "max":       s_max,
                    "mean":      s_sum / max(s_n, 1),
                    "out_path":  out_path,
                }
            }

    def run(self, graph, seed=None):
        if self.fusion_enabled and self._should_fuse(graph):
            try:
                return self._run_fused(graph, seed)
            except Exception as e:
                warnings.warn(f"Fusion failed, falling back to original: {e}")
        return self._run_original(graph, seed)

    def _should_fuse(self, graph: Graph) -> bool:
        if self.device.type != "cuda":
            return False
        if fc is None:
            return False
        for node in graph._nodes.values():
            if node.op_name in fc.FUSIBLE_OPS:
                return True
        return False

    def _run_fused(self, graph: Graph, seed: Optional[Dict] = None) -> Dict:
        for nid, node in graph._nodes.items():
            if nid not in graph._shapes:
                if nid in (seed or {}):
                    val = seed[nid]
                    if isinstance(val, torch.Tensor):
                        graph.set_node_shape(nid, val.shape, val.dtype)
                elif len(node.input_ids) == 0:
                    val = node.params.get("value")
                    if isinstance(val, torch.Tensor):
                        graph.set_node_shape(nid, val.shape, val.dtype)
                    else:
                        graph.set_node_shape(nid, (1,))
                else:
                    pass
        return fc.compile_and_execute(graph, self, seed)

    def _run_original(
        self,
        graph: Graph,
        seed:  Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results:   Dict[str, Any]   = dict(seed or {})
        ref_count: Dict[str, int]   = self._ref_counts(graph)

        for node in graph.topological_sort():
            if node.id in results:
                continue

            inputs = []
            for i in node.input_ids:
                if i in results:
                    val = results[i]
                else:
                    input_node = graph._nodes.get(i)
                    if input_node and input_node.op_name == "_constant":
                        val = input_node.params.get("value")
                    else:
                        raise RuntimeError(
                            f"Input '{i}' for node '{node.id}' not found in results"
                        )
                if isinstance(val, LazyField):
                    arr = val[:]
                    val = (
                        torch.from_numpy(arr.astype(np.float64, copy=False))
                        .to(_DTYPE).to(self.device)
                    )
                inputs.append(val)

            kwargs = dict(node.params)
            for k, v in list(kwargs.items()):
                if isinstance(v, LazyField):
                    arr       = v[:]
                    kwargs[k] = (
                        torch.from_numpy(arr.astype(np.float64, copy=False))
                        .to(_DTYPE).to(self.device)
                    )

            tensor_inputs = [i for i in inputs if isinstance(i, torch.Tensor)]
            if tensor_inputs:
                cuda_devs  = [t.device for t in tensor_inputs if t.device.type == "cuda"]
                target_dev = cuda_devs[0] if cuda_devs else tensor_inputs[0].device
                inputs     = [
                    i.to(target_dev) if isinstance(i, torch.Tensor) else i
                    for i in inputs
                ]
                for k, v in list(kwargs.items()):
                    if isinstance(v, torch.Tensor) and v.device != target_dev:
                        kwargs[k] = v.to(target_dev)

            meta   = OP_METADATA.get(node.op_name, {})
            radius = meta.get("stencil_radius", 0)
            if radius > 0:
                first_t     = next((i for i in inputs if isinstance(i, torch.Tensor)), None)
                static_dims = meta.get("exchange_dims")
                if first_t is not None:
                    dims = (
                        [d for d in static_dims if d < first_t.ndim]
                        if static_dims is not None
                        else list(range(first_t.ndim))
                    )
                else:
                    dims = static_dims

                should_exchange = is_distributed() or (_MPI_ACTIVE and _MPI_WORLD > 1)
                if should_exchange:
                    for inp in inputs:
                        if isinstance(inp, torch.Tensor):
                            _validate_halo_contract(inp, radius, dims, node.op_name)

                inputs = [
                    self.mm.halo_exchange(i, radius, dims)
                    if isinstance(i, torch.Tensor) else i
                    for i in inputs
                ]

            output = None
            if True:
                op_fn = OP_REGISTRY[node.op_name]
                sig   = inspect.signature(op_fn)
                params = sig.parameters
                if "alloc" in params:
                    kwargs["alloc"] = self.mm.allocate
                if "mm" in params:
                    kwargs["mm"] = self.mm
                output = op_fn(*inputs, **kwargs)

            def _wrap(o: Any) -> Any:
                if isinstance(o, torch.Tensor):
                    return o
                if isinstance(o, (list, tuple)):
                    return type(o)(_wrap(x) for x in o)
                if isinstance(o, dict):
                    return {k: _wrap(v) for k, v in o.items()}
                return o

            output = _wrap(output)

            if isinstance(output, torch.Tensor):
                if not output.is_complex():
                    _assert_fp64(output, node.op_name)
                is_managed = self.mm.owns(output)
                if is_managed and output._base is not None:
                    is_managed = False
                if (
                    not self.skip_pool
                    and not is_managed
                    and not output.is_complex()  
                    and self.mm.should_manage(output.shape)
                    and output.is_contiguous()
                ):
                    if output.device.type == "cuda":
                        needed = output.numel() * 8
                        if _vram_headroom_ok(output.device, needed):
                            key = _uid(node.id + "_out")
                            buf = self.mm.allocate(output.shape, output.device, key=key)
                            buf.copy_(output)
                            del output
                            output = buf
                    else:
                        key = _uid(node.id + "_out")
                        buf = self.mm.allocate(output.shape, output.device, key=key)
                        buf.copy_(output)
                        del output
                        output = buf

            results[node.id] = output

            for dep in node.input_ids:
                if dep in ref_count:
                    ref_count[dep] -= 1
                    if ref_count[dep] <= 0 and dep in results:
                        dep_node = graph._nodes.get(dep)
                        if dep_node is None or dep_node.op_name != "_constant":
                            val = results.pop(dep)
                            if isinstance(val, torch.Tensor) and self.mm.owns(val):
                                self.mm.release(val)

        return results

# ---------------------------------------------------------------------------
# Tile iterator
# ---------------------------------------------------------------------------

def _tile_iter(
    data:       torch.Tensor,
    spacing:    Tuple[float, ...],
    origin:     Tuple[float, ...],
    tile_shape: Tuple[int, ...],
    overlap:    int,
    ndim:       int,
) -> Iterator[Tuple[Tuple, torch.Tensor, Tuple]]:
    spatial = data.shape[:ndim]
    ranges  = [range(0, spatial[d], tile_shape[d]) for d in range(ndim)]
    for starts in itertools.product(*ranges):
        write_s, read_s, tile_origin = [], [], []
        for d, st in enumerate(starts):
            n    = spatial[d]
            w_e  = min(st + tile_shape[d], n)
            r_s  = max(0, st - overlap)
            r_e  = min(n, w_e + overlap)
            write_s.append(slice(st, w_e))
            read_s.append(slice(r_s, r_e))
            tile_origin.append(origin[d] + r_s * spacing[d])
        for _ in range(ndim, data.ndim):
            write_s.append(slice(None))
            read_s.append(slice(None))
        yield tuple(write_s), data[tuple(read_s)], tuple(tile_origin)


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------


OP_REGISTRY: Dict[str, Callable] = {}
OP_METADATA: Dict[str, dict]     = {}


def register_operator(
    name:          str,
    func:          Callable,
    radius:        int = 0,
    halo_l:        int = 0,
    halo_r:        int = 0,
    cost:          str = "low",
    exchange_dims: Optional[List[int]] = None,
    flops:         int = 0,              
    bytes_in:      int = 0,              
    bytes_out:     int = 0,              
    fp64_fusion:   bool = True,          
) -> None:
    OP_REGISTRY[name] = func
    OP_METADATA[name] = {
        "stencil_radius": radius,
        "halo_left":      halo_l if halo_l else radius,
        "halo_right":     halo_r if halo_r else radius,
        "cost":           cost,
        "exchange_dims":  exchange_dims,
        "flops_per_element": flops,
        "bytes_read_per_element": bytes_in,
        "bytes_written_per_element": bytes_out,
        "supports_fp64_fusion": fp64_fusion,
    }

# ---------------------------------------------------------------------------
# Operators — arithmetic
# ---------------------------------------------------------------------------

def _constant(*, value) -> torch.Tensor:
    if isinstance(value, (int, float)):
        value = torch.tensor(value, dtype=_DTYPE)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"_constant: expected torch.Tensor or number, got {type(value).__name__}")
    return value.to(_DTYPE)

register_operator("_constant", _constant, cost="low", flops=0, bytes_in=0, bytes_out=8)


def add(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "add(a)"); _assert_fp64(b, "add(b)")
    if alloc is None:
        return torch.add(a, b)
    out = alloc(torch.broadcast_shapes(a.shape, b.shape), a.device, key=_uid("add"))
    return torch.add(a, b, out=out)

register_operator("add", add, cost="low", flops=1, bytes_in=16, bytes_out=8)


def sub(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "sub(a)"); _assert_fp64(b, "sub(b)")
    if alloc is None:
        return torch.sub(a, b)
    out = alloc(torch.broadcast_shapes(a.shape, b.shape), a.device, key=_uid("sub"))
    return torch.sub(a, b, out=out)

register_operator("sub", sub, cost="low", flops=1, bytes_in=16, bytes_out=8)


def mul(a: torch.Tensor, b: torch.Tensor = None,
        scalar: float = None, alloc=None) -> torch.Tensor:
    if scalar is not None and b is None:
        b = torch.full_like(a, scalar, dtype=_DTYPE)
    elif b is None:
        raise ValueError("mul: must provide either a second tensor 'b' or 'scalar'")
    _assert_fp64(a, "mul(a)"); _assert_fp64(b, "mul(b)")
    if alloc is None:
        return torch.mul(a, b)
    out = alloc(torch.broadcast_shapes(a.shape, b.shape), a.device, key=_uid("mul"))
    return torch.mul(a, b, out=out)

register_operator("mul", mul, cost="low", flops=1, bytes_in=16, bytes_out=8)


def div(
    a: torch.Tensor, b: torch.Tensor,
    eps: float = 1e-15, alloc=None,
) -> torch.Tensor:
    _assert_fp64(a, "div(a)"); _assert_fp64(b, "div(b)")
    eps_t  = torch.as_tensor(eps, dtype=b.dtype, device=b.device)
    b_safe = torch.where(
        b.abs() < eps_t,
        torch.where(b < 0, -eps_t, eps_t),
        b,
    )
    if alloc is None:
        return torch.div(a, b_safe)
    out = alloc(torch.broadcast_shapes(a.shape, b.shape), a.device, key=_uid("div"))
    return torch.div(a, b_safe, out=out)

register_operator("div", div, cost="low", flops=1, bytes_in=16, bytes_out=8)


def neg(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "neg(a)")
    if alloc is None:
        return torch.neg(a)
    out = alloc(a.shape, a.device, key=_uid("neg"))
    return torch.neg(a, out=out)

register_operator("neg", neg, cost="low", flops=1, bytes_in=8, bytes_out=8)


def clamp(
    a: torch.Tensor,
    lo: float = 0.0,
    hi: float = 1.0,
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(a, "clamp(a)")
    if alloc is None:
        return torch.clamp(a, lo, hi)
    out = alloc(a.shape, a.device, key=_uid("clamp"))
    return torch.clamp(a, lo, hi, out=out)

register_operator("clamp", clamp, cost="low", flops=2, bytes_in=8, bytes_out=8)


# ---------------------------------------------------------------------------
# Operators — elementwise
# ---------------------------------------------------------------------------

def exp(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "exp(a)")
    if alloc is None:
        return torch.exp(a)
    out = alloc(a.shape, a.device, key=_uid("exp"))
    return torch.exp(a, out=out)

register_operator("exp", exp, cost="medium", flops=8, bytes_in=8, bytes_out=8)


def log(a: torch.Tensor, eps: float = 1e-15, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "log(a)")
    if alloc is None:
        return torch.log(a.clamp(min=eps))
    out = alloc(a.shape, a.device, key=_uid("log"))
    return torch.log(a.clamp(min=eps), out=out)

register_operator("log", log, cost="medium", flops=8, bytes_in=8, bytes_out=8)


def sqrt(a: torch.Tensor, eps: float = 0.0, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "sqrt(a)")
    if alloc is None:
        return torch.sqrt(a.clamp(min=eps))
    out = alloc(a.shape, a.device, key=_uid("sqrt"))
    return torch.sqrt(a.clamp(min=eps), out=out)

register_operator("sqrt", sqrt, cost="medium", flops=8, bytes_in=8, bytes_out=8)


def sin(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "sin(a)")
    if alloc is None:
        return torch.sin(a)
    out = alloc(a.shape, a.device, key=_uid("sin"))
    return torch.sin(a, out=out)

register_operator("sin", sin, cost="medium", flops=8, bytes_in=8, bytes_out=8)

def cos(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "cos(a)")
    if alloc is None:
        return torch.cos(a)
    out = alloc(a.shape, a.device, key=_uid("cos"))
    return torch.cos(a, out=out)

register_operator("cos", cos, cost="medium", flops=8, bytes_in=8, bytes_out=8)


def tanh(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "tanh(a)")
    if alloc is None:
        return torch.tanh(a)
    out = alloc(a.shape, a.device, key=_uid("tanh"))
    return torch.tanh(a, out=out)

register_operator("tanh", tanh, cost="medium", flops=8, bytes_in=8, bytes_out=8, fp64_fusion=True)


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------

def _pad1d(t: torch.Tensor, dim: int, r: int, bc: str) -> torch.Tensor:
    mode_map = {
        "neumann":   "replicate",
        "dirichlet": "constant",
        "periodic":  "circular",
        "reflect":   "reflect",
    }
    mode  = mode_map.get(bc, "constant")
    pad   = [0] * (2 * t.ndim)
    pad_idx        = 2 * (t.ndim - 1 - dim)
    pad[pad_idx]   = r
    pad[pad_idx+1] = r

    try:
        return F.pad(t, pad, mode=mode, value=0.0)
    except RuntimeError:
        pass

    new_shape      = list(t.shape)
    new_shape[dim] += 2 * r
    padded         = torch.zeros(new_shape, dtype=t.dtype, device=t.device)
    dst = [slice(None)] * t.ndim
    dst[dim] = slice(r, r + t.shape[dim])
    padded[tuple(dst)] = t

    if mode == "replicate":
        for i in range(r):
            sl      = [slice(None)] * t.ndim
            sl[dim] = i;          padded[tuple(sl)] = t[(slice(None),) * dim + (0,)]
            sl[dim] = -(i+1);     padded[tuple(sl)] = t[(slice(None),) * dim + (-1,)]
    elif mode == "circular":
        for i in range(r):
            sl      = [slice(None)] * t.ndim
            sl[dim] = i;          padded[tuple(sl)] = t[(slice(None),) * dim + (-(r-i),)]
            sl[dim] = -(i+1);     padded[tuple(sl)] = t[(slice(None),) * dim + (i,)]
    elif mode == "reflect" and t.shape[dim] > 1:
        for i in range(r):
            src_lo  = min(i+1, t.shape[dim]-1)
            src_hi  = max(t.shape[dim]-2-i, 0)
            sl      = [slice(None)] * t.ndim
            sl[dim] = r-1-i;      padded[tuple(sl)] = t[(slice(None),) * dim + (src_lo,)]
            sl[dim] = -(i+1);     padded[tuple(sl)] = t[(slice(None),) * dim + (src_hi,)]
    return padded

@triton.jit
def laplacian_3d_kernel(
    x_ptr, out_ptr,
    nx, ny, nz,
    dx2, dy2, dz2,
    boundary_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (nx * ny * nz)

    i = offsets // (ny * nz)
    j = (offsets // nz) % ny
    k = offsets % nz

    idx_c = i * (ny * nz) + j * nz + k
    c = tl.load(x_ptr + idx_c, mask=mask, other=0.0)

    if boundary_mode == 0:       
        im = (i - 1 + nx) % nx; ip = (i + 1) % nx
        jm = (j - 1 + ny) % ny; jp = (j + 1) % ny
        km = (k - 1 + nz) % nz; kp = (k + 1) % nz
    else:                         
        im = tl.where(i - 1 >= 0, i - 1, 0);   ip = tl.where(i + 1 < nx, i + 1, nx - 1)
        jm = tl.where(j - 1 >= 0, j - 1, 0);   jp = tl.where(j + 1 < ny, j + 1, ny - 1)
        km = tl.where(k - 1 >= 0, k - 1, 0);   kp = tl.where(k + 1 < nz, k + 1, nz - 1)

    idx_xm = im * (ny * nz) + j * nz + k
    idx_xp = ip * (ny * nz) + j * nz + k
    idx_ym = i * (ny * nz) + jm * nz + k
    idx_yp = i * (ny * nz) + jp * nz + k
    idx_zm = i * (ny * nz) + j * nz + km
    idx_zp = i * (ny * nz) + j * nz + kp

    fxm = tl.load(x_ptr + idx_xm, mask=mask, other=0.0)
    fxp = tl.load(x_ptr + idx_xp, mask=mask, other=0.0)
    fym = tl.load(x_ptr + idx_ym, mask=mask, other=0.0)
    fyp = tl.load(x_ptr + idx_yp, mask=mask, other=0.0)
    fzm = tl.load(x_ptr + idx_zm, mask=mask, other=0.0)
    fzp = tl.load(x_ptr + idx_zp, mask=mask, other=0.0)

    lap = ((fxp + fxm - 2.0*c) / dx2 +
           (fyp + fym - 2.0*c) / dy2 +
           (fzp + fzm - 2.0*c) / dz2)
    tl.store(out_ptr + idx_c, lap, mask=mask)


class Laplacian:
    def __init__(self, boundary: str = "periodic") -> None:
        self.boundary = boundary

    def _normalize_dx(self, dx, D, device, dtype):
        if isinstance(dx, (int, float)):
            return torch.full((D,), float(dx), device=device, dtype=dtype)
        dx = torch.tensor(dx, device=device, dtype=dtype)
        if dx.numel() != D:
            raise ValueError(f"dx must have length {D}, got {dx.numel()}")
        return dx

    def __call__(self, x: torch.Tensor, dx: Any = 1.0) -> torch.Tensor:
        D  = x.ndim
        dv = self._normalize_dx(dx, D, x.device, x.dtype)
        if x.is_cuda and D == 3:
            try:
                x_c = x.contiguous()
                out = torch.empty_like(x_c)
                total = math.prod(x_c.shape)
                BLOCK = 512
                grid = (triton.cdiv(total, BLOCK),)
                boundary_mode = 0 if self.boundary == "periodic" else 1
                laplacian_3d_kernel[grid](
                    x_c, out,
                    x_c.shape[0], x_c.shape[1], x_c.shape[2],
                    float(dv[0]**2), float(dv[1]**2), float(dv[2]**2),
                    boundary_mode,
                    BLOCK_SIZE=BLOCK,
                )
                return out
            except Exception as e:
                if _DEBUG:
                    print(f"Triton kernel failed, using GPU torch.roll fallback: {e}")

        out = torch.zeros_like(x)
        for d in range(D):
            if self.boundary == "periodic":
                xp = torch.roll(x, -1, d)
                xm = torch.roll(x,  1, d)
            else:
                padded = _pad1d(x, d, 1, self.boundary)
                N = x.shape[d]
                xp = padded.narrow(d, 2, N)
                xm = padded.narrow(d, 0, N)
            out.add_((xp + xm - 2.0 * x) / (dv[d] ** 2))
        return out


def laplacian(x: torch.Tensor, dx: Any = 1.0, boundary: str = "periodic") -> torch.Tensor:
    return Laplacian(boundary)(x, dx)

register_operator("laplacian", laplacian, radius=1, cost="high", exchange_dims=[0],
                  flops=8, bytes_in=56, bytes_out=8, fp64_fusion=True)



# ---------------------------------------------------------------------------
# Differential operators
# ---------------------------------------------------------------------------

def _grad2_along_dim(
    t:        torch.Tensor,
    dim:      int,
    dx:       float,
    boundary: str,
) -> torch.Tensor:
    _assert_fp64(t, "_grad2_along_dim")
    padded = _pad1d(t, dim, 1, boundary)
    N = t.shape[dim]
    left  = padded.narrow(dim, 0, N)
    right = padded.narrow(dim, 2, N)
    out = (right - left) * (0.5 / dx)
    return out.to(_DTYPE)

def gradient(
    t: torch.Tensor,
    dx: float = 1.0,
    boundary: str = "neumann",
    dim: Optional[int] = None,
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "gradient")
    use_distributed = _MPI_ACTIVE and _MPI_WORLD > 1 and boundary == "periodic"
    periodic = (boundary == "periodic")

    if dim is not None:
        if t.shape[dim] == 1:
            result = torch.zeros_like(t)
            if alloc:
                buf = alloc(result.shape, result.device, key=_uid("grad"))
                buf.copy_(result)
                return buf
            return result

        if use_distributed and dim == 0:
            left   = t[0:-2]
            right  = t[2:]
            grad = torch.zeros_like(t)
            grad[1:-1] = (right - left) / (2.0 * dx)
        elif periodic:
            xp = torch.roll(t, shifts=-1, dims=dim)
            xm = torch.roll(t, shifts=1, dims=dim)
            grad = (xp - xm) / (2.0 * dx)
        else:
            order = 2 if t.shape[dim] >= 3 else 1
            grad = torch.gradient(t, spacing=dx, dim=dim, edge_order=order)[0]

        if alloc:
            buf = alloc(grad.shape, grad.device, key=_uid("grad"))
            buf.copy_(grad)
            return buf
        return grad

    grads = []
    for d in range(t.ndim):
        if use_distributed and d == 0:
            left   = t[0:-2]
            right  = t[2:]
            g = torch.zeros_like(t)
            g[1:-1] = (right - left) / (2.0 * dx)
        elif periodic:
            xp = torch.roll(t, shifts=-1, dims=d)
            xm = torch.roll(t, shifts=1, dims=d)
            g = (xp - xm) / (2.0 * dx)
        else:
            order = 2 if t.shape[d] >= 3 else 1
            g = torch.gradient(t, spacing=dx, dim=d, edge_order=order)[0]
        grads.append(g)
    out = torch.stack(grads, dim=-1)
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("grad"))
        buf.copy_(out)
        return buf
    return out

register_operator("gradient", gradient, radius=1, cost="medium", exchange_dims=[0])


def divergence(
    t: torch.Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    boundary: str = "neumann",
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "divergence")
    D = t.shape[-1]
    if isinstance(dx, (int, float)):
        dx = (dx,) * D

    use_distributed = _MPI_ACTIVE and _MPI_WORLD > 1 and boundary == "periodic"
    periodic = (boundary == "periodic")

    out = torch.zeros_like(t[..., 0])
    for d in range(D):
        comp = t[..., d]
        if use_distributed and d == 0:
            left   = comp[0:-2]
            right  = comp[2:]
            grad = torch.zeros_like(comp)
            grad[1:-1] = (right - left) / (2.0 * dx[d])
        elif periodic:
            xp = torch.roll(comp, shifts=-1, dims=d)
            xm = torch.roll(comp, shifts=1, dims=d)
            grad = (xp - xm) / (2.0 * dx[d])
        else:
            grad = torch.gradient(comp, spacing=dx[d], dim=d, edge_order=2)[0]
        out += grad

    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("div"))
        buf.copy_(out)
        return buf
    return out

register_operator("divergence", divergence, radius=1, cost="medium", exchange_dims=[0])


def curl(
    t:        torch.Tensor,
    dx:       Union[float, Tuple[float, ...]] = 1.0,
    boundary: str = "neumann",
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "curl")
    if t.ndim < 2 or t.shape[-1] != 3:
        raise ValueError(f"curl: need (*spatial, 3), got {tuple(t.shape)}")
    if isinstance(dx, (int, float)):
        dx = (dx,) * 3

    use_distributed = _MPI_ACTIVE and _MPI_WORLD > 1 and boundary == "periodic"
    
    if use_distributed:
        g0 = gradient(t[..., 0], dx=dx, boundary=boundary, dim=None)
        g1 = gradient(t[..., 1], dx=dx, boundary=boundary, dim=None)
        g2 = gradient(t[..., 2], dx=dx, boundary=boundary, dim=None)
        
        curl_x = g2[..., 1] - g1[..., 2]
        curl_y = g0[..., 2] - g2[..., 0]
        curl_z = g1[..., 0] - g0[..., 1]
    else:
        def _grad_all(comp):
            return torch.gradient(comp, spacing=dx, edge_order=2)
        
        g0 = _grad_all(t[..., 0])
        g1 = _grad_all(t[..., 1])
        g2 = _grad_all(t[..., 2])
        
        curl_x = g2[1] - g1[2]
        curl_y = g0[2] - g2[0]
        curl_z = g1[0] - g0[1]
    
    out = torch.stack([curl_x, curl_y, curl_z], dim=-1)
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("curl"))
        buf.copy_(out)
        return buf
    return out
register_operator("curl", curl, radius=1, cost="high", exchange_dims=None)


def gradient_nd(
    t: torch.Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    boundary: str = "neumann",
    dim: Optional[int] = None, 
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "gradient_nd")

    original_shape = t.shape
    t_squeezed = t
    squeeze_dims = []
    for i, s in enumerate(t.shape):
        if s == 1:
            squeeze_dims.append(i)
    if squeeze_dims and t.squeeze().ndim > 0:
        t_squeezed = t.squeeze()
    
    if t_squeezed.ndim == 0:
        result = torch.zeros(original_shape + (0,), device=t.device, dtype=_DTYPE)
        if alloc:
            buf = alloc(result.shape, result.device, key=_uid("gradient_nd"))
            buf.copy_(result)
            return buf
        return result
    
    D = t_squeezed.ndim
    min_size = min(t_squeezed.shape)
    order = 2 if min_size >= 3 else 1
    
    try:
        grads = torch.gradient(t_squeezed, spacing=dx, edge_order=order)
    except RuntimeError:
        grads = torch.gradient(t_squeezed, spacing=dx, edge_order=1)
    
    out = torch.stack(grads, dim=-1)
    if out.shape[:-1] != original_shape:
        new_shape = list(original_shape) + [out.shape[-1]]
        out = out.reshape(new_shape)
    
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("gradient_nd"))
        buf.copy_(out)
        return buf
    return out

register_operator("gradient_nd", gradient_nd, radius=1, cost="high", exchange_dims=None)


def hessian(
    t:        torch.Tensor,
    dx:       Union[float, Tuple[float, ...]] = 1.0,
    boundary: str = "neumann",
    alloc     = None,
) -> torch.Tensor:
    _assert_fp64(t, "hessian")
    D = t.ndim
    if isinstance(dx, (int, float)):
        dx = (dx,) * D
    grad = gradient_nd(t, dx=dx, boundary=boundary) 
    if alloc:
        out = alloc((*t.shape, D, D), t.device, key=_uid("hessian"))
    else:
        out = torch.empty((*t.shape, D, D), dtype=_DTYPE, device=t.device)
    for i in range(D):
        hess_i = gradient_nd(grad[..., i], dx=dx, boundary=boundary)
        out[..., i, :] = hess_i
    return out

register_operator("hessian", hessian, radius=2, cost="very_high", exchange_dims=None)


def mean_curvature(
    t:        torch.Tensor,
    dx:       Union[float, Tuple[float, ...]] = 1.0,
    boundary: str   = "neumann",
    eps:      float = 1e-15,
    alloc     = None,
) -> torch.Tensor:
    _assert_fp64(t, "mean_curvature")
    D = t.ndim
    if isinstance(dx, (int, float)):
        dx = (dx,) * D
    grad = gradient_nd(t, dx=dx, boundary=boundary)     
    mag = torch.linalg.norm(grad, dim=-1, keepdim=True).clamp(min=eps)
    nhat = grad / mag
    out = divergence(nhat, dx=dx, boundary=boundary, alloc=alloc)
    return out

register_operator("mean_curvature", mean_curvature, radius=2, cost="very_high", exchange_dims=None)

def surface_normals(
    t:        torch.Tensor,
    dx:       Union[float, Tuple[float, ...]] = 1.0,
    boundary: str = "neumann",
    alloc     = None,
) -> torch.Tensor:
    _assert_fp64(t, "surface_normals")
    if t.ndim != 2:
        raise ValueError(f"surface_normals: expected 2D field, got {t.ndim}D")
    if isinstance(dx, (int, float)):
        dx = (dx, dx)
    grads = torch.gradient(t, spacing=dx, edge_order=2)  
    gx, gy = grads[0], grads[1]
    n = torch.stack([-gx, -gy, torch.ones_like(gx)], dim=-1)
    mag = torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-15)
    out = n / mag
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("surface_normals"))
        buf.copy_(out)
        return buf
    return out

register_operator("surface_normals", surface_normals, radius=1, cost="high", exchange_dims=None)

def material_derivative(
    t:        torch.Tensor,
    velocity: torch.Tensor,
    dx:       Union[float, Tuple[float, ...]] = 1.0,
    boundary: str = "neumann",
    alloc     = None,
) -> torch.Tensor:
    _assert_fp64(t, "material_derivative(f)")
    _assert_fp64(velocity, "material_derivative(v)")
    D = t.ndim
    if isinstance(dx, (int, float)):
        dx = (dx,) * D
    grad = gradient_nd(t, dx=dx, boundary=boundary)  
    if alloc:
        out = alloc(t.shape, t.device, key=_uid("material_derivative"))
        out.zero_()
    else:
        out = torch.zeros_like(t)
    for d in range(D):
        out.add_(velocity[..., d] * grad[..., d])
    return out

register_operator("material_derivative", material_derivative, radius=1, cost="high", exchange_dims=None)

def stack_components(*tensors: torch.Tensor, dim: int = -1) -> torch.Tensor:
    for i, t in enumerate(tensors):
        _assert_fp64(t, f"stack_components[{i}]")
    return torch.stack(list(tensors), dim=dim).to(_DTYPE)

register_operator("stack_components", stack_components, radius=0, cost="low")


def select_component(t: torch.Tensor, index: int) -> torch.Tensor:
    _assert_fp64(t, "select_component")
    return t[..., index].contiguous().to(_DTYPE)

register_operator("select_component", select_component, radius=0, cost="low")


def norm_last(t: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    _assert_fp64(t, "norm_last")
    return torch.linalg.norm(t, dim=-1, keepdim=True).clamp(min=eps).to(_DTYPE)

register_operator("norm_last", norm_last, radius=0, cost="low")


def div_last(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _assert_fp64(a, "div_last(a)")
    _assert_fp64(b, "div_last(b)")
    if b.ndim == a.ndim - 1:
        b = b.unsqueeze(-1)
    return (a / b).to(_DTYPE)

register_operator("div_last", div_last, radius=0, cost="low")


def neg_stack3(
    gx: torch.Tensor,
    gy: torch.Tensor,
) -> torch.Tensor:
    _assert_fp64(gx, "neg_stack3(gx)")
    _assert_fp64(gy, "neg_stack3(gy)")
    ones = torch.ones_like(gx)
    n    = torch.stack([-gx, -gy, ones], dim=-1)
    mag  = torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-15)
    return (n / mag).to(_DTYPE)

register_operator("neg_stack3", neg_stack3, radius=0, cost="low")


def scale_eye(
    scalar_field: torch.Tensor,
    D:            int,
) -> torch.Tensor:
    _assert_fp64(scalar_field, "scale_eye")
    s  = scalar_field
    I  = torch.eye(D, dtype=_DTYPE, device=scalar_field.device)
    return (s.unsqueeze(-1).unsqueeze(-1) * I).to(_DTYPE)

register_operator("scale_eye", scale_eye, radius=0, cost="low")


def velocity_dot_grad(
    velocity_component: torch.Tensor,
    grad_component:     torch.Tensor,
) -> torch.Tensor:
    _assert_fp64(velocity_component, "velocity_dot_grad(v)")
    _assert_fp64(grad_component,     "velocity_dot_grad(g)")
    return torch.mul(velocity_component, grad_component).to(_DTYPE)

register_operator("velocity_dot_grad", velocity_dot_grad, radius=0, cost="low")








# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

_DIST_MM: Optional[MemoryManager] = None


def _set_dist_mm(mm: MemoryManager) -> None:
    global _DIST_MM
    _DIST_MM = mm


def _get_dist_mm() -> MemoryManager:
    if _DIST_MM is None:
        raise RuntimeError("Distributed MemoryManager not initialized")
    return _DIST_MM


def _dist_all_reduce(t: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        return _get_dist_mm().all_reduce_sum(t)
    return t


def op_sum(
    t:       torch.Tensor,
    dim=None,
    keepdim: bool = False,
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "sum")
    local = t.sum(dim=dim, keepdim=keepdim)
    return _dist_all_reduce(local) if dim is None else local

register_operator("sum", op_sum, cost="low")


def mean(
    t:       torch.Tensor,
    dim=None,
    keepdim: bool = False,
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "mean")
    if dim is None and is_distributed():
        local_sum   = t.sum()
        local_count = torch.tensor(float(t.numel()), dtype=_DTYPE, device=t.device)
        return (_dist_all_reduce(local_sum) / _dist_all_reduce(local_count)).to(_DTYPE)
    return t.mean(dim=dim, keepdim=keepdim)

register_operator("mean", mean, cost="low")


def norm_l2(t: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "norm_l2")
    if is_distributed():
        return _dist_all_reduce((t * t).sum()).sqrt().to(_DTYPE)
    return t.norm()

register_operator("norm_l2", norm_l2, cost="low")


def min_max(t: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "min_max")
    return torch.stack([t.min(), t.max()])

register_operator("min_max", min_max, cost="low")


def moving_average(
    t:        torch.Tensor,
    window:   int = 3,
    boundary: str = "neumann",
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "moving_average")
    if t.ndim > 3:
        raise NotImplementedError("moving_average: only 1-D/2-D/3-D supported")
    pad    = window // 2
    weight = torch.full((1, 1, window), 1.0/window, dtype=_DTYPE, device=t.device)
    out    = t
    for d in range(t.ndim):
        p    = _pad1d(out, d, pad, boundary)
        perm = list(range(out.ndim)); perm[d], perm[-1] = perm[-1], perm[d]
        x    = p.permute(perm).contiguous()
        B    = math.prod(x.shape[:-1]) if x.ndim > 1 else 1
        y    = F.conv1d(x.reshape(B, 1, x.shape[-1]), weight)
        inv  = [0] * out.ndim
        for s, dd in enumerate(perm): inv[dd] = s
        out  = y.reshape(*x.shape[:-1], out.shape[d]).permute(inv).contiguous()
    return out.to(_DTYPE)

register_operator("moving_average", moving_average, cost="medium")


# ---------------------------------------------------------------------------
# Integral operators
# ---------------------------------------------------------------------------

def _simpson_1d(y: torch.Tensor, dx: float, dim: int) -> torch.Tensor:
    n = y.shape[dim]
    if n < 3:
        return torch.trapezoid(y, dx=dx, dim=dim)
    if n % 2 == 0:
        fst = y.select(dim, 0);   mid = y.select(dim, n-2);  lst = y.select(dim, n-1)
        odd_idx = torch.arange(1,  n-2, 2, device=y.device)
        evn_idx = torch.arange(2,  n-2, 2, device=y.device)
        odd  = y.index_select(dim, odd_idx).sum(dim) if odd_idx.numel() else torch.zeros_like(fst)
        evn  = y.index_select(dim, evn_idx).sum(dim) if evn_idx.numel() else torch.zeros_like(fst)
        simp = (dx/3.0)   * (fst + mid + 4.0*odd + 2.0*evn)
        trap = (dx*0.5)   * (mid + lst)
        return simp + trap
    odd_idx = torch.arange(1, n-1, 2, device=y.device)
    evn_idx = torch.arange(2, n-1, 2, device=y.device)
    fst = y.select(dim, 0); lst = y.select(dim, n-1)
    odd = y.index_select(dim, odd_idx).sum(dim)
    evn = y.index_select(dim, evn_idx).sum(dim) if evn_idx.numel() else torch.zeros_like(fst)
    return (dx/3.0) * (fst + lst + 4.0*odd + 2.0*evn)


def integrate(
    t:      torch.Tensor,
    dx:     Union[float, Tuple[float, ...]] = 1.0,
    dims:   Optional[Union[int, Tuple[int, ...]]] = None,
    method: str = "simpson",
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "integrate")
    ndim  = t.ndim
    dxs   = (dx,)*ndim if isinstance(dx, float) else tuple(dx)
    dim_seq = (
        tuple(range(ndim)) if dims is None
        else ((dims,) if isinstance(dims, int) else tuple(dims))
    )
    result = t
    for d in sorted(dim_seq, reverse=True):
        result = (
            _simpson_1d(result, dxs[d], d)
            if method == "simpson"
            else torch.trapezoid(result, dx=dxs[d], dim=d)
        )
    return result.to(_DTYPE)

register_operator("integrate", integrate, cost="medium")


def cumulative_integral(
    t:     torch.Tensor,
    dx:    float = 1.0,
    dim:   int   = 0,
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "cumulative_integral")
    n = t.shape[dim]
    if n < 2:
        return torch.zeros_like(t)

    def _sl(a, b):
        s = [slice(None)] * t.ndim
        s[dim] = slice(a, b)
        return tuple(s)

    trapz      = (t[_sl(None,-1)] + t[_sl(1,None)]) * (dx * 0.5)
    zero_shape = list(t.shape); zero_shape[dim] = 1
    zero       = torch.zeros(zero_shape, dtype=_DTYPE, device=t.device)
    return torch.cat([zero, torch.cumsum(trapz, dim=dim)], dim=dim).to(_DTYPE)

register_operator("cumulative_integral", cumulative_integral, cost="low")


def surface_integral(
    t:            torch.Tensor,
    normals:      torch.Tensor,
    area_weights: torch.Tensor,
    alloc=None,
) -> torch.Tensor:
    for x, n in ((t,"t"),(normals,"normals"),(area_weights,"area_weights")):
        _assert_fp64(x, f"surface_integral({n})")
    return ((t * normals).sum(dim=-1) * area_weights).sum().to(_DTYPE)

register_operator("surface_integral", surface_integral, cost="medium")


# ---------------------------------------------------------------------------
# Statistical operators
# ---------------------------------------------------------------------------

def variance(
    t:        torch.Tensor,
    dim=None,
    unbiased: bool = True,
    keepdim:  bool = False,
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "variance")
    return torch.var(t, dim=dim, unbiased=unbiased, keepdim=keepdim).to(_DTYPE)

register_operator("variance", variance, cost="low")


def covariance(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "covariance(a)"); _assert_fp64(b, "covariance(b)")
    if a.shape != b.shape:
        raise ValueError(f"covariance: shape mismatch {a.shape} vs {b.shape}")
    af = (a - a.mean()).flatten()
    bf = (b - b.mean()).flatten()
    return (torch.dot(af, bf) / max(af.numel()-1, 1)).to(_DTYPE)

register_operator("covariance", covariance, cost="medium")


def correlation(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "correlation(a)"); _assert_fp64(b, "correlation(b)")
    if a.shape != b.shape:
        raise ValueError(f"correlation: shape mismatch {a.shape} vs {b.shape}")
    af = (a - a.mean()).flatten()
    bf = (b - b.mean()).flatten()
    return (torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-15)).to(_DTYPE)

register_operator("correlation", correlation, cost="medium")


def entropy(
    t:   torch.Tensor,
    dim=None,
    eps: float = 1e-15,
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "entropy")
    p = t.abs()
    if dim is None:
        p = p / (p.sum() + eps)
        return -(p * (p + eps).log()).sum().to(_DTYPE)
    p = p / (p.sum(dim=dim, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(dim=dim).to(_DTYPE)

register_operator("entropy", entropy, cost="medium")


# ---------------------------------------------------------------------------
# Linear-algebra operators
# ---------------------------------------------------------------------------

def trace(t: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "trace")
    if t.ndim < 2 or t.shape[-1] != t.shape[-2]:
        raise ValueError(f"trace: need (*spatial, D, D), got {tuple(t.shape)}")
    return torch.diagonal(t, dim1=-2, dim2=-1).sum(-1).to(_DTYPE)

register_operator("trace", trace, cost="low")


def determinant(t: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "determinant")
    if t.ndim < 2 or t.shape[-1] != t.shape[-2]:
        raise ValueError(f"determinant: need (*spatial, D, D), got {tuple(t.shape)}")
    
    D = t.shape[-1]
    epsilon = 1e-12
    I_reg = torch.eye(D, dtype=_DTYPE, device=t.device)

    diag = torch.diagonal(t, dim1=-2, dim2=-1)
    if (diag.abs() < epsilon).any():
        t_reg = t + epsilon * I_reg
    else:
        t_reg = t
    
    try:
        det = torch.linalg.det(t_reg)
    except RuntimeError:
        det = torch.linalg.det(t + 1e-6 * I_reg)
    
    det = torch.nan_to_num(det, nan=0.0, posinf=1e30, neginf=-1e30)
    det = torch.clamp(det, min=-1e30, max=1e30)
    
    if alloc:
        buf = alloc(det.shape, det.device, key=_uid("det"))
        buf.copy_(det)
        return buf
    return det.to(_DTYPE)

register_operator("determinant", determinant, cost="high")


def eigenvalues(t: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "eigenvalues")
    if t.ndim < 2 or t.shape[-1] != t.shape[-2]:
        raise ValueError(f"eigenvalues: need (*spatial, D, D), got {tuple(t.shape)}")
    return torch.linalg.eigvalsh(t).to(_DTYPE)

register_operator("eigenvalues", eigenvalues, cost="very_high")


def inverse(t: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "inverse")
    if t.ndim < 2 or t.shape[-1] != t.shape[-2]:
        raise ValueError(f"inverse: need (*spatial, D, D), got {tuple(t.shape)}")
    return torch.linalg.inv(t).to(_DTYPE)

register_operator("inverse", inverse, cost="very_high")

# ---------------------------------------------------------------------------
# GPU-accelerated batched matrix multiply
# ---------------------------------------------------------------------------
def matmul_batched_gpu(
    a: torch.Tensor,
    alloc=None,
) -> torch.Tensor:
    """Batched A @ Aᵀ  (uses torch.bmm)."""
    n = a.shape[-1]
    b = a.reshape(-1, n, n)                     
    r = b @ b.transpose(-1, -2)             
    return r.reshape(*a.shape[:-2], n, n)     

register_operator(
    "matmul_batched_custom", matmul_batched_gpu,   
    cost="high", flops=2, bytes_in=8, bytes_out=8,
)


# ---------------------------------------------------------------------------
# GPU-accelerated 3×3×3 convolution (Gaussian blur)
# ---------------------------------------------------------------------------

def conv3d_spectral(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "conv3d_spectral")

    if not hasattr(conv3d_spectral, "kernel"):
        k = torch.tensor(
            [[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
             [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
             [[1, 2, 1], [2, 4, 2], [1, 2, 1]]],
            dtype=_DTYPE,
        ) / 64.0
        conv3d_spectral.kernel = k

    kernel = conv3d_spectral.kernel.to(a.device)
    s = a.shape

    k_full = torch.zeros(s, dtype=a.dtype, device=a.device)

    k_full[:3, :3, :3] = kernel
 
    k_full = torch.roll(k_full, shifts=(-1, -1, -1), dims=(0, 1, 2))

    A = torch.fft.fftn(a)
    K = torch.fft.fftn(k_full)
    C = torch.fft.ifftn(A * K)

    return C.real.to(_DTYPE)

register_operator(
    "conv3d_spectral", conv3d_spectral,
    cost="very_high", flops=2, bytes_in=8, bytes_out=8,
)

def deviatoric(t: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "deviatoric")
    if t.ndim < 2 or t.shape[-1] != t.shape[-2]:
        raise ValueError(f"deviatoric: need (*spatial, D, D), got {tuple(t.shape)}")
    
    D  = t.shape[-1]
    tr = torch.diagonal(t, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    I = torch.eye(D, dtype=_DTYPE, device=t.device)
    result = t - (tr / max(D, 1e-15)) * I
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    if alloc:
        buf = alloc(result.shape, result.device, key=_uid("deviatoric"))
        buf.copy_(result)
        return buf
    return result.to(_DTYPE)

register_operator("deviatoric", deviatoric, cost="medium")

# ---------------------------------------------------------------------------
# FFT operators 
# ---------------------------------------------------------------------------

def fft(t: torch.Tensor, norm: str = "backward", alloc=None) -> torch.Tensor:
    if t.dtype != _DTYPE and t.dtype != torch.complex128:
        _assert_fp64(t, "fft")         
    if not torch.is_complex(t):
        t = t.to(torch.complex128)      
    out = torch.fft.fftn(t, norm=norm)
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("fft"))
        buf.copy_(out)
        return buf
    return out

def ifft(t: torch.Tensor, norm: str = "backward", alloc=None) -> torch.Tensor:
    if not torch.is_complex(t):
        t = torch.complex(t, torch.zeros_like(t))
    result = torch.fft.ifftn(t, norm=norm)
    out = result.real.to(_DTYPE)      
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("ifft"))
        buf.copy_(out)
        return buf
    return out

register_operator("fft", fft, cost="high")
register_operator("ifft", ifft, cost="high")

def fft_complex(t: torch.Tensor, norm: str = "backward", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "fft_complex")
    if norm == "numpy_compatible":
        norm = "backward"
    result = torch.fft.fftn(t, norm=norm)
    if alloc:
        return result        
    return result

def ifft_complex(t: torch.Tensor, norm: str = "backward", alloc=None) -> torch.Tensor:
    if not torch.is_complex(t):
        t = torch.complex(t, torch.zeros_like(t))
    if norm == "numpy_compatible":
        norm = "backward"
    result = torch.fft.ifftn(t, norm=norm)
    if alloc:
        return result          
    return result


register_operator("fft_complex", fft_complex, cost="high")
register_operator("ifft_complex", ifft_complex, cost="high")

# ---------------------------------------------------------------------------
# Spectral operators 
# ---------------------------------------------------------------------------

@dataclass
class _KGridEntry:
    grids: List[torch.Tensor]
    k2:    torch.Tensor


_K_GRID_CACHE_MAX = int(os.environ.get("OPS_K_GRID_CACHE_MAX", "32"))
_K_GRID_CACHE: "OrderedDict[Tuple, _KGridEntry]" = collections.OrderedDict()
_K_GRID_CACHE_LOCK = threading.Lock()

def _k_grid_cached(shape, spacing, device_str):
    key = (shape, spacing, device_str)
    with _K_GRID_CACHE_LOCK:
        if key in _K_GRID_CACHE:
            _K_GRID_CACHE.move_to_end(key)
            return _K_GRID_CACHE[key]

    device = torch.device(device_str)
    grids = []
    for d, (n, dx) in enumerate(zip(shape, spacing)):
        k = torch.fft.fftfreq(n, d=dx, device=device).to(_DTYPE) * (2 * math.pi)
        sh = [1] * len(shape)
        sh[d] = -1
        grids.append(k.view(sh))
    k2 = sum(kg**2 for kg in grids)

    entry = _KGridEntry(grids=grids, k2=k2)

    with _K_GRID_CACHE_LOCK:
        if key in _K_GRID_CACHE:
            _K_GRID_CACHE.move_to_end(key)
            return _K_GRID_CACHE[key]
        _K_GRID_CACHE[key] = entry
        _K_GRID_CACHE.move_to_end(key)
        while len(_K_GRID_CACHE) > _K_GRID_CACHE_MAX:
            _K_GRID_CACHE.popitem(last=False)
    return entry


def _k_grid(
    shape:   Tuple[int, ...],
    spacing: Tuple[float, ...],
    device:  torch.device,
) -> _KGridEntry:
    return _k_grid_cached(shape, spacing, str(device))


def _clear_k_grid_cache() -> None:
    _K_GRID_CACHE.clear()



def spectral_gradient(
    t:   torch.Tensor,
    dx:  Union[float, Tuple[float, ...]] = 1.0,
    dim: int = 0,
    norm: str = "backward", 
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "spectral_gradient")
    ndim  = t.ndim
    dxs   = (dx,)*ndim if isinstance(dx, float) else tuple(dx)
    entry = _k_grid(t.shape, dxs, t.device)
    T_hat = torch.fft.fftn(t, norm=norm)
    result = torch.fft.ifftn(1j * entry.grids[dim] * T_hat, norm=norm)
    return result.real.to(_DTYPE)

register_operator("spectral_gradient", spectral_gradient, cost="high")


def spectral_laplacian(
    t:  torch.Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    norm: str = "backward",
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "spectral_laplacian")
    ndim  = t.ndim
    dxs   = (dx,)*ndim if isinstance(dx, float) else tuple(dx)
    entry = _k_grid(t.shape, dxs, t.device)
    T_hat = torch.fft.fftn(t, norm=norm)
    result = torch.fft.ifftn(-entry.k2 * T_hat, norm=norm)
    return result.real.to(_DTYPE)

register_operator("spectral_laplacian", spectral_laplacian, cost="high")


# ---------------------------------------------------------------------------
# Geometry operators
# ---------------------------------------------------------------------------







def distance_transform(
    t:  torch.Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    alloc=None,
) -> torch.Tensor:
    _assert_fp64(t, "distance_transform")
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("distance_transform requires scipy: pip install scipy")
    ndim = t.ndim
    dxs  = (dx,)*ndim if isinstance(dx, float) else tuple(dx)
    arr  = (t == 0).cpu().numpy()
    out  = distance_transform_edt(arr, sampling=list(dxs))
    return torch.from_numpy(out.astype(np.float64)).to(t.device).to(_DTYPE)

register_operator("distance_transform", distance_transform, cost="very_high")


# ---------------------------------------------------------------------------
# LazyField and I/O
# ---------------------------------------------------------------------------

_SPACING_HINTS = [
    "spacing", "dx", "dy", "dz", "delta_x", "delta_y", "delta_z",
    "resolution", "grid_spacing", "cell_size", "h",
]


class LazyField:
    def __init__(self, path: str, dataset_name: str = "field") -> None:
        self.path         = path
        self.dataset_name = dataset_name
        self._lock        = threading.RLock()
        self._local       = threading.local()
        self._all_files: List[h5py.File] = []

    def _ensure_open(self) -> None:
        if getattr(self._local, "file", None) is not None:
            return
        

        with self._lock:
            if getattr(self._local, "file", None) is not None:
                return
            try:
                f = h5py.File(self.path, "r", swmr=True)
            except Exception:
                f = h5py.File(self.path, "r")
            
            self._local.file = f
            self._local.dset = None
            
            self._all_files.append(f)
            if len(self._all_files) > _MAX_LAZY_FIELD_HANDLES:
                old = self._all_files.pop(0)
                try:
                    old.close()
                except Exception:
                    pass

            for cand in (self.dataset_name, "data", "field"):
                if cand in f and isinstance(f[cand], h5py.Dataset):
                    self._local.dset = f[cand]
                    break
            
            if self._local.dset is None:
                for k in f:
                    if isinstance(f[k], h5py.Dataset):
                        self._local.dset = f[k]
                        break
            
            if self._local.dset is None:
                raise KeyError(f"No array dataset found in {self.path}")

    @property
    def shape(self) -> Tuple:
        with self._lock:
            self._ensure_open()
            return self._local.dset.shape

    @property
    def dtype(self) -> type:
        return np.float64

    def __getitem__(self, idx: Any) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            return self._local.dset[idx]

    def close(self) -> None:
        with self._lock:
            for f in self._all_files:
                try: f.close()
                except Exception: pass
            self._all_files.clear()
        if getattr(self._local, "file", None) is not None:
            self._local.file = None
            self._local.dset = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        try:    sh = self.shape
        except: sh = "?"
        return f"LazyField(path={self.path}, shape={sh}, fp64)"


@contextlib.contextmanager
def _open_hdf5_field(path: str) -> Iterator[Tuple[h5py.Dataset, Tuple[int, ...]]]:
    f = h5py.File(path, "r")
    try:
        for cand in ("data", "field"):
            if cand in f and isinstance(f[cand], h5py.Dataset):
                yield f[cand], tuple(f[cand].shape)
                return
        for k in f:
            if isinstance(f[k], h5py.Dataset):
                yield f[k], tuple(f[k].shape)
                return
        raise KeyError(f"No array dataset found in {path}")
    finally:
        f.close()


def load_tensor(
    path:          str,
    device:        Union[str, torch.device] = "cpu",
    normalize:     bool                     = False,
    return_mode:   str                      = "lazy",
    max_eager_gb:  float                    = 0.5,
    **nkw,
) -> Tuple[Union[LazyField, torch.Tensor], Tuple[float, ...], Tuple[float, ...]]:
    if normalize:
        path = _normalize_hdf5(path, **nkw)

    dev = torch.device(device)

    with h5py.File(path, "r") as f:
        for cand in ("field", "data"):
            if cand in f and isinstance(f[cand], h5py.Dataset):
                dset = f[cand]; break
        else:
            raise KeyError(f"No field dataset found in {path}")

        if "spacing" in f:
            spacing = tuple(float(v) for v in f["spacing"][:])
        elif "spacing" in f.attrs:
            spacing = tuple(float(v) for v in np.atleast_1d(f.attrs["spacing"]))
        elif "spacing" in dset.attrs:
            spacing = tuple(float(v) for v in np.atleast_1d(dset.attrs["spacing"]))
        else:
            spacing = (1.0,) * max(1, dset.ndim)
        if "origin" in f:
            origin = tuple(float(v) for v in f["origin"][:])
        elif "origin" in f.attrs:
            origin = tuple(float(v) for v in np.atleast_1d(f.attrs["origin"]))
        elif "origin" in dset.attrs:
            origin = tuple(float(v) for v in np.atleast_1d(dset.attrs["origin"]))
        else:
            origin = tuple(0.0 for _ in spacing)

        shape     = dset.shape
        total_gb  = np.prod(shape) * 8 / 1024**3

    if return_mode == "auto":
        return_mode = "eager" if total_gb <= max_eager_gb else "lazy"

    if return_mode == "lazy":
        return LazyField(path), spacing, origin

    if return_mode == "eager":
        if total_gb > max_eager_gb:
            raise MemoryError(
                f"Refusing to eagerly load {total_gb:.2f} GB "
                f"(limit={max_eager_gb} GB). Use return_mode='lazy'."
            )
        with h5py.File(path, "r") as f:
            for cand in ("field", "data"):
                if cand in f:
                    arr = f[cand][:]; break
        tensor = torch.from_numpy(arr).to(_DTYPE)
        if dev.type == "cuda" and torch.cuda.is_available():
            try:
                tensor = tensor.pin_memory().to(dev, non_blocking=True)
                torch.cuda.synchronize()
            except Exception:
                tensor = tensor.to(dev)
        else:
            tensor = tensor.to(dev)
        return tensor, spacing, origin

    raise ValueError(f"Invalid return_mode: {return_mode!r}")


def save_tensor(
    tensor:  torch.Tensor,
    spacing: Tuple[float, ...],
    origin:  Tuple[float, ...],
    path:    str,
    chunks:  bool = True,
) -> None:
    _assert_fp64(tensor, "save_tensor")
    arr   = tensor.cpu().numpy()
    chunk = tuple(min(128, s) for s in arr.shape) if chunks else None
    with h5py.File(path, "w") as f:
        f.create_dataset("field",   data=arr,                               chunks=chunk)
        f.create_dataset("spacing", data=np.array(spacing, dtype=np.float64))
        f.create_dataset("origin",  data=np.array(origin,  dtype=np.float64))


def _normalize_hdf5(
    path:           str,
    field_key:      str                           = "field",
    spacing_key:    str                           = "spacing",
    origin_key:     str                           = "origin",
    spacing_value:  Optional[Tuple[float, ...]]   = None,
    output_path:    Optional[str]                 = None,
) -> str:
    import tempfile
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".h5", prefix="ops_norm_")
        os.close(fd)

    with h5py.File(path, "r") as fi, h5py.File(output_path, "w") as fo:
        if field_key not in fi:
            cands = [k for k in fi if isinstance(fi[k], h5py.Dataset) and fi[k].ndim >= 1]
            if not cands:
                raise KeyError(f"No array dataset found in {path}")
            field_key = cands[0]
            warnings.warn(f"_normalize_hdf5: using '{field_key}' as field")
        data  = fi[field_key][:]
        chunk = tuple(min(128, s) for s in data.shape)
        fo.create_dataset("field", data=data, chunks=chunk)

        spacing = None
        if spacing_key in fi:
            spacing = fi[spacing_key][:].tolist()
        else:
            for src in (fi[field_key].attrs, fi.attrs):
                for hint in _SPACING_HINTS:
                    if hint in src:
                        v = src[hint]
                        spacing = (
                            [float(v)] * data.ndim
                            if np.isscalar(v) else list(v)
                        )
                        break
                if spacing:
                    break
        if spacing is None and spacing_value:
            spacing = list(spacing_value)
        if spacing is None:
            spacing = [1.0] * data.ndim
            warnings.warn(f"_normalize_hdf5: spacing not found in {path}, defaulting to 1.0")
        fo.create_dataset("spacing", data=np.array(spacing, dtype=np.float64))

        origin = (
            fi[origin_key][:].tolist()
            if origin_key in fi
            else list(fi.attrs.get("origin", [0.0] * data.ndim))
        )
        fo.create_dataset("origin", data=np.array(origin, dtype=np.float64))

    return output_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(
    op_name:   str,
    in_tensor: torch.Tensor,
    out_tensor: torch.Tensor,
    **params,
) -> None:
    if _VALIDATION_LEVEL == "none":
        return
    if torch.isnan(out_tensor).any() or torch.isinf(out_tensor).any():
        raise ValueError(f"{op_name}: output contains NaN/Inf")
    if _VALIDATION_LEVEL == "sample":
        flat  = out_tensor.flatten()
        n     = max(1, min(int(flat.numel() * _VALIDATION_SAMPLE), flat.numel()))
        idx   = torch.randint(0, flat.numel(), (n,), device=flat.device)
        out_n = flat[idx].norm().item()
        in_f  = in_tensor.flatten()
        in_n  = (
            in_f[idx].norm().item()
            if in_f.numel() >= flat.numel()
            else in_f.norm().item()
        )
    else:
        out_n = out_tensor.norm().item()
        in_n  = in_tensor.norm().item()
    if in_n > 0 and out_n / (in_n + 1e-15) > 1e6:
        warnings.warn(
            f"{op_name}: output norm {out_n/in_n:.2e}× input — possible blow-up"
        )


# ---------------------------------------------------------------------------
# Distributed support
# ---------------------------------------------------------------------------

_DIST_INITIALIZED    = False
_DIST_RANK           = 0
_DIST_WORLD_SIZE     = 1
_DIST_CART_TOPOLOGY: Optional[Dict] = None


def dist_init(
    backend: str = "nccl" if torch.cuda.is_available() else "gloo",
    dims: Optional[List[int]] = None,
) -> bool:
    global _DIST_INITIALIZED, _DIST_RANK, _DIST_WORLD_SIZE, _DIST_CART_TOPOLOGY
    
    if _DIST_INITIALIZED:
        return True
    
    if not dist.is_available():
        warnings.warn("torch.distributed not available – multi-node disabled")
        return False
    
    if not dist.is_initialized():
        try:
            timeout_minutes = int(os.environ.get("OPS_DIST_TIMEOUT_MINUTES", "30"))
            if backend == "nccl":
                os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                timeout=datetime.timedelta(minutes=timeout_minutes),
            )
            _DIST_RANK = dist.get_rank()
            _DIST_WORLD_SIZE = dist.get_world_size()
            if dims is not None:
                _DIST_CART_TOPOLOGY = _create_cartesian_topology(
                    dims, [True] * len(dims)
                )
            _DIST_INITIALIZED = True
        except Exception as e:
            warnings.warn(f"Failed to init distributed: {e}")
            return False
    else:
        _DIST_RANK = dist.get_rank()
        _DIST_WORLD_SIZE = dist.get_world_size()
        _DIST_INITIALIZED = True
    
    return True



def _create_cartesian_topology(dims: List[int], periods: List[bool]) -> Dict:
    world = dist.get_world_size()
    rank_to_coord: Dict[int, Tuple]  = {}
    coord_to_rank: Dict[Tuple, int]  = {}
    for r in range(world):
        coords, tmp = [], r
        for d in range(len(dims)-1, -1, -1):
            coords.insert(0, tmp % dims[d])
            tmp //= dims[d]
        coords = tuple(coords)
        rank_to_coord[r] = coords
        coord_to_rank[coords] = r
    neighbors: Dict[int, Dict] = {}
    for r, coord in rank_to_coord.items():
        neigh: Dict = {}
        for d in range(len(dims)):
            for direction, delta in ((-1,-1),(1,1)):
                nc = list(coord); nc[d] += delta
                if 0 <= nc[d] < dims[d]:
                    neigh[(direction, d)] = coord_to_rank[tuple(nc)]
                elif periods[d]:
                    nc[d] %= dims[d]
                    neigh[(direction, d)] = coord_to_rank[tuple(nc)]
        neighbors[r] = neigh
    return {
        "dims": dims, "periods": periods,
        "rank_to_coord": rank_to_coord, "neighbors": neighbors,
    }


def dist_rank()  -> int:  return _DIST_RANK       if _DIST_INITIALIZED else 0
def dist_size()  -> int:  return _DIST_WORLD_SIZE  if _DIST_INITIALIZED else 1
def is_distributed() -> bool:
    return _DIST_INITIALIZED and _DIST_WORLD_SIZE > 1


def _halo_exchange_nd(
    shard:  torch.Tensor,
    radius: int,
    dims:   List[int],
) -> torch.Tensor:
    if not is_distributed() or radius <= 0 or _DIST_CART_TOPOLOGY is None:
        return shard
    rank      = dist_rank()
    neighbors = _DIST_CART_TOPOLOGY["neighbors"][rank]
    reqs, recv_bufs = [], []
    for (direction, d), neigh_rank in neighbors.items():
        if d not in dims:
            continue
        sl = [slice(None)] * shard.ndim
        sl[d] = (
            slice(radius, 2*radius) if direction == -1
            else (
                slice(-2*radius, -radius) if shard.shape[d] >= 2*radius
                else slice(-radius, None)
            )
        )
        send_buf = shard[tuple(sl)].clone()
        recv_buf = torch.empty_like(send_buf)
        if rank < neigh_rank:
            reqs += [dist.isend(send_buf, dst=neigh_rank),
                    dist.irecv(recv_buf, src=neigh_rank)]
        else:
            reqs += [dist.irecv(recv_buf, src=neigh_rank),
                    dist.isend(send_buf, dst=neigh_rank)]
        recv_bufs.append((d, direction, recv_buf))
    for req in reqs: req.wait()
    for d, direction, buf in recv_bufs:
        sl    = [slice(None)] * shard.ndim
        sl[d] = slice(0, radius) if direction == -1 else slice(-radius, None)
        shard[tuple(sl)] = buf
    return shard


def _halo_exchange_1d(shard: torch.Tensor, radius: int) -> torch.Tensor:
    if not is_distributed() or radius <= 0 or dist_size() == 1:
        return shard
    rank       = dist_rank()
    world      = dist_size()
    left_rank  = rank - 1 if rank > 0       else None
    right_rank = rank + 1 if rank < world-1 else None
    reqs       = []

    def _buf(side: str):
        sl    = [slice(None)] * shard.ndim
        sl[0] = (
            slice(radius, 2*radius) if side == "left"
            else slice(-2*radius, -radius)
        )
        return (shard[tuple(sl)].clone()
                if shard.shape[0] >= 2*radius else None)

    send_l, send_r = _buf("left"), _buf("right")
    recv_l = torch.empty_like(send_l) if left_rank  is not None and send_l is not None else None
    recv_r = torch.empty_like(send_r) if right_rank is not None and send_r is not None else None
    if left_rank is not None and recv_l is not None:
        reqs.append(dist.irecv(recv_l, src=left_rank))
    if right_rank is not None and recv_r is not None:
        reqs.append(dist.irecv(recv_r, src=right_rank))

    if left_rank is not None and send_l is not None:
        reqs.append(dist.isend(send_l, dst=left_rank))
    if right_rank is not None and send_r is not None:
        reqs.append(dist.isend(send_r, dst=right_rank))

    for req in reqs:
        req.wait()
    if recv_l is not None: shard[:radius]  = recv_l
    if recv_r is not None: shard[-radius:] = recv_r
    return shard


class _DistMemoryManager(MemoryManager):
    def halo_exchange(
        self,
        shard:  torch.Tensor,
        radius: int,
        dims:   Optional[List[int]] = None,
    ) -> torch.Tensor:
        if dims is None:
            dims = [0]
        return (
            _halo_exchange_nd(shard, radius, dims)
            if len(dims) > 1 and _DIST_CART_TOPOLOGY is not None
            else _halo_exchange_1d(shard, radius)
        )

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        if not is_distributed():
            return t
        out = t.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


if dist.is_available() and is_distributed():
    MemoryManager = _DistMemoryManager


def decompose_field(
    field:   Field,
    dim:     int = 0,
    overlap: int = 0,
) -> Tuple[Field, Tuple, Tuple[int, int]]:
    if not is_distributed():
        if overlap > 0:
            shape      = list(field.shape)
            shape[dim] += 2 * overlap
            padded     = torch.zeros(shape, dtype=_DTYPE, device=field.tensor.device)
            dst        = [slice(None)] * field.ndim
            dst[dim]   = slice(overlap, overlap + field.shape[dim])
            padded[tuple(dst)] = field.tensor
            new_origin    = list(field.origin)
            new_origin[dim] -= overlap * field.spacing[dim]
            return (
                Field(padded, field.spacing, tuple(new_origin), field._mm, field._key),
                (slice(None),), (0, field.shape[dim]),
            )
        return field, (slice(None),), (0, field.shape[dim])

    rank  = dist_rank(); world = dist_size()
    size  = field.shape[dim]
    chunk = (size + world - 1) // world
    start = rank * chunk; end = min(start + chunk, size)
    lo    = max(0, start - overlap); hi = min(size, end + overlap)
    new_shape      = list(field.shape); new_shape[dim] = hi - lo
    local_tensor   = torch.zeros(new_shape, dtype=_DTYPE, device=field.tensor.device)
    src            = [slice(None)] * field.ndim; src[dim] = slice(lo, hi)
    dst            = [slice(None)] * field.ndim
    off            = overlap if start > 0 else 0
    dst[dim]       = slice(off, off + (hi - lo))
    local_tensor[tuple(dst)] = field.tensor[tuple(src)]
    new_origin     = list(field.origin); new_origin[dim] += lo * field.spacing[dim]
    return (
        Field(local_tensor, field.spacing, tuple(new_origin), field._mm, field._key),
        tuple(src), (start, end),
    )


def load_tensor_parallel(
    path:      str,
    device:    Union[str, torch.device] = "cpu",
    normalize: bool                     = False,
    **nkw,
) -> Tuple[torch.Tensor, Tuple[float, ...], Tuple[float, ...]]:
    if not is_distributed() or not _HAS_MPI:
        return load_tensor(path, device, normalize, **nkw)
    try:
        comm  = _MPI.COMM_WORLD
        rank  = comm.Get_rank(); world = comm.Get_size()
        with h5py.File(path, "r", driver="mpio", comm=comm) as f:
            spacing = tuple(float(v) for v in f["spacing"][:])
            origin  = tuple(float(v) for v in f["origin"][:]) if "origin" in f else tuple(0.0 for _ in spacing)
            dataset = f["field"]
            n       = dataset.shape[0]
            chunk   = (n + world - 1) // world
            lo, hi  = rank*chunk, min(rank*chunk + chunk, n)
            data    = np.empty((hi-lo,) + dataset.shape[1:], dtype=np.float64)
            if hi > lo:
                dataset.read_direct(data, source_sel=np.s_[lo:hi], dest_sel=np.s_[:])
        return torch.from_numpy(data).to(_DTYPE).to(torch.device(device)), spacing, origin
    except Exception as exc:
        warnings.warn(f"Parallel HDF5 failed ({exc}), falling back to serial")
        return load_tensor(path, device, normalize, **nkw)


# ---------------------------------------------------------------------------
# execute helpers
# ---------------------------------------------------------------------------

def _execute_serial(
    op_name:    str,
    input_path: str,
    out_path:   str,
    device:     str   = "cpu",
    tile_shape: Optional[Tuple] = None,
    normalize:  bool  = False,
    dx:         Union[float, Tuple[float, ...]] = 1.0,
    boundary:   str   = "dirichlet",
    **params,
) -> None:
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unknown operator '{op_name}' (known: {list(OP_REGISTRY)})")
    mm = MemoryManager(); rt = Runtime(mm, device=device)
    dev = torch.device(device)
    raw_t, spacing, origin = load_tensor(input_path, device=dev, normalize=normalize)
    _assert_fp64(raw_t, "execute/load")
    key_in   = _uid("input")
    mm_input = mm.allocate(raw_t.shape, dev, key=key_in)
    mm_input.copy_(raw_t); del raw_t
    g = Graph()
    src_id   = g.add("_constant", (), {"value": mm_input})
    full_params = {"dx": dx, "boundary": boundary, **params}
    sink_id  = g.add(op_name, (src_id,), full_params)
    out_tensor = rt.run(g).get(sink_id)
    if isinstance(out_tensor, torch.Tensor):
        if not mm.owns(out_tensor):
            tmp = mm.allocate(out_tensor.shape, out_tensor.device)
            tmp.copy_(out_tensor); out_tensor = tmp
        _assert_fp64(out_tensor, f"execute/{op_name}")
        validate_output(op_name, mm_input, out_tensor, **full_params)
        save_tensor(out_tensor, spacing, origin, out_path)
        print(f"[stenpy] '{op_name}' fp64 on {device} → {out_path}")
        if mm.owns(out_tensor):
            mm.release(out_tensor)
    else:
        print(f"[stenpy] '{op_name}' → scalar: {out_tensor}")
    mm.release(mm_input); mm.clear_pool()


def execute_dist(
    op_name:    str,
    input_path: str,
    out_path:   str,
    device:     str   = "cpu",
    tile_shape: Optional[Tuple] = None,
    normalize:  bool  = False,
    dx:         Union[float, Tuple[float, ...]] = 1.0,
    boundary:   str   = "dirichlet",
    **params,
) -> None:
    if not is_distributed():
        return _execute_serial(
            op_name, input_path, out_path, device,
            tile_shape, normalize, dx, boundary, **params,
        )
    raw_t, spacing, origin = load_tensor_parallel(
        input_path, device=device, normalize=normalize
    )
    _assert_fp64(raw_t, "execute_dist/load")
    mm = MemoryManager(); rt = Runtime(mm, device=device)
    dev    = torch.device(device)
    key_in = _uid("input")
    mm_input = mm.allocate(raw_t.shape, dev, key=key_in)
    mm_input.copy_(raw_t); del raw_t
    overlap    = OP_METADATA.get(op_name, {}).get("stencil_radius", 0)
    field_in   = Field(mm_input, spacing, origin, mm=mm, key=key_in)
    local_field, _, _ = decompose_field(field_in, dim=0, overlap=overlap)
    mm_input   = local_field.tensor
    g          = Graph()
    src_id     = g.add("_constant", (), {"value": mm_input})
    full_params = {"dx": dx, "boundary": boundary, **params}
    sink_id    = g.add(op_name, (src_id,), full_params)
    out_tensor = rt.run(g).get(sink_id)
    if isinstance(out_tensor, torch.Tensor) and not mm.owns(out_tensor):
        tmp = mm.allocate(out_tensor.shape, out_tensor.device)
        tmp.copy_(out_tensor); out_tensor = tmp
    if isinstance(out_tensor, torch.Tensor):
        _assert_fp64(out_tensor, f"execute_dist/{op_name}")
        out_valid = out_tensor[overlap:-overlap] if overlap > 0 and out_tensor.shape[0] > 2*overlap else out_tensor
        in_valid  = mm_input[overlap:-overlap]   if overlap > 0 and mm_input.shape[0]   > 2*overlap else mm_input
        validate_output(op_name, in_valid, out_valid, **full_params)
        rank  = dist_rank(); world = dist_size()
        local_size = torch.tensor([out_valid.shape[0]], device=out_valid.device)
        sizes = [torch.zeros_like(local_size) for _ in range(world)]
        dist.all_gather(sizes, local_size)
        sizes = [int(s.item()) for s in sizes]
        max_size = max(sizes)
        padded   = (
            F.pad(out_valid, [0, max_size - out_valid.shape[0]] + [0,0]*(out_valid.ndim-1))
            if out_valid.shape[0] < max_size else out_valid
        )
        gathered = [torch.zeros_like(padded) for _ in range(world)]
        dist.all_gather(gathered, padded)
        if rank == 0:
            full_out = torch.cat([g[:sizes[i]] for i, g in enumerate(gathered)], dim=0)
            save_tensor(full_out, spacing, origin, out_path)
            mm.release(full_out)
        print(f"[stenpy] Distributed '{op_name}' rank {rank}/{world} done")
        if mm.owns(out_tensor):
            mm.release(out_tensor)
    else:
        print(f"[stenpy] Distributed '{op_name}' → scalar: {out_tensor}")
    mm.release(mm_input); mm.clear_pool()


def execute(*args, **kwargs) -> None:
    (execute_dist if is_distributed() else _execute_serial)(*args, **kwargs)


# ---------------------------------------------------------------------------
# Capability report
# ---------------------------------------------------------------------------

def capability_report() -> str:
    cost_counts: Dict[str, int] = {}
    for meta in OP_METADATA.values():
        c = meta["cost"]
        cost_counts[c] = cost_counts.get(c, 0) + 1
    lines = [
        "=" * 64,
        f"  Registered operators : {len(OP_REGISTRY)} (incl. _constant)",
        f"  Compute dtype        : float64 (enforced)",
        f"  CUDA available       : {torch.cuda.is_available()}",
        f"  Distributed          : {'yes' if is_distributed() else 'no (call dist_init())'}",
        f"  Validation level     : {_VALIDATION_LEVEL}",
        f"  VRAM headroom frac   : {_VRAM_HEADROOM_FRAC:.0%}",
        f"  Pinned staging pool  : {_PINNED_POOL._max} buffers",
        "",
        "  Cost distribution:",
    ]
    for c in ("low", "medium", "high", "very_high"):
        lines.append(f"    {c:12s}  {cost_counts.get(c, 0):3d} ops")
    lines.append("=" * 64)
    return "\n".join(lines)


def validate_fft_consistency() -> None:
    t = torch.randn(64, 64, dtype=_DTYPE)
    
    for norm in ['backward', 'ortho', 'forward']:
        F = fft(t, norm=norm)
        t_back = ifft(F, norm=norm)
        error = (t - t_back).abs().max().item()
        if error > 1e-10:
            raise ValueError(f"FFT+IFFT not invertible with norm='{norm}': error={error:.2e}")
        
        F_pt = torch.fft.fftn(t, norm=norm)
        t_back_pt = torch.fft.ifftn(F_pt, norm=norm).real
        error_pt = (t - t_back_pt).abs().max().item()
        
        if error_pt > 1e-10:
            raise ValueError(f"PyTorch FFT+IFFT not invertible: error={error_pt:.2e}")
    
    _dbg("validate_fft_consistency: all normalizations pass ✓")

def validate_operator_identity() -> None:
    t   = torch.randn(32, 32, dtype=_DTYPE)
    lap = laplacian(t, dx=1.0, boundary="periodic")
    lap_approx = torch.zeros_like(t)
    for d in range(t.ndim):
        g  = _grad2_along_dim(t, d, 1.0, "periodic")
        g2 = _grad2_along_dim(g, d, 1.0, "periodic")
        lap_approx = lap_approx + g2
    err = (lap_approx - lap).abs().mean().item()
    if err > 1e-2:
        raise ValueError(
            f"Operator inconsistency: sum(d²f/dx_d²) - laplacian(f) error = {err:.3e}"
        )
    _dbg(f"validate_operator_identity: error = {err:.2e} ✓")


__all__ = [
    "Field", "MemoryManager", "Graph", "Runtime",
    "register_operator", "OP_REGISTRY", "OP_METADATA",
    "execute", "execute_dist",
    "dist_init", "is_distributed", "dist_rank", "dist_size",
    "decompose_field",
    "load_tensor", "save_tensor", "load_tensor_parallel",
    "capability_report", "validate_output", "validate_operator_identity",
    "use_advanced_mm",
    "LazyField", "_open_hdf5_field",
    "_clear_k_grid_cache", "_vram_headroom_ok", "_vram_free_bytes",
    "_PINNED_POOL",
]
