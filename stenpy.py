#stenpy.py

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

_DTYPE = torch.float64

_DEBUG = os.environ.get("OPS_DEBUG", "0") == "1"
_ENV_TILE = os.environ.get("OPS_TILE_SHAPE", "")
_VALIDATION_SAMPLE = float(os.environ.get("OPS_VALIDATION_SAMPLE", "0.01"))
_VALIDATION_LEVEL = os.environ.get("OPS_VALIDATION", "basic")
_VRAM_HEADROOM_FRAC = float(os.environ.get("OPS_VRAM_HEADROOM", "0.15"))

_call_counter = itertools.count()


def _uid(prefix: str = "t") -> str:
    return f"{prefix}_{next(_call_counter)}"


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[ops] {msg}")


def _assert_fp64(t: torch.Tensor, where: str) -> None:
    if t.dtype != _DTYPE:
        raise TypeError(f"{where}: expected float64, got {t.dtype}")


def _vram_headroom_ok(device: torch.device, needed_bytes: int = 0) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    props = torch.cuda.get_device_properties(device)
    free = props.total_memory - torch.cuda.memory_allocated(device)
    headroom = props.total_memory * _VRAM_HEADROOM_FRAC
    return free - needed_bytes > headroom


def _vram_free_bytes(device: torch.device) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 2**62
    props = torch.cuda.get_device_properties(device)
    return props.total_memory - torch.cuda.memory_allocated(device)


# -----------------------------------------------------------------------------
# Pinned staging buffer pool
# -----------------------------------------------------------------------------

class _PinnedStagingPool:

    def __init__(self, max_buffers: int = 4) -> None:
        self._lock = threading.Lock()
        self._free: List[torch.Tensor] = []
        self._max = max_buffers
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
    arr: np.ndarray,
    device: torch.device,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    nbytes = arr.nbytes
    pinned = _PINNED_POOL.get(nbytes)
    if pinned is not None:
        pinned_np = pinned.numpy()
        np.copyto(pinned_np, arr.ravel().view(np.uint8))
        src = pinned.view(torch.float64).reshape(arr.shape)
        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with ctx:
            gpu_t = src.to(device, non_blocking=True)
        return gpu_t, pinned
    else:
        t = torch.from_numpy(arr.astype(np.float64, copy=False)).to(_DTYPE)
        return t.to(device), None


# -----------------------------------------------------------------------------
# Field and MemoryManager
# -----------------------------------------------------------------------------

class Field:
    __slots__ = ("tensor", "spacing", "origin", "_mm", "_key")

    def __init__(
        self,
        tensor: torch.Tensor,
        spacing: Tuple[float, ...],
        origin: Optional[Tuple[float, ...]] = None,
        mm: Optional["MemoryManager"] = None,
        key: Optional[str] = None,
    ) -> None:
        if tensor.dtype != _DTYPE:
            raise TypeError(f"Field requires float64 tensor, got {tensor.dtype}")
        self.tensor = tensor
        self.spacing = tuple(float(s) for s in spacing)
        self.origin = tuple(origin) if origin is not None else tuple(0.0 for _ in spacing)
        self._mm = mm
        self._key = key

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
        self._key = None

    def __enter__(self) -> "Field":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __repr__(self) -> str:
        sh = tuple(self.tensor.shape) if self.tensor is not None else None
        return f"Field(shape={sh}, spacing={self.spacing}, fp64)"


class MemoryManager:

    _DEFAULT_POOL_DEPTH = int(os.environ.get("OPS_POOL_DEPTH", "4"))

    def __init__(self, pool_depth: int = _DEFAULT_POOL_DEPTH) -> None:
        self._pool: Dict[Tuple, List[torch.Tensor]] = collections.defaultdict(list)
        self._lock = threading.Lock()
        self._pool_depth = pool_depth
        self._live: Dict[str, torch.Tensor] = {}
        self._pool_ptrs: Dict[Tuple, set] = {}

    @staticmethod
    def _pool_key(
        shape: Tuple[int, ...],
        device: torch.device,
        layout: torch.memory_format = torch.contiguous_format,
    ) -> Tuple:
        return (shape, str(device), layout)

    def _tag(self, t: torch.Tensor, key: str) -> None:
        t._mm_key = key

    def allocate(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        key: Optional[str] = None,
        layout: torch.memory_format = torch.contiguous_format,
    ) -> torch.Tensor:
        shape = tuple(shape)
        if device.type == "cuda":
            needed = math.prod(shape) * 8
            if not _vram_headroom_ok(device, needed):
                raise MemoryError(f"Insufficient VRAM headroom for shape {shape} on {device}")

        pk = self._pool_key(shape, device, layout)
        with self._lock:
            pool = self._pool[pk]
            ptrs = self._pool_ptrs.setdefault(pk, set())
            if pool:
                t = pool.pop()
                ptr = t.data_ptr()
                ptrs.discard(ptr)
                t.zero_()
            else:
                t = torch.zeros(shape, dtype=_DTYPE, device=device).to(memory_format=layout)
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
            pk = self._pool_key(tuple(tensor.shape), tensor.device)
            pool = self._pool[pk]
            ptrs = self._pool_ptrs.setdefault(pk, set())
            ptr = tensor.data_ptr()
            if ptr in ptrs:
                return
            pool.append(tensor)
            ptrs.add(ptr)
            while len(pool) > self._pool_depth * 2:
                old = pool.pop(0)
                ptrs.discard(old.data_ptr())

    def should_manage(self, shape: Tuple[int, ...]) -> bool:
        return math.prod(shape) >= 1024

    def make_field(
        self,
        tensor: torch.Tensor,
        spacing: Tuple[float, ...],
        origin: Optional[Tuple[float, ...]] = None,
        key: Optional[str] = None,
    ) -> Field:
        _assert_fp64(tensor, "MemoryManager.make_field")
        k = key or _uid("field")
        with self._lock:
            self._live[k] = tensor
            self._tag(tensor, k)
        return Field(tensor=tensor, spacing=spacing, origin=origin, mm=self, key=k)

    def allocate_field(
        self,
        shape: Tuple[int, ...],
        spacing: Tuple[float, ...],
        origin: Optional[Tuple[float, ...]] = None,
        device: torch.device = torch.device("cpu"),
        key: Optional[str] = None,
        layout: torch.memory_format = torch.contiguous_format,
    ) -> Field:
        k = key or _uid("field")
        t = self.allocate(shape, device, key=k, layout=layout)
        return Field(tensor=t, spacing=spacing, origin=origin, mm=self, key=k)

    def clear_pool(self) -> None:
        with self._lock:
            self._pool.clear()
            self._pool_ptrs.clear()

    def halo_exchange(self, shard: torch.Tensor, radius: int, dims: Optional[List[int]] = None) -> torch.Tensor:
        return shard

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def memory_pressure(self, device: Optional[torch.device] = None) -> float:
        if not torch.cuda.is_available():
            return 0.0
        dev = device or torch.device("cuda", 0)
        try:
            return torch.cuda.memory_allocated(dev) / torch.cuda.get_device_properties(dev).total_memory
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
        warnings.warn(f"memory_manager.py not found; staying with built-in pool ({exc})")


# -----------------------------------------------------------------------------
# Graph and Runtime
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphNode:
    id: str
    op_name: str
    input_ids: Tuple[str, ...] = dc_field(default_factory=tuple)
    params: Dict[str, Any] = dc_field(default_factory=dict)


class Graph:
    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._order: List[str] = []

    def add(
        self,
        op_name: str,
        input_ids: Tuple[str, ...] = (),
        params: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or _uid(op_name)
        node = GraphNode(
            id=nid,
            op_name=op_name,
            input_ids=tuple(input_ids),
            params=dict(params or {}),
        )
        self._nodes[nid] = node
        self._order.append(nid)
        return nid

    def topological_sort(self) -> List[GraphNode]:
        in_deg: Dict[str, int] = {nid: 0 for nid in self._nodes}
        children: Dict[str, List[str]] = {nid: [] for nid in self._nodes}
        for nid, node in self._nodes.items():
            for dep in node.input_ids:
                if dep not in self._nodes:
                    raise ValueError(f"Node '{nid}' references unknown input '{dep}'")
                in_deg[nid] += 1
                children[dep].append(nid)
        queue = collections.deque(nid for nid, d in in_deg.items() if d == 0)
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

    def clone_with_replacement(self, source_id: str, tensor: torch.Tensor) -> "Graph":
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


def _validate_halo_contract(tensor: torch.Tensor, radius: int, dims: Optional[List[int]], op_name: str) -> None:
    if radius < 0:
        raise ValueError(f"{op_name}: invalid halo radius {radius}")
    if tensor.ndim == 0:
        return
    for d in (dims if dims is not None else range(tensor.ndim)):
        if tensor.shape[d] < 2 * radius + 1:
            raise ValueError(f"{op_name}: dim {d} size {tensor.shape[d]} too small for halo radius {radius}")


def _op_accepts_alloc(fn: Callable) -> bool:
    try:
        return "alloc" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False


class Runtime:
    def __init__(self, mm: MemoryManager, device: Union[str, torch.device] = "cpu", skip_pool=False) -> None:
        self.mm = mm
        _set_dist_mm(mm)
        self.device = torch.device(device)
        self._alloc_cache: Dict[str, bool] = {}
        self.skip_pool = skip_pool
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
        lock = getattr(self.mm, "_lock", None) or getattr(self.mm, "_live_lock", None)

        if live is not None:
            ctx = lock if lock is not None else contextlib.nullcontext()
            with ctx:
                dead_keys = []
                for k, state in list(live.items()):
                    tensor = state if isinstance(state, torch.Tensor) else getattr(state, "tensor", None)
                    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                        dead_keys.append(k)
                for k in dead_keys:
                    state = live.get(k)
                    if isinstance(state, torch.Tensor):
                        live.pop(k, None)
                    elif state is not None:
                        state.tensor = None
                        state.is_free = True
                        state.is_evicted = True

                pool_ptrs = getattr(self.mm, "_pool_ptrs", None)
                if pool_ptrs is not None:
                    pool_ptrs.clear()

                pending = getattr(self.mm, "_pending_writes", None)
                if pending is not None:
                    pending.clear()
                    cond = getattr(self.mm, "_pending_writes_cond", None)
                    if cond is not None:
                        cond.notify_all()

        clear_pool = getattr(self.mm, "clear_pool", None)
        if callable(clear_pool):
            clear_pool()
        _PINNED_POOL.clear()
        if self.device.type == "cuda" and torch.cuda.is_available():
            _clear_k_grid_cache()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        _dbg("flush_vram: done")

    def _auto_chunk_size(self, lazy: "LazyField", chunk_dim: int, safety_factor: float = 8.0) -> int:
        shape = lazy.shape
        slice_elements = math.prod(s for i, s in enumerate(shape) if i != chunk_dim)
        bytes_per_slice = slice_elements * 8
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            free = _vram_free_bytes(self.device)
            usable = int(free * (1.0 - _VRAM_HEADROOM_FRAC * 2))
        else:
            try:
                usable = int(_psutil.virtual_memory().available * 0.30)
            except Exception:
                usable = 1 * 1024**3
        chunk = max(1, int(usable / (safety_factor * bytes_per_slice)))
        result = min(chunk, shape[chunk_dim])
        _dbg(f"_auto_chunk_size: free={usable/1024**3:.2f}GB bytes/slice={bytes_per_slice/1024**2:.1f}MB chunk={result}")
        return result

    def run_chunked(
        self,
        graph: Graph,
        sink_id: str,
        chunk_dim: int = 0,
        chunk_size: Optional[int] = None,
        out_path: Optional[str] = None,
        spacing: Optional[Tuple] = None,
        origin: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        import queue as _queue

        lazy_node_ids: List[str] = []
        lazy_field: Optional[LazyField] = None
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

        n_chunks = math.ceil(total / chunk_size)

        idx0: List[Any] = [slice(None)] * len(lazy_field.shape)
        idx0[chunk_dim] = slice(0, min(chunk_size, total))
        arr0 = lazy_field[tuple(idx0)]
        t0 = torch.from_numpy(arr0.astype(np.float64)).to(_DTYPE).to(self.device)
        g0 = graph
        for nid in lazy_node_ids:
            g0 = g0.clone_with_replacement(nid, t0)
        with torch.no_grad():
            r0 = self.run(g0)
        c0 = r0[sink_id].cpu()
        out_shape = list(c0.shape)
        out_shape[chunk_dim] = total
        out_shape = tuple(out_shape)
        s_min = float(c0.min())
        s_max = float(c0.max())
        s_sum = float(c0.sum())
        s_n = c0.numel()
        del t0, g0, r0, arr0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        h5f = h5_dset = None
        write_dim = chunk_dim
        if out_path is not None:
            chunk_h5 = tuple(min(128, s) for s in c0.shape)
            h5f = h5py.File(out_path, "w")
            h5_dset = h5f.create_dataset(
                "field", shape=out_shape, dtype=np.float64,
                chunks=chunk_h5, compression="gzip", compression_opts=1,
            )
            sp = spacing or tuple(1.0 for _ in range(len(out_shape)))
            og = origin or tuple(0.0 for _ in range(len(out_shape)))
            h5f.create_dataset("spacing", data=np.array(sp, dtype=np.float64))
            h5f.create_dataset("origin", data=np.array(og, dtype=np.float64))

        WRITE_Q_SIZE = 2
        write_q: _queue.Queue = _queue.Queue(maxsize=WRITE_Q_SIZE)
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
                        sl = [slice(None)] * len(out_shape)
                        sl[write_dim] = slice(w_start, w_end)
                        h5_dset[tuple(sl)] = arr_cpu
                except Exception as e:
                    write_errors.append(e)
                finally:
                    del arr_cpu
                    write_q.task_done()

        writer = threading.Thread(target=_writer_thread, daemon=True, name="chunk-writer")
        writer.start()

        PREFETCH_SIZE = 1
        prefetch_q: _queue.Queue = _queue.Queue(maxsize=PREFETCH_SIZE)

        def _reader_thread(starts: List[int]) -> None:
            for st in starts:
                en = min(st + chunk_size, total)
                idxr = [slice(None)] * len(lazy_field.shape)
                idxr[chunk_dim] = slice(st, en)
                arr = lazy_field[tuple(idxr)]
                prefetch_q.put((st, en, arr))
            prefetch_q.put(None)

        chunk_starts = list(range(chunk_size, total, chunk_size))
        reader = threading.Thread(target=_reader_thread, args=(chunk_starts,), daemon=True, name="chunk-reader")
        reader.start()

        write_q.put((0, min(chunk_size, total), c0.numpy()))
        del c0

        pinned_bufs: List = []
        sentinel_seen = len(chunk_starts) == 0
        while not sentinel_seen:
            item = prefetch_q.get()
            if item is None:
                sentinel_seen = True
                break
            p_start, p_end, arr = item

            if self.device.type == "cuda" and self._h2d_stream is not None:
                gpu_t, pinned = _lazy_to_gpu(arr, self.device, self._h2d_stream)
                torch.cuda.current_stream(self.device).wait_stream(self._h2d_stream)
                if pinned is not None:
                    pinned_bufs.append(pinned)
            else:
                gpu_t = torch.from_numpy(arr.astype(np.float64, copy=False)).to(_DTYPE).to(self.device)
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
            s_n += cout.numel()

            cout_np = cout.cpu().numpy()
            if self.mm.owns(cout):
                self.mm.release(cout)
            del gpu_t, cg, cr, cout

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

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
                "min": s_min,
                "max": s_max,
                "mean": s_sum / max(s_n, 1),
                "out_path": out_path,
            }
        }

    def run(self, graph: Graph, seed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = dict(seed or {})
        ref_count: Dict[str, int] = self._ref_counts(graph)

        for node in graph.topological_sort():
            if node.id in results:
                continue

            inputs = []
            for i in node.input_ids:
                val = results[i]
                if isinstance(val, LazyField):
                    arr = val[:]
                    val = torch.from_numpy(arr.astype(np.float64, copy=False)).to(_DTYPE).to(self.device)
                inputs.append(val)

            kwargs = dict(node.params)
            for k, v in list(kwargs.items()):
                if isinstance(v, LazyField):
                    arr = v[:]
                    kwargs[k] = torch.from_numpy(arr.astype(np.float64, copy=False)).to(_DTYPE).to(self.device)

            tensor_inputs = [i for i in inputs if isinstance(i, torch.Tensor)]
            if tensor_inputs:
                cuda_devs = [t.device for t in tensor_inputs if t.device.type == "cuda"]
                target_dev = cuda_devs[0] if cuda_devs else tensor_inputs[0].device
                inputs = [i.to(target_dev) if isinstance(i, torch.Tensor) else i for i in inputs]
                for k, v in list(kwargs.items()):
                    if isinstance(v, torch.Tensor) and v.device != target_dev:
                        kwargs[k] = v.to(target_dev)

            meta = OP_METADATA.get(node.op_name, {})
            radius = meta.get("stencil_radius", 0)
            if radius > 0:
                first_t = next((i for i in inputs if isinstance(i, torch.Tensor)), None)
                static_dims = meta.get("exchange_dims")
                if first_t is not None:
                    dims = [d for d in static_dims if d < first_t.ndim] if static_dims is not None else list(range(first_t.ndim))
                else:
                    dims = static_dims
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        _validate_halo_contract(inp, radius, dims, node.op_name)
                inputs = [self.mm.halo_exchange(i, radius, dims) if isinstance(i, torch.Tensor) else i for i in inputs]

            op_fn = OP_REGISTRY[node.op_name]
            sig = inspect.signature(op_fn)
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
                _assert_fp64(output, node.op_name)
                is_managed = self.mm.owns(output)
                if is_managed and output._base is not None:
                    is_managed = False
                if not self.skip_pool and not is_managed and self.mm.should_manage(output.shape) and output.is_contiguous():
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
                        val = results.pop(dep)
                        if isinstance(val, torch.Tensor) and self.mm.owns(val):
                            self.mm.release(val)

        return results


# -----------------------------------------------------------------------------
# Tile iterator
# -----------------------------------------------------------------------------

def _tile_iter(
    data: torch.Tensor,
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
    tile_shape: Tuple[int, ...],
    overlap: int,
    ndim: int,
) -> Iterator[Tuple[Tuple, torch.Tensor, Tuple]]:
    spatial = data.shape[:ndim]
    ranges = [range(0, spatial[d], tile_shape[d]) for d in range(ndim)]
    for starts in itertools.product(*ranges):
        write_s, read_s, tile_origin = [], [], []
        for d, st in enumerate(starts):
            n = spatial[d]
            w_e = min(st + tile_shape[d], n)
            r_s = max(0, st - overlap)
            r_e = min(n, w_e + overlap)
            write_s.append(slice(st, w_e))
            read_s.append(slice(r_s, r_e))
            tile_origin.append(origin[d] + r_s * spacing[d])
        for _ in range(ndim, data.ndim):
            write_s.append(slice(None))
            read_s.append(slice(None))
        yield tuple(write_s), data[tuple(read_s)], tuple(tile_origin)


def _trim_slices(
    tile_shape: Tuple[int, ...],
    write_slices: Tuple,
    overlap: int,
    ndim: int,
    global_shape: Tuple[int, ...],
    read_tile_shape: Optional[Tuple[int, ...]] = None,
) -> List:
    trim = []
    for d in range(ndim):
        ws = write_slices[d]
        lo = overlap if ws.start > 0 else 0
        hi = overlap if ws.stop < global_shape[d] else 0
        n_t = read_tile_shape[d] if (read_tile_shape is not None) else tile_shape[d]
        trim.append(slice(lo, n_t - hi if hi else n_t))
    for _ in range(ndim, len(tile_shape)):
        trim.append(slice(None))
    return trim


# -----------------------------------------------------------------------------
# Operator registry
# -----------------------------------------------------------------------------

OP_REGISTRY: Dict[str, Callable] = {}
OP_METADATA: Dict[str, dict] = {}


def register_operator(
    name: str,
    func: Callable,
    radius: int = 0,
    halo_l: int = 0,
    halo_r: int = 0,
    cost: str = "low",
    exchange_dims: Optional[List[int]] = None,
) -> None:
    OP_REGISTRY[name] = func
    OP_METADATA[name] = {
        "stencil_radius": radius,
        "halo_left": halo_l if halo_l else radius,
        "halo_right": halo_r if halo_r else radius,
        "cost": cost,
        "exchange_dims": exchange_dims,
    }


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------

def _constant(*, value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(_DTYPE)
    raise TypeError(f"_constant: expected torch.Tensor, got {type(value).__name__}")


register_operator("_constant", _constant, cost="low")


def add(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "add(a)")
    _assert_fp64(b, "add(b)")
    if alloc is None:
        return torch.add(a, b)
    out = alloc(torch.broadcast_shapes(a.shape, b.shape), a.device, key=_uid("add"))
    return torch.add(a, b, out=out)


register_operator("add", add, cost="low")


def sub(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "sub(a)")
    _assert_fp64(b, "sub(b)")
    if alloc is None:
        return torch.sub(a, b)
    out = alloc(torch.broadcast_shapes(a.shape, b.shape), a.device, key=_uid("sub"))
    return torch.sub(a, b, out=out)


register_operator("sub", sub, cost="low")


def mul(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "mul(a)")
    _assert_fp64(b, "mul(b)")
    if alloc is None:
        return torch.mul(a, b)
    out = alloc(torch.broadcast_shapes(a.shape, b.shape), a.device, key=_uid("mul"))
    return torch.mul(a, b, out=out)


register_operator("mul", mul, cost="low")


def div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-15, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "div(a)")
    _assert_fp64(b, "div(b)")
    eps_t = torch.as_tensor(eps, dtype=b.dtype, device=b.device)
    b_safe = torch.where(
        b.abs() < eps_t,
        torch.where(b < 0, -eps_t, eps_t),
        b,
    )
    if alloc is None:
        return torch.div(a, b_safe)
    out = alloc(torch.broadcast_shapes(a.shape, b.shape), a.device, key=_uid("div"))
    return torch.div(a, b_safe, out=out)


register_operator("div", div, cost="low")


def neg(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "neg(a)")
    if alloc is None:
        return torch.neg(a)
    out = alloc(a.shape, a.device, key=_uid("neg"))
    return torch.neg(a, out=out)


register_operator("neg", neg, cost="low")


def clamp(a: torch.Tensor, lo: float = 0.0, hi: float = 1.0, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "clamp(a)")
    if alloc is None:
        return torch.clamp(a, lo, hi)
    out = alloc(a.shape, a.device, key=_uid("clamp"))
    return torch.clamp(a, lo, hi, out=out)


register_operator("clamp", clamp, cost="low")


def exp(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "exp(a)")
    if alloc is None:
        return torch.exp(a)
    out = alloc(a.shape, a.device, key=_uid("exp"))
    return torch.exp(a, out=out)


register_operator("exp", exp, cost="medium")


def log(a: torch.Tensor, eps: float = 1e-15, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "log(a)")
    if alloc is None:
        return torch.log(a.clamp(min=eps))
    out = alloc(a.shape, a.device, key=_uid("log"))
    return torch.log(a.clamp(min=eps), out=out)


register_operator("log", log, cost="medium")


def sqrt(a: torch.Tensor, eps: float = 0.0, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "sqrt(a)")
    if alloc is None:
        return torch.sqrt(a.clamp(min=eps))
    out = alloc(a.shape, a.device, key=_uid("sqrt"))
    return torch.sqrt(a.clamp(min=eps), out=out)


register_operator("sqrt", sqrt, cost="medium")


def sin(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "sin(a)")
    if alloc is None:
        return torch.sin(a)
    out = alloc(a.shape, a.device, key=_uid("sin"))
    return torch.sin(a, out=out)


register_operator("sin", sin, cost="medium")


def tanh(a: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "tanh(a)")
    if alloc is None:
        return torch.tanh(a)
    out = alloc(a.shape, a.device, key=_uid("tanh"))
    return torch.tanh(a, out=out)


register_operator("tanh", tanh, cost="medium")


def _pad1d(t: torch.Tensor, dim: int, r: int, bc: str) -> torch.Tensor:
    mode_map = {
        "neumann": "replicate",
        "dirichlet": "constant", 
        "periodic": "circular",
        "reflect": "reflect",
    }
    mode = mode_map.get(bc, "constant")
    value = 0.0 if mode == "constant" else 0.0
    
    pad = [0] * (2 * t.ndim)
    
    pad_idx = 2 * (t.ndim - 1 - dim)
    pad[pad_idx] = r
    pad[pad_idx + 1] = r
    
    try:
        return F.pad(t, pad, mode=mode, value=value)
    except RuntimeError:
        new_shape = list(t.shape)
        new_shape[dim] += 2 * r
        padded = torch.zeros(new_shape, dtype=t.dtype, device=t.device)
        
        src_slices = [slice(None)] * t.ndim
        dst_slices = [slice(None)] * t.ndim
        dst_slices[dim] = slice(r, -r) if r > 0 else slice(None)
        padded[tuple(dst_slices)] = t
        
        if mode == "replicate":
            if r > 0:
                left_src = [slice(None)] * t.ndim
                left_src[dim] = 0
                left_val = t[tuple(left_src)]
                
                for i in range(r):
                    left_dst = [slice(None)] * t.ndim
                    left_dst[dim] = i
                    padded[tuple(left_dst)] = left_val
                
                right_src = [slice(None)] * t.ndim
                right_src[dim] = -1
                right_val = t[tuple(right_src)]
                
                for i in range(r):
                    right_dst = [slice(None)] * t.ndim
                    right_dst[dim] = -r + i
                    padded[tuple(right_dst)] = right_val
        elif mode == "constant":
            pass
        elif mode == "circular":
            if r > 0:
                left_src = [slice(None)] * t.ndim
                left_src[dim] = slice(-r, None)
                left_data = t[tuple(left_src)]
                
                left_dst = [slice(None)] * t.ndim
                left_dst[dim] = slice(0, r)
                padded[tuple(left_dst)] = left_data
                
                right_src = [slice(None)] * t.ndim
                right_src[dim] = slice(0, r)
                right_data = t[tuple(right_src)]
                
                right_dst = [slice(None)] * t.ndim
                right_dst[dim] = slice(-r, None)
                padded[tuple(right_dst)] = right_data
        elif mode == "reflect":
            if r > 0 and t.shape[dim] > 1:
                for i in range(r):
                    src_idx = min(i + 1, t.shape[dim] - 1)
                    left_src = [slice(None)] * t.ndim
                    left_src[dim] = src_idx
                    left_val = t[tuple(left_src)]
                    
                    left_dst = [slice(None)] * t.ndim
                    left_dst[dim] = r - 1 - i
                    padded[tuple(left_dst)] = left_val
                
                for i in range(r):
                    src_idx = max(t.shape[dim] - 2 - i, 0)
                    right_src = [slice(None)] * t.ndim
                    right_src[dim] = src_idx
                    right_val = t[tuple(right_src)]
                    
                    right_dst = [slice(None)] * t.ndim
                    right_dst[dim] = -r + i
                    padded[tuple(right_dst)] = right_val
        
        return padded

def gradient(t: torch.Tensor, dim: int = 0, dx: float = 1.0, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "gradient")
    
    is_vector_field = t.ndim >= 2 and t.shape[-1] in (2, 3)
    
    if is_vector_field:
        components = []
        n_components = t.shape[-1]
        
        for i in range(n_components):
            comp = t[..., i]
            
            ext = _pad1d(comp, dim, 1, boundary)
            left = ext.narrow(dim, 0, comp.shape[dim])
            right = ext.narrow(dim, 2, comp.shape[dim])
            comp_grad = (right - left) * (0.5 / dx)
            components.append(comp_grad)
        
        out = torch.stack(components, dim=-1)
        
        if out.ndim == t.ndim:
            out = out.unsqueeze(-1)
    else:
        ext = _pad1d(t, dim, 1, boundary)
        left = ext.narrow(dim, 0, t.shape[dim])
        right = ext.narrow(dim, 2, t.shape[dim])
        out = (right - left) * (0.5 / dx)
    
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("grad"))
        buf.copy_(out)
        return buf
    return out.to(_DTYPE)


register_operator("gradient", gradient, radius=1, cost="medium", exchange_dims=None)


def divergence(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "divergence")
    ndim = t.shape[-1]
    dxs = (dx,) * ndim if isinstance(dx, float) else dx
    out = torch.zeros_like(t[..., 0])
    for d in range(ndim):
        comp = t[..., d]
        ext = _pad1d(comp, d, 1, boundary)
        left = ext.narrow(d, 0, comp.shape[d])
        right = ext.narrow(d, 2, comp.shape[d])
        out.add_((right - left), alpha=0.5 / dxs[d])
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("div"))
        buf.copy_(out)
        return buf
    return out.to(_DTYPE)


register_operator("divergence", divergence, radius=1, cost="medium", exchange_dims=None)


@triton.jit
def laplacian_3d_kernel(
    x_ptr, out_ptr,
    nx, ny, nz,
    dx2, dy2, dz2,
    periodic: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_Z: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    off_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    off_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    off_z = pid_z * BLOCK_Z + tl.arange(0, BLOCK_Z)
    mask = (off_x < nx) & (off_y < ny) & (off_z < nz)

    def idx(x, y, z):
        return z * (nx * ny) + y * nx + x

    if periodic:
        xm = (off_x - 1) % nx
        xp = (off_x + 1) % nx
        ym = (off_y - 1) % ny
        yp = (off_y + 1) % ny
        zm = (off_z - 1) % nz
        zp = (off_z + 1) % nz
    else:
        xm = tl.maximum(off_x - 1, 0)
        xp = tl.minimum(off_x + 1, nx - 1)
        ym = tl.maximum(off_y - 1, 0)
        yp = tl.minimum(off_y + 1, ny - 1)
        zm = tl.maximum(off_z - 1, 0)
        zp = tl.minimum(off_z + 1, nz - 1)

    c = tl.load(x_ptr + idx(off_x, off_y, off_z), mask=mask, other=0.0)
    fxm = tl.load(x_ptr + idx(xm, off_y, off_z), mask=mask, other=0.0)
    fxp = tl.load(x_ptr + idx(xp, off_y, off_z), mask=mask, other=0.0)
    fym = tl.load(x_ptr + idx(off_x, ym, off_z), mask=mask, other=0.0)
    fyp = tl.load(x_ptr + idx(off_x, yp, off_z), mask=mask, other=0.0)
    fzm = tl.load(x_ptr + idx(off_x, off_y, zm), mask=mask, other=0.0)
    fzp = tl.load(x_ptr + idx(off_x, off_y, zp), mask=mask, other=0.0)

    lap = ((fxp + fxm - 2.0 * c) / dx2 +
           (fyp + fym - 2.0 * c) / dy2 +
           (fzp + fzm - 2.0 * c) / dz2)
    tl.store(out_ptr + idx(off_x, off_y, off_z), lap, mask=mask)


class Laplacian:
    def __init__(self, boundary: str = "periodic") -> None:
        self.boundary = boundary

    def _normalize_dx(self, dx: Any, D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(dx, (int, float)):
            return torch.full((D,), float(dx), device=device, dtype=dtype)
        dx = torch.tensor(dx, device=device, dtype=dtype)
        if dx.numel() != D:
            raise ValueError(f"dx must have length {D}, got {dx.numel()}")
        return dx

    def __call__(self, x: torch.Tensor, dx: Any = 1.0) -> torch.Tensor:
        D = x.ndim
        dv = self._normalize_dx(dx, D, x.device, x.dtype)

        if x.is_cuda and D == 3 and self.boundary == "periodic":
            try:
                x_c = x.contiguous()
                out = torch.empty_like(x_c)
                nx, ny, nz = x_c.shape
                if nx < 1 or ny < 1 or nz < 1:
                    raise ValueError("Empty dimension")
                BLOCK_X, BLOCK_Y, BLOCK_Z = 8, 8, 8
                grid = (
                    triton.cdiv(nx, BLOCK_X),
                    triton.cdiv(ny, BLOCK_Y),
                    triton.cdiv(nz, BLOCK_Z),
                )
                laplacian_3d_kernel[grid](
                    x_c, out,
                    nx, ny, nz,
                    float(dv[0] * dv[0]),
                    float(dv[1] * dv[1]),
                    float(dv[2] * dv[2]),
                    True,  
                    BLOCK_X=BLOCK_X, BLOCK_Y=BLOCK_Y, BLOCK_Z=BLOCK_Z,
                )
                torch.cuda.synchronize()   
                return out
            except Exception:
                pass

        out = torch.zeros_like(x)
        for d in range(D):
            xp = torch.roll(x, shifts=-1, dims=d)
            xm = torch.roll(x, shifts=1, dims=d)
            if self.boundary != "periodic":
                slc0 = [slice(None)] * D
                slc0[d] = 0
                slc1 = [slice(None)] * D
                slc1[d] = -1
                xp[tuple(slc1)] = x[tuple(slc1)]
                xm[tuple(slc0)] = x[tuple(slc0)]
            out.add_((xp + xm - 2.0 * x) / (dv[d] * dv[d]))
        return out


def laplacian(x: torch.Tensor, dx: Any = 1.0, boundary: str = "periodic") -> torch.Tensor:
    return Laplacian(boundary)(x, dx)


register_operator("laplacian", laplacian, radius=1, cost="high", exchange_dims=None)


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


def op_sum(t: torch.Tensor, dim=None, keepdim: bool = False, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "sum")
    local = t.sum(dim=dim, keepdim=keepdim)
    return _dist_all_reduce(local) if dim is None else local


register_operator("sum", op_sum, cost="low")


def mean(t: torch.Tensor, dim=None, keepdim: bool = False, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "mean")
    if dim is None and is_distributed():
        local_sum = t.sum()
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


def moving_average(t: torch.Tensor, window: int = 3, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "moving_average")
    if t.ndim > 3:
        raise NotImplementedError("moving_average: only 1-D/2-D/3-D supported")
    pad = window // 2
    weight = torch.full((1, 1, window), 1.0 / window, dtype=_DTYPE, device=t.device)
    out = t
    for d in range(t.ndim):
        p = _pad1d(out, d, pad, boundary)
        perm = list(range(out.ndim))
        perm[d], perm[-1] = perm[-1], perm[d]
        x = p.permute(perm).contiguous()
        B = math.prod(x.shape[:-1]) if x.ndim > 1 else 1
        y = F.conv1d(x.reshape(B, 1, x.shape[-1]), weight)
        inv = [0] * out.ndim
        for s, dd in enumerate(perm):
            inv[dd] = s
        out = y.reshape(*x.shape[:-1], out.shape[d]).permute(inv).contiguous()
    return out.to(_DTYPE)


register_operator("moving_average", moving_average, cost="medium")


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
    return torch.linalg.det(t).to(_DTYPE)


register_operator("determinant", determinant, cost="high")


def fft(t: torch.Tensor, norm: str = "ortho", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "fft")
    return torch.fft.fftn(t, norm=norm).real.to(_DTYPE)


register_operator("fft", fft, cost="high")


def ifft(t: torch.Tensor, norm: str = "ortho", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "ifft")
    return torch.fft.ifftn(t, norm=norm).real.to(_DTYPE)


register_operator("ifft", ifft, cost="high")


def gradient_nd(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "gradient_nd")
    ndim = t.ndim
    dxs = (dx,) * ndim if isinstance(dx, float) else tuple(dx)
    comps = []
    for d in range(ndim):
        ext = _pad1d(t, d, 1, boundary)
        left = ext.narrow(d, 0, t.shape[d])
        right = ext.narrow(d, 2, t.shape[d])
        comps.append((right - left) * (0.5 / dxs[d]))
    return torch.stack(comps, dim=-1)


register_operator("gradient_nd", gradient_nd, radius=1, cost="medium", exchange_dims=None)


def curl(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "curl")
    if t.ndim < 2 or t.shape[-1] != 3:
        raise ValueError(f"curl: need (*spatial, 3), got {tuple(t.shape)}")
    dxs = (dx,) * 3 if isinstance(dx, float) else tuple(dx)

    def _g(ci: int, dim: int) -> torch.Tensor:
        comp = t[..., ci].contiguous()
        ext = _pad1d(comp, dim, 1, boundary)
        left = ext.narrow(dim, 0, comp.shape[dim])
        right = ext.narrow(dim, 2, comp.shape[dim])
        return (right - left) * (0.5 / dxs[dim])

    return torch.stack([_g(2, 1) - _g(1, 2), _g(0, 2) - _g(2, 0), _g(1, 0) - _g(0, 1)], dim=-1).to(_DTYPE)


register_operator("curl", curl, radius=1, cost="high", exchange_dims=None)


def material_derivative(f: torch.Tensor, velocity: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(f, "material_derivative")
    _assert_fp64(velocity, "material_derivative(v)")
    ndim = velocity.shape[-1]
    dxs = (dx,) * ndim if isinstance(dx, float) else dx
    out = torch.zeros_like(f)
    for d in range(ndim):
        out.add_(velocity[..., d] * gradient(f, dim=d, dx=dxs[d], boundary=boundary))
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("matderiv"))
        buf.copy_(out)
        return buf
    return out.to(_DTYPE)


register_operator("material_derivative", material_derivative, radius=1, cost="medium", exchange_dims=None)


def hessian(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "hessian")
    g = gradient_nd(t, dx=dx, boundary=boundary)
    rows = [gradient_nd(g[..., i].contiguous(), dx=dx, boundary=boundary) for i in range(t.ndim)]
    return torch.stack(rows, dim=-2).to(_DTYPE)


register_operator("hessian", hessian, radius=2, cost="high", exchange_dims=None)


def _simpson_1d(y: torch.Tensor, dx: float, dim: int) -> torch.Tensor:
    n = y.shape[dim]
    if n < 3:
        return torch.trapezoid(y, dx=dx, dim=dim)
    if n % 2 == 0:
        fst  = y.select(dim, 0)
        mid  = y.select(dim, n - 2)   
        lst  = y.select(dim, n - 1)
        odd_idx = torch.arange(1, n - 2, 2, device=y.device)
        evn_idx = torch.arange(2, n - 2, 2, device=y.device)
        odd = y.index_select(dim, odd_idx).sum(dim) if odd_idx.numel() > 0 else torch.zeros_like(fst)
        evn = y.index_select(dim, evn_idx).sum(dim) if evn_idx.numel() > 0 else torch.zeros_like(fst)
        simp = (dx / 3.0) * (fst + mid + 4.0 * odd + 2.0 * evn)
        trap = (dx * 0.5) * (mid + lst)
        return simp + trap
    odd_idx = torch.arange(1, n - 1, 2, device=y.device)
    evn_idx = torch.arange(2, n - 1, 2, device=y.device)
    fst = y.select(dim, 0)
    lst = y.select(dim, n - 1)
    odd = y.index_select(dim, odd_idx).sum(dim)
    evn = y.index_select(dim, evn_idx).sum(dim) if evn_idx.numel() > 0 else torch.zeros_like(fst)
    return (dx / 3.0) * (fst + lst + 4.0 * odd + 2.0 * evn)


def integrate(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, dims: Optional[Union[int, Tuple[int, ...]]] = None, method: str = "simpson", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "integrate")
    ndim = t.ndim
    dxs = (dx,) * ndim if isinstance(dx, float) else tuple(dx)
    dim_seq = tuple(range(ndim)) if dims is None else ((dims,) if isinstance(dims, int) else tuple(dims))
    result = t
    for d in sorted(dim_seq, reverse=True):
        result = _simpson_1d(result, dxs[d], d) if method == "simpson" else torch.trapezoid(result, dx=dxs[d], dim=d)
    return result.to(_DTYPE)


register_operator("integrate", integrate, cost="medium")


def cumulative_integral(t: torch.Tensor, dx: float = 1.0, dim: int = 0, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "cumulative_integral")
    n = t.shape[dim]
    if n < 2:
        return torch.zeros_like(t)

    def _sl(a, b):
        s = [slice(None)] * t.ndim
        s[dim] = slice(a, b)
        return tuple(s)

    trapz = (t[_sl(None, -1)] + t[_sl(1, None)]) * (dx * 0.5)
    zero_shape = list(t.shape)
    zero_shape[dim] = 1
    zero = torch.zeros(zero_shape, dtype=_DTYPE, device=t.device)
    return torch.cat([zero, torch.cumsum(trapz, dim=dim)], dim=dim).to(_DTYPE)


register_operator("cumulative_integral", cumulative_integral, cost="low")


def surface_integral(t: torch.Tensor, normals: torch.Tensor, area_weights: torch.Tensor, alloc=None) -> torch.Tensor:
    for x, n in ((t, "t"), (normals, "normals"), (area_weights, "area_weights")):
        _assert_fp64(x, f"surface_integral({n})")
    return ((t * normals).sum(dim=-1) * area_weights).sum().to(_DTYPE)


register_operator("surface_integral", surface_integral, cost="medium")


def variance(t: torch.Tensor, dim=None, unbiased: bool = True, keepdim: bool = False, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "variance")
    return torch.var(t, dim=dim, unbiased=unbiased, keepdim=keepdim).to(_DTYPE)


register_operator("variance", variance, cost="low")


def covariance(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "covariance(a)")
    _assert_fp64(b, "covariance(b)")
    if a.shape != b.shape:
        raise ValueError(f"covariance: shape mismatch {a.shape} vs {b.shape}")
    af = (a - a.mean()).flatten()
    bf = (b - b.mean()).flatten()
    return (torch.dot(af, bf) / max(af.numel() - 1, 1)).to(_DTYPE)


register_operator("covariance", covariance, cost="medium")


def correlation(a: torch.Tensor, b: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(a, "correlation(a)")
    _assert_fp64(b, "correlation(b)")
    if a.shape != b.shape:
        raise ValueError(f"correlation: shape mismatch {a.shape} vs {b.shape}")
    af = (a - a.mean()).flatten()
    bf = (b - b.mean()).flatten()
    return (torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-15)).to(_DTYPE)


register_operator("correlation", correlation, cost="medium")


def entropy(t: torch.Tensor, dim=None, eps: float = 1e-15, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "entropy")
    p = t.abs()
    if dim is None:
        p = p / (p.sum() + eps)
        return -(p * (p + eps).log()).sum().to(_DTYPE)
    p = p / (p.sum(dim=dim, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(dim=dim).to(_DTYPE)


register_operator("entropy", entropy, cost="medium")


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


def deviatoric(t: torch.Tensor, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "deviatoric")
    if t.ndim < 2 or t.shape[-1] != t.shape[-2]:
        raise ValueError(f"deviatoric: need (*spatial, D, D), got {tuple(t.shape)}")
    D = t.shape[-1]
    tr = torch.diagonal(t, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    I = torch.eye(D, dtype=_DTYPE, device=t.device)
    return (t - (tr / D) * I).to(_DTYPE)


register_operator("deviatoric", deviatoric, cost="medium")


# -----------------------------------------------------------------------------
# Spectral k‑grid cache
# -----------------------------------------------------------------------------

@dataclass
class _KGridEntry:
    grids: List[torch.Tensor]
    k2: torch.Tensor


_K_GRID_CACHE: Dict[Tuple, _KGridEntry] = {}


def _k_grid_cached(shape: Tuple[int, ...], spacing: Tuple[float, ...], device_str: str) -> _KGridEntry:
    key = (shape, spacing, device_str)
    if key not in _K_GRID_CACHE:
        device = torch.device(device_str)
        grids = []
        for d, (n, dx) in enumerate(zip(shape, spacing)):
            k = torch.fft.fftfreq(n, d=dx, device=device).to(_DTYPE) * (2 * math.pi)
            sh = [1] * len(shape)
            sh[d] = -1
            grids.append(k.view(sh))
        k2 = sum(kg**2 for kg in grids)
        _K_GRID_CACHE[key] = _KGridEntry(grids=grids, k2=k2)
    return _K_GRID_CACHE[key]


def _clear_k_grid_cache() -> None:
    _K_GRID_CACHE.clear()


def _k_grid(shape: Tuple[int, ...], spacing: Tuple[float, ...], device: torch.device) -> _KGridEntry:
    return _k_grid_cached(shape, spacing, str(device))


def spectral_gradient(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, dim: int = 0, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "spectral_gradient")
    ndim = t.ndim
    dxs = (dx,) * ndim if isinstance(dx, float) else tuple(dx)
    entry = _k_grid(t.shape, dxs, t.device)
    T_hat = torch.fft.fftn(t)
    dT_hat = 1j * entry.grids[dim] * T_hat
    return torch.fft.ifftn(dT_hat).real.to(_DTYPE)


register_operator("spectral_gradient", spectral_gradient, cost="high")


def spectral_laplacian(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "spectral_laplacian")
    ndim = t.ndim
    dxs = (dx,) * ndim if isinstance(dx, float) else tuple(dx)
    entry = _k_grid(t.shape, dxs, t.device)
    T_hat = torch.fft.fftn(t)
    return torch.fft.ifftn(-entry.k2 * T_hat).real.to(_DTYPE)


register_operator("spectral_laplacian", spectral_laplacian, cost="high")


def surface_normals(height: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, dy: Optional[float] = None, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(height, "surface_normals")
    if isinstance(dx, (tuple, list)):
        dx_val, dy_val = float(dx[0]), float(dx[1]) if len(dx) > 1 else float(dx[0])
    else:
        dx_val = float(dx)
        dy_val = float(dy) if dy is not None else float(dx)
    gx = gradient(height, dim=0, dx=dx_val, boundary=boundary)
    gy = gradient(height, dim=1, dx=dy_val, boundary=boundary)
    n = torch.stack((-gx, -gy, torch.ones_like(height)), dim=-1)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-15)
    if alloc:
        buf = alloc(n.shape, n.device, key=_uid("normals"))
        buf.copy_(n)
        return buf
    return n.to(_DTYPE)


register_operator("surface_normals", surface_normals, radius=1, cost="medium", exchange_dims=None)


def mean_curvature(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, boundary: str = "neumann", alloc=None) -> torch.Tensor:
    _assert_fp64(t, "mean_curvature")
    ndim = t.ndim
    dxs = (dx,) * ndim if isinstance(dx, float) else tuple(dx)
    comps = []
    for d in range(ndim):
        ext = _pad1d(t, d, 1, boundary)
        left = ext.narrow(d, 0, t.shape[d])
        right = ext.narrow(d, 2, t.shape[d])
        comps.append((right - left) * (0.5 / dxs[d]))
    grad = torch.stack(comps, dim=-1)
    mag = torch.linalg.norm(grad, dim=-1, keepdim=True).clamp(min=1e-15)
    n_hat = grad / mag
    out = torch.zeros_like(t)
    for d in range(ndim):
        comp = n_hat[..., d].contiguous()
        ext = _pad1d(comp, d, 1, boundary)
        left = ext.narrow(d, 0, t.shape[d])
        right = ext.narrow(d, 2, t.shape[d])
        out.add_((right - left) * (0.5 / dxs[d]))
    if alloc:
        buf = alloc(out.shape, out.device, key=_uid("mean_curv"))
        buf.copy_(out)
        return buf
    return out.to(_DTYPE)


register_operator("mean_curvature", mean_curvature, radius=2, cost="high", exchange_dims=None)


def distance_transform(t: torch.Tensor, dx: Union[float, Tuple[float, ...]] = 1.0, alloc=None) -> torch.Tensor:
    _assert_fp64(t, "distance_transform")
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("distance_transform requires scipy: pip install scipy")
    ndim = t.ndim
    dxs = (dx,) * ndim if isinstance(dx, float) else tuple(dx)
    arr = (t == 0).cpu().numpy()
    out = distance_transform_edt(arr, sampling=list(dxs))
    return torch.from_numpy(out.astype(np.float64)).to(t.device).to(_DTYPE)


register_operator("distance_transform", distance_transform, cost="very_high")


# -----------------------------------------------------------------------------
# LazyField and I/O
# -----------------------------------------------------------------------------

_SPACING_HINTS = [
    "spacing", "dx", "dy", "dz", "delta_x", "delta_y", "delta_z",
    "resolution", "grid_spacing", "cell_size", "h",
]


class LazyField:

    def __init__(self, path: str, dataset_name: str = "field") -> None:
        self.path = path
        self.dataset_name = dataset_name
        self._lock = threading.RLock()
        self._local = threading.local()  # Each thread gets own file handle

    def _ensure_open(self) -> None:
        if not hasattr(self._local, 'file') or self._local.file is None:
            self._local.file = h5py.File(self.path, "r")
            self._local.dset = None
            
            # Find dataset
            for cand in (self.dataset_name, "data", "field"):
                if cand in self._local.file and isinstance(self._local.file[cand], h5py.Dataset):
                    self._local.dset = self._local.file[cand]
                    break
            
            if self._local.dset is None:
                for k in self._local.file:
                    if isinstance(self._local.file[k], h5py.Dataset):
                        self._local.dset = self._local.file[k]
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
        if hasattr(self._local, 'file') and self._local.file is not None:
            try:
                self._local.file.close()
            except Exception:
                pass
            finally:
                self._local.file = None
                self._local.dset = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        try:
            sh = self.shape
        except Exception:
            sh = "?"
        return f"LazyField(path={self.path}, shape={sh}, fp64)"



def _open_hdf5_field(path: str) -> Tuple[h5py.Dataset, Tuple[int, ...]]:
    f = h5py.File(path, "r")
    for cand in ("data", "field"):
        if cand in f and isinstance(f[cand], h5py.Dataset):
            ds = f[cand]
            return ds, tuple(ds.shape)
    for k in f:
        if isinstance(f[k], h5py.Dataset):
            ds = f[k]
            return ds, tuple(ds.shape)
    raise KeyError(f"No array dataset found in {path}")


def load_tensor(
    path: str,
    device: Union[str, torch.device] = "cpu",
    normalize: bool = False,
    return_mode: str = "lazy",
    max_eager_gb: float = 0.5,
    **nkw,
) -> Tuple[Union[LazyField, torch.Tensor], Tuple[float, ...], Tuple[float, ...]]:
    if normalize:
        path = _normalize_hdf5(path, **nkw)

    dev = torch.device(device)

    with h5py.File(path, "r") as f:
        for cand in ("field", "data"):
            if cand in f and isinstance(f[cand], h5py.Dataset):
                dset = f[cand]
                break
        else:
            raise KeyError(f"No field dataset found in {path}")
        if "spacing" in f:
            spacing = tuple(float(v) for v in f["spacing"][:])
        elif "spacing" in f.attrs:
            raw_spacing = f.attrs["spacing"]
            spacing = tuple(float(v) for v in np.atleast_1d(raw_spacing))
        elif "spacing" in dset.attrs:
            raw_spacing = dset.attrs["spacing"]
            spacing = tuple(float(v) for v in np.atleast_1d(raw_spacing))
        else:
            spacing = (1.0,) * max(1, dset.ndim)

        if "origin" in f:
            origin = tuple(float(v) for v in f["origin"][:])
        elif "origin" in f.attrs:
            raw_origin = f.attrs["origin"]
            origin = tuple(float(v) for v in np.atleast_1d(raw_origin))
        elif "origin" in dset.attrs:
            raw_origin = dset.attrs["origin"]
            origin = tuple(float(v) for v in np.atleast_1d(raw_origin))
        else:
            origin = tuple(0.0 for _ in spacing)
        shape = dset.shape
        total_bytes = np.prod(shape) * 8
        total_gb = total_bytes / 1024**3

    if return_mode == "auto":
        return_mode = "eager" if total_gb <= max_eager_gb else "lazy"

    if return_mode == "lazy":
        return LazyField(path), spacing, origin

    if return_mode == "eager":
        if total_gb > max_eager_gb:
            raise MemoryError(f"Refusing to eagerly load {total_gb:.2f} GB (limit={max_eager_gb} GB). Use return_mode='lazy'.")
        with h5py.File(path, "r") as f:
            for cand in ("field", "data"):
                if cand in f:
                    arr = f[cand][:]
                    break
        tensor = torch.from_numpy(arr).to(_DTYPE)
        if dev.type == "cuda":
            if torch.cuda.is_available():
                try:
                    tensor = tensor.pin_memory().to(dev, non_blocking=True)
                    torch.cuda.synchronize()
                except Exception:
                    tensor = tensor.to(dev)
            else:
                tensor = tensor.to(dev)
        return tensor, spacing, origin

    raise ValueError(f"Invalid return_mode: {return_mode!r}")


def save_tensor(tensor: torch.Tensor, spacing: Tuple[float, ...], origin: Tuple[float, ...], path: str, chunks: bool = True) -> None:
    _assert_fp64(tensor, "save_tensor")
    arr = tensor.cpu().numpy()
    chunk = tuple(min(128, s) for s in arr.shape) if chunks else None
    with h5py.File(path, "w") as f:
        f.create_dataset("field", data=arr, chunks=chunk, compression="gzip" if chunks else None)
        f.create_dataset("spacing", data=np.array(spacing, dtype=np.float64))
        f.create_dataset("origin", data=np.array(origin, dtype=np.float64))


def _normalize_hdf5(path: str, field_key: str = "field", spacing_key: str = "spacing", origin_key: str = "origin", spacing_value: Optional[Tuple[float, ...]] = None, output_path: Optional[str] = None) -> str:
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
        data = fi[field_key][:]
        chunk = tuple(min(128, s) for s in data.shape)
        fo.create_dataset("field", data=data, chunks=chunk, compression="gzip")

        spacing = None
        if spacing_key in fi:
            spacing = fi[spacing_key][:].tolist()
        else:
            for src in (fi[field_key].attrs, fi.attrs):
                for hint in _SPACING_HINTS:
                    if hint in src:
                        v = src[hint]
                        spacing = [float(v)] * data.ndim if np.isscalar(v) else list(v)
                        break
                if spacing:
                    break
        if spacing is None and spacing_value:
            spacing = list(spacing_value)
        if spacing is None:
            spacing = [1.0] * data.ndim
            warnings.warn(f"_normalize_hdf5: spacing not found in {path}, defaulting to 1.0")
        fo.create_dataset("spacing", data=np.array(spacing, dtype=np.float64))

        origin = fi[origin_key][:].tolist() if origin_key in fi else list(fi.attrs.get("origin", [0.0] * data.ndim))
        fo.create_dataset("origin", data=np.array(origin, dtype=np.float64))

    return output_path


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_output(op_name: str, in_tensor: torch.Tensor, out_tensor: torch.Tensor, **params) -> None:
    if _VALIDATION_LEVEL == "none":
        return
    if torch.isnan(out_tensor).any() or torch.isinf(out_tensor).any():
        raise ValueError(f"{op_name}: output contains NaN/Inf")
    if _VALIDATION_LEVEL == "sample":
        flat = out_tensor.flatten()
        n = max(1, min(int(flat.numel() * _VALIDATION_SAMPLE), flat.numel()))
        idx = torch.randint(0, flat.numel(), (n,), device=flat.device)
        out_n = flat[idx].norm().item()
        in_f = in_tensor.flatten()
        in_n = in_f[idx].norm().item() if in_f.numel() >= flat.numel() else in_f.norm().item()
    else:
        out_n = out_tensor.norm().item()
        in_n = in_tensor.norm().item()
    if in_n > 0 and out_n / (in_n + 1e-15) > 1e6:
        warnings.warn(f"{op_name}: output norm {out_n / in_n:.2e}× input — possible blow-up")
    if _VALIDATION_LEVEL == "full":
        order = params.get("order", 2)
        dx = min(params.get("spacing", [1.0])) if "spacing" in params else 1.0
        trunc = (dx**order) * in_n
        if out_n > 0 and trunc / (out_n + 1e-15) > 0.1:
            warnings.warn(f"{op_name}: truncation error may dominate (dx^{order} ≈ {trunc:.2e}, |out| ≈ {out_n:.2e})")


def _infer_output_shape(op_name: str, in_shape: Tuple, ndim: int) -> Optional[Tuple]:
    return tuple(in_shape)


# -----------------------------------------------------------------------------
# Distributed support
# -----------------------------------------------------------------------------

_DIST_INITIALIZED = False
_DIST_RANK = 0
_DIST_WORLD_SIZE = 1
_DIST_CART_TOPOLOGY: Optional[Dict] = None


def dist_init(backend: str = "nccl" if torch.cuda.is_available() else "gloo", dims: Optional[List[int]] = None) -> bool:
    global _DIST_INITIALIZED, _DIST_RANK, _DIST_WORLD_SIZE, _DIST_CART_TOPOLOGY
    if _DIST_INITIALIZED:
        return True
    if not dist.is_available():
        warnings.warn("torch.distributed not available – multi-node disabled")
        return False
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend=backend)
            _DIST_RANK = dist.get_rank()
            _DIST_WORLD_SIZE = dist.get_world_size()
            if dims is not None:
                _DIST_CART_TOPOLOGY = _create_cartesian_topology(dims, [True] * len(dims))
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
    rank_to_coord: Dict[int, Tuple] = {}
    coord_to_rank: Dict[Tuple, int] = {}
    for r in range(world):
        coords, tmp = [], r
        for d in range(len(dims) - 1, -1, -1):
            coords.insert(0, tmp % dims[d])
            tmp //= dims[d]
        coords = tuple(coords)
        rank_to_coord[r] = coords
        coord_to_rank[coords] = r
    neighbors: Dict[int, Dict] = {}
    for r, coord in rank_to_coord.items():
        neigh: Dict = {}
        for d in range(len(dims)):
            for direction, delta in ((-1, -1), (1, 1)):
                nc = list(coord)
                nc[d] += delta
                if 0 <= nc[d] < dims[d]:
                    neigh[(direction, d)] = coord_to_rank[tuple(nc)]
                elif periods[d]:
                    nc[d] %= dims[d]
                    neigh[(direction, d)] = coord_to_rank[tuple(nc)]
        neighbors[r] = neigh
    return {"dims": dims, "periods": periods, "rank_to_coord": rank_to_coord, "neighbors": neighbors}


def dist_rank() -> int:
    return _DIST_RANK if _DIST_INITIALIZED else 0


def dist_size() -> int:
    return _DIST_WORLD_SIZE if _DIST_INITIALIZED else 1


def is_distributed() -> bool:
    return _DIST_INITIALIZED and _DIST_WORLD_SIZE > 1


def _halo_exchange_nd(shard: torch.Tensor, radius: int, dims: List[int]) -> torch.Tensor:
    if not is_distributed() or radius <= 0 or _DIST_CART_TOPOLOGY is None:
        return shard
    rank = dist_rank()
    neighbors = _DIST_CART_TOPOLOGY["neighbors"][rank]
    reqs = []
    recv_bufs = []
    for (direction, d), neigh_rank in neighbors.items():
        if d not in dims:
            continue
        sl = [slice(None)] * shard.ndim
        if direction == -1:
            sl[d] = slice(radius, 2 * radius)
        else:
            sl[d] = slice(-2 * radius, -radius) if shard.shape[d] >= 2 * radius else slice(-radius, None)
        send_buf = shard[tuple(sl)].clone()
        recv_buf = torch.empty_like(send_buf)
        reqs += [dist.isend(send_buf, dst=neigh_rank), dist.irecv(recv_buf, src=neigh_rank)]
        recv_bufs.append((d, direction, recv_buf))
    for req in reqs:
        req.wait()
    for d, direction, buf in recv_bufs:
        sl = [slice(None)] * shard.ndim
        sl[d] = slice(0, radius) if direction == -1 else slice(-radius, None)
        shard[tuple(sl)] = buf
    return shard


def _halo_exchange_1d(shard: torch.Tensor, radius: int) -> torch.Tensor:
    if not is_distributed() or radius <= 0 or dist_size() == 1:
        return shard
    rank = dist_rank()
    world = dist_size()
    left_rank = rank - 1 if rank > 0 else None
    right_rank = rank + 1 if rank < world - 1 else None
    reqs = []

    def _buf(side: str):
        sl = [slice(None)] * shard.ndim
        sl[0] = slice(radius, 2 * radius) if side == "left" else slice(-2 * radius, -radius)
        return shard[tuple(sl)].clone() if shard.shape[0] >= 2 * radius else None

    send_l, send_r = _buf("left"), _buf("right")
    recv_l = torch.empty_like(send_l) if left_rank is not None and send_l is not None else None
    recv_r = torch.empty_like(send_r) if right_rank is not None and send_r is not None else None
    if left_rank is not None and send_l is not None:
        reqs += [dist.isend(send_l, dst=left_rank), dist.irecv(recv_l, src=left_rank)]
    if right_rank is not None and send_r is not None:
        reqs += [dist.isend(send_r, dst=right_rank), dist.irecv(recv_r, src=right_rank)]
    for req in reqs:
        req.wait()
    if recv_l is not None:
        shard[:radius] = recv_l
    if recv_r is not None:
        shard[-radius:] = recv_r
    return shard


class _DistMemoryManager(MemoryManager):
    def halo_exchange(self, shard: torch.Tensor, radius: int, dims: Optional[List[int]] = None) -> torch.Tensor:
        if dims is None:
            dims = [0]
        return _halo_exchange_nd(shard, radius, dims) if len(dims) > 1 and _DIST_CART_TOPOLOGY is not None else _halo_exchange_1d(shard, radius)

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        if not is_distributed():
            return t
        out = t.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


if dist.is_available():
    MemoryManager = _DistMemoryManager


def decompose_field(field: Field, dim: int = 0, overlap: int = 0) -> Tuple[Field, Tuple, Tuple[int, int]]:
    if not is_distributed():
        if overlap > 0:
            shape = list(field.shape)
            shape[dim] += 2 * overlap
            padded = torch.zeros(shape, dtype=_DTYPE, device=field.tensor.device)
            dst = [slice(None)] * field.ndim
            dst[dim] = slice(overlap, overlap + field.shape[dim])
            padded[tuple(dst)] = field.tensor
            new_origin = list(field.origin)
            new_origin[dim] -= overlap * field.spacing[dim]
            return Field(padded, field.spacing, tuple(new_origin), field._mm, field._key), (slice(None),), (0, field.shape[dim])
        return field, (slice(None),), (0, field.shape[dim])
    rank = dist_rank()
    world = dist_size()
    size = field.shape[dim]
    chunk = (size + world - 1) // world
    start = rank * chunk
    end = min(start + chunk, size)
    lo = max(0, start - overlap)
    hi = min(size, end + overlap)
    new_shape = list(field.shape)
    new_shape[dim] = hi - lo
    local_tensor = torch.zeros(new_shape, dtype=_DTYPE, device=field.tensor.device)
    src = [slice(None)] * field.ndim
    src[dim] = slice(lo, hi)
    dst = [slice(None)] * field.ndim
    dst[dim] = slice(overlap if start > 0 else 0, (overlap if start > 0 else 0) + (hi - lo))
    local_tensor[tuple(dst)] = field.tensor[tuple(src)]
    new_origin = list(field.origin)
    new_origin[dim] += lo * field.spacing[dim]
    return Field(local_tensor, field.spacing, tuple(new_origin), field._mm, field._key), tuple(src), (start, end)


def load_tensor_parallel(path: str, device: Union[str, torch.device] = "cpu", normalize: bool = False, **nkw) -> Tuple[torch.Tensor, Tuple[float, ...], Tuple[float, ...]]:
    if not is_distributed() or not _HAS_MPI:
        return load_tensor(path, device, normalize, **nkw)
    try:
        comm = _MPI.COMM_WORLD
        rank = comm.Get_rank()
        world = comm.Get_size()
        with h5py.File(path, "r", driver="mpio", comm=comm) as f:
            spacing = tuple(float(v) for v in f["spacing"][:])
            origin = tuple(float(v) for v in f["origin"][:]) if "origin" in f else tuple(0.0 for _ in spacing)
            dataset = f["field"]
            n = dataset.shape[0]
            chunk = (n + world - 1) // world
            lo, hi = rank * chunk, min(rank * chunk + chunk, n)
            data = np.empty((hi - lo,) + dataset.shape[1:], dtype=np.float64)
            if hi > lo:
                dataset.read_direct(data, source_sel=np.s_[lo:hi], dest_sel=np.s_[:])
        return torch.from_numpy(data).to(_DTYPE).to(torch.device(device)), spacing, origin
    except Exception as exc:
        warnings.warn(f"Parallel HDF5 failed ({exc}), falling back to serial")
        return load_tensor(path, device, normalize, **nkw)


# -----------------------------------------------------------------------------
# Execute helpers
# -----------------------------------------------------------------------------

def _execute_serial(
    op_name: str,
    input_path: str,
    out_path: str,
    device: str = "cpu",
    tile_shape: Optional[Tuple] = None,
    normalize: bool = False,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    boundary: str = "dirichlet",
    **params,
) -> None:
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unknown operator '{op_name}' (known: {list(OP_REGISTRY)})")
    mm = MemoryManager()
    rt = Runtime(mm, device=device)
    dev = torch.device(device)
    raw_t, spacing, origin = load_tensor(input_path, device=dev, normalize=normalize)
    _assert_fp64(raw_t, "execute/load")
    key_in = _uid("input")
    mm_input = mm.allocate(raw_t.shape, dev, key=key_in)
    mm_input.copy_(raw_t)
    del raw_t
    g = Graph()
    src_id = g.add("_constant", (), {"value": mm_input})
    full_params = {"dx": dx, "boundary": boundary, **params}
    sink_id = g.add(op_name, (src_id,), full_params)
    out_tensor = rt.run(g).get(sink_id)
    if isinstance(out_tensor, torch.Tensor):
        if not mm.owns(out_tensor):
            tmp = mm.allocate(out_tensor.shape, out_tensor.device)
            tmp.copy_(out_tensor)
            out_tensor = tmp
        _assert_fp64(out_tensor, f"execute/{op_name}")
        validate_output(op_name, mm_input, out_tensor, **full_params)
        save_tensor(out_tensor, spacing, origin, out_path)
        print(f"[ops] '{op_name}' fp64 on {device} → {out_path}")
        if mm.owns(out_tensor):
            mm.release(out_tensor)
    else:
        print(f"[ops] '{op_name}' → scalar: {out_tensor}")
    mm.release(mm_input)
    mm.clear_pool()


def execute_dist(
    op_name: str,
    input_path: str,
    out_path: str,
    device: str = "cpu",
    tile_shape: Optional[Tuple] = None,
    normalize: bool = False,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    boundary: str = "dirichlet",
    **params,
) -> None:
    if not is_distributed():
        return _execute_serial(op_name, input_path, out_path, device, tile_shape, normalize, dx, boundary, **params)
    raw_t, spacing, origin = load_tensor_parallel(input_path, device=device, normalize=normalize)
    _assert_fp64(raw_t, "execute_dist/load")
    mm = MemoryManager()
    rt = Runtime(mm, device=device)
    dev = torch.device(device)
    key_in = _uid("input")
    mm_input = mm.allocate(raw_t.shape, dev, key=key_in)
    mm_input.copy_(raw_t)
    del raw_t
    overlap = OP_METADATA.get(op_name, {}).get("stencil_radius", 0)
    field_in = Field(mm_input, spacing, origin, mm=mm, key=key_in)
    local_field, _, _ = decompose_field(field_in, dim=0, overlap=overlap)
    mm_input = local_field.tensor
    g = Graph()
    src_id = g.add("_constant", (), {"value": mm_input})
    full_params = {"dx": dx, "boundary": boundary, **params}
    sink_id = g.add(op_name, (src_id,), full_params)
    out_tensor = rt.run(g).get(sink_id)
    if isinstance(out_tensor, torch.Tensor) and not mm.owns(out_tensor):
        tmp = mm.allocate(out_tensor.shape, out_tensor.device)
        tmp.copy_(out_tensor)
        out_tensor = tmp
    if isinstance(out_tensor, torch.Tensor):
        _assert_fp64(out_tensor, f"execute_dist/{op_name}")
        out_valid = out_tensor[overlap:-overlap] if overlap > 0 and out_tensor.shape[0] > 2 * overlap else out_tensor
        in_valid = mm_input[overlap:-overlap] if overlap > 0 and mm_input.shape[0] > 2 * overlap else mm_input
        validate_output(op_name, in_valid, out_valid, **full_params)
        rank = dist_rank()
        world = dist_size()
        local_size = torch.tensor([out_valid.shape[0]], device=out_valid.device)
        sizes = [torch.zeros_like(local_size) for _ in range(world)]
        dist.all_gather(sizes, local_size)
        sizes = [int(s.item()) for s in sizes]
        max_size = max(sizes)
        padded = F.pad(out_valid, [0, max_size - out_valid.shape[0]] + [0, 0] * (out_valid.ndim - 1)) if out_valid.shape[0] < max_size else out_valid
        gathered = [torch.zeros_like(padded) for _ in range(world)]
        dist.all_gather(gathered, padded)
        if rank == 0:
            full_out = torch.cat([g[:sizes[i]] for i, g in enumerate(gathered)], dim=0)
            save_tensor(full_out, spacing, origin, out_path)
            mm.release(full_out)
        print(f"[ops] Distributed '{op_name}' rank {rank}/{world} done")
        if mm.owns(out_tensor):
            mm.release(out_tensor)
    else:
        print(f"[ops] Distributed '{op_name}' → scalar: {out_tensor}")
    mm.release(mm_input)
    mm.clear_pool()


def execute(*args, **kwargs) -> None:
    (execute_dist if is_distributed() else _execute_serial)(*args, **kwargs)


# -----------------------------------------------------------------------------
# Capability report
# -----------------------------------------------------------------------------

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


def validate_operator_identity() -> None:
    t = torch.randn(32, 32, dtype=_DTYPE)
    g = gradient_nd(t)
    lap_approx = sum(gradient_nd(g[..., i])[..., i] for i in range(t.ndim))
    lap = laplacian(t)
    err = (lap_approx - lap).abs().mean().item()
    if err > 1e-2:
        raise ValueError(f"Operator inconsistency: div(grad(f)) - laplacian(f) error = {err:.3e}")
    _dbg(f"validate_operator_identity: error = {err:.2e} ✓")


__all__ = [
    "Field",
    "MemoryManager",
    "Graph",
    "Runtime",
    "register_operator",
    "OP_REGISTRY",
    "OP_METADATA",
    "execute",
    "execute_dist",
    "dist_init",
    "is_distributed",
    "dist_rank",
    "dist_size",
    "decompose_field",
    "load_tensor",
    "save_tensor",
    "load_tensor_parallel",
    "capability_report",
    "validate_output",
    "validate_operator_identity",
    "use_advanced_mm",
    "LazyField",
    "_open_hdf5_field",
    "_clear_k_grid_cache",
    "_vram_headroom_ok",
    "_vram_free_bytes",
    "_PINNED_POOL",
]
