# memory_manager.py

from __future__ import annotations

import heapq
import math
import os
import queue
import threading
import atexit
import time
import uuid
import warnings
from collections import OrderedDict, deque
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import h5py

# ----------------------------------------------------------------------
# dependencies
# ----------------------------------------------------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    warnings.warn("psutil not installed; CPU memory pressure not monitored.", stacklevel=2)

_ENABLE_NUMA = os.environ.get("OPS_MM_ENABLE_NUMA", "0").lower() in ("1", "true", "yes")
if _ENABLE_NUMA:
    try:
        import numa as _numa_mod
        _HAS_NUMA = True
    except ImportError:
        _numa_mod = None
        _HAS_NUMA = False
        warnings.warn("NUMA requested but 'numa' module not found", stacklevel=2)
else:
    _numa_mod = None
    _HAS_NUMA = False

# ----------------------------------------------------------------------
# HDF5 compression backend: prefer LZ4, fallback to Zstd-1, then gzip-1
# ----------------------------------------------------------------------

try:
    import hdf5plugin as _hdf5plugin
    _COMPRESS_KWARGS = dict(**_hdf5plugin.LZ4())
    _COMPRESS_NAME = "lz4"
except Exception:
    try:
        import hdf5plugin as _hdf5plugin
        _COMPRESS_KWARGS = dict(**_hdf5plugin.Zstd(clevel=1))
        _COMPRESS_NAME = "zstd-1"
    except Exception:
        _COMPRESS_KWARGS = dict(compression="gzip", compression_opts=1)
        _COMPRESS_NAME = "gzip-1"

# ----------------------------------------------------------------------
# Configuration from environment
# ----------------------------------------------------------------------

CPU_PRESSURE_HIGH = float(os.environ.get("OPS_MM_CPU_PRESSURE_HIGH", "0.75"))
CPU_PRESSURE_LOW  = float(os.environ.get("OPS_MM_CPU_PRESSURE_LOW",  "0.65"))
GPU_FRACTION       = float(os.environ.get("OPS_MM_GPU_FRACTION",      "0.8"))
SPILL_DIR          = os.environ.get("OPS_MM_SPILL_DIR",               "./spill")
MAX_SPILL_BYTES    = int(os.environ.get("OPS_MM_MAX_SPILL_BYTES",     str(10 * 1024 ** 3)))
PREFETCH_DEPTH     = int(os.environ.get("OPS_MM_PREFETCH_DEPTH",      "2"))
EVICTION_POLICY    = os.environ.get("OPS_MM_EVICTION_POLICY",         "lru_freq")
MAX_LIVE_ENTRIES   = int(os.environ.get("OPS_MM_MAX_LIVE_ENTRIES",    "1000"))
BACKPRESSURE_QUEUE = int(os.environ.get("OPS_MM_BACKPRESSURE_QUEUE",  "1000"))
POOL_DEPTH         = int(os.environ.get("OPS_MM_POOL_DEPTH",          "8"))
TELEMETRY          = os.environ.get("OPS_MM_TELEMETRY", "1").lower() in ("1", "true", "yes")
DEBUG              = os.environ.get("OPS_DEBUG", "0") == "1"

# ----------------------------------------------------------------------
# Custom exceptions
# ----------------------------------------------------------------------
class DataLossError(RuntimeError):
    """Raised when a spill file is missing or corrupted on reload."""

# ----------------------------------------------------------------------
# BufferState: per-tensor metadata
# ----------------------------------------------------------------------
class BufferState:
    __slots__ = (
        "key", "shape", "dtype", "device", "size_bytes",
        "tensor", "is_pinned", "on_disk", "spill_generation",
        "last_access", "access_count", "is_free", "is_evicted",
        "copy_event", "load_failed", "last_modified",
    )

    def __init__(self, key: str, tensor: torch.Tensor, is_pinned: bool = False) -> None:
        self.key              = key
        self.shape            = tuple(tensor.shape)
        self.dtype            = tensor.dtype
        self.device           = tensor.device
        self.size_bytes       = tensor.numel() * tensor.element_size()
        self.tensor           = tensor
        self.is_pinned        = is_pinned
        self.on_disk          = False
        self.spill_generation = 0
        self.last_access      = time.monotonic()
        self.access_count     = 1
        self.is_free          = False
        self.is_evicted       = False
        self.copy_event: Optional[torch.cuda.Event] = None
        self.load_failed      = False
        self.last_modified    = time.monotonic()

# ----------------------------------------------------------------------
# _AsyncFuture
# ----------------------------------------------------------------------
class _AsyncFuture:
    __slots__ = ("_result", "_exception", "_done", "_cond")

    def __init__(self) -> None:
        self._result    = None
        self._exception: Optional[BaseException] = None
        self._done      = False
        self._cond      = threading.Condition()

    def set_result(self, result: Any) -> None:
        with self._cond:
            self._result = result
            self._done   = True
            self._cond.notify_all()

    def set_exception(self, exc: BaseException) -> None:
        with self._cond:
            self._exception = exc
            self._done      = True
            self._cond.notify_all()

    def result(self, timeout: Optional[float] = None) -> Any:
        with self._cond:
            if not self._cond.wait_for(lambda: self._done, timeout=timeout):
                raise TimeoutError("_AsyncFuture timed out")
            if self._exception:
                raise self._exception
            return self._result

# ----------------------------------------------------------------------
# AdaptiveThrottler
# ----------------------------------------------------------------------
class AdaptiveThrottler:
    def __init__(
        self,
        queue_ref,
        target_fraction: float = 0.70,
        min_sleep:       float = 0.001,
        max_sleep:       float = 0.500,
    ) -> None:
        self._q               = queue_ref
        self._target          = target_fraction
        self._min             = min_sleep
        self._max             = max_sleep
        self._sleep           = min_sleep
        self._lock            = threading.Lock()

    def step(self) -> None:
        maxsize = self._q.maxsize
        if maxsize <= 0:
            return
        depth = self._q.qsize() / maxsize
        with self._lock:
            if depth > self._target:
                self._sleep = min(self._sleep * 1.5, self._max)
            else:
                self._sleep = max(self._sleep * 0.8, self._min)
            t = self._sleep
        if t > self._min:
            time.sleep(t)

    @property
    def current_sleep(self) -> float:
        with self._lock:
            return self._sleep

# ----------------------------------------------------------------------
# NUMA thread pinning helper
# ----------------------------------------------------------------------
def _pin_thread_to_numa_node(node: int) -> None:
    if _HAS_NUMA and _numa_mod is not None:
        try:
            cpus = list(_numa_mod.node_to_cpus(node))
            if cpus and _HAS_PSUTIL:
                psutil.Process().cpu_affinity(cpus)
                return
        except Exception:
            pass
    if _HAS_PSUTIL:
        try:
            total = psutil.cpu_count(logical=True) or 2
            half  = max(1, total // 2)
            cpus  = list(range(half * node, min(half * (node + 1), total)))
            if cpus:
                psutil.Process().cpu_affinity(cpus)
        except Exception:
            pass

# ----------------------------------------------------------------------
# SpillManager
# ----------------------------------------------------------------------
class SpillManager:
    def __init__(
        self,
        spill_dir:         str,
        max_bytes:         int,
        max_queue:         int,
        on_file_deleted:   Callable[[str, int], None],
        on_write_complete: Callable[[str, int], None],
        numa_node:         int = 0,
    ) -> None:
        self.spill_dir         = spill_dir
        self.max_bytes         = max_bytes
        self.on_file_deleted   = on_file_deleted
        self.on_write_complete = on_write_complete
        self.current_bytes     = 0
        self._numa_node        = numa_node
        os.makedirs(spill_dir, exist_ok=True)

        self._write_queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._read_queue:  queue.Queue = queue.Queue()
        self._stop                     = False
        self._files_lru: OrderedDict   = OrderedDict()  
        self._lock                     = threading.RLock()
        self._key_to_path: Dict[Tuple[str, int], str] = {}

        self._throttler = AdaptiveThrottler(self._write_queue)
        self._start_workers()

    def _start_workers(self) -> None:
        node = self._numa_node

        def writer_loop() -> None:
            _pin_thread_to_numa_node(node)
            while not self._stop:
                try:
                    item = self._write_queue.get(timeout=0.1)
                    try:
                        self._write_sync(*item)
                    finally:
                        self._write_queue.task_done()
                except queue.Empty:
                    continue

        def reader_loop() -> None:
            _pin_thread_to_numa_node(node)
            while not self._stop:
                try:
                    key, generation, future = self._read_queue.get(timeout=0.1)
                    try:
                        tensor = self._load_sync(key, generation)
                        future.set_result(tensor)
                    except Exception as exc:
                        future.set_exception(exc)
                    finally:
                        self._read_queue.task_done()
                except queue.Empty:
                    continue

        self._writer_thread = threading.Thread(target=writer_loop, daemon=True, name="mm-spill-writer")
        self._reader_thread = threading.Thread(target=reader_loop, daemon=True, name="mm-spill-reader")
        self._writer_thread.start()
        self._reader_thread.start()

    def _get_path(self, key: str, generation: int) -> str:
        safe = key.replace("/", "_").replace("\\", "_")
        ts   = int(time.time() * 1e6)
        return os.path.join(self.spill_dir, f"{safe}_gen{generation}_{ts}.h5")

    def _write_sync(self, key: str, generation: int, tensor: torch.Tensor, old_generation: int) -> None:
        path = self._get_path(key, generation)
        arr  = tensor.cpu().numpy()

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=arr, chunks=True, **_COMPRESS_KWARGS)
            f.attrs["shape"]    = arr.shape
            f.attrs["dtype"]    = str(arr.dtype)
            f.attrs["device"]   = str(tensor.device)
            f.attrs["compress"] = _COMPRESS_NAME

        file_size = os.path.getsize(path)

        with self._lock:
            if old_generation >= 0:
                old_path = self._key_to_path.pop((key, old_generation), None)
                if old_path and old_path in self._files_lru:
                    old_size, _, _ = self._files_lru.pop(old_path)
                    self.current_bytes -= old_size
                if old_path and os.path.exists(old_path):
                    try:
                        os.unlink(old_path)
                    except OSError:
                        pass
            self._files_lru[path]                = (file_size, key, generation)
            self._key_to_path[(key, generation)] = path
            self.current_bytes                  += file_size
            if self.current_bytes > self.max_bytes:
                self._enforce_quota()

        self.on_write_complete(key, generation)

    def _load_sync(self, key: str, generation: int) -> Optional[torch.Tensor]:
        path = self._key_to_path.get((key, generation))
        if path is None or not os.path.exists(path):
            return None
        try:
            with h5py.File(path, "r") as f:
                arr = f["data"][:]
            tensor = torch.from_numpy(arr)
            with self._lock:
                if path in self._files_lru:
                    self._files_lru.move_to_end(path)
            return tensor
        except Exception as exc:
            raise DataLossError(f"Corrupted spill file {path}: {exc}") from exc

    def _enforce_quota(self) -> None:
        while self.current_bytes > self.max_bytes and self._files_lru:
            path, (size, key, gen) = self._files_lru.popitem(last=False)
            self._key_to_path.pop((key, gen), None)
            self.current_bytes -= size
            try:
                os.unlink(path)
                self.on_file_deleted(key, gen)
            except OSError:
                pass

    def spill_async(self, key: str, generation: int, tensor: torch.Tensor, old_generation: int = -1) -> None:
        for _ in range(5):
            try:
                self._write_queue.put((key, generation, tensor, old_generation), timeout=1.0)
                return
            except queue.Full:
                self._throttler.step()
        self._write_sync(key, generation, tensor, old_generation)

    def load_async(self, key: str, generation: int, callback: Callable[[Optional[torch.Tensor]], None]) -> None:
        future = _AsyncFuture()
        self._read_queue.put((key, generation, future))

        def _on_done() -> None:
            try:
                callback(future.result())
            except Exception as exc:
                warnings.warn(f"Async load failed for {key} gen={generation}: {exc}")
                callback(None)

        threading.Thread(target=_on_done, daemon=True, name="mm-load-cb").start()

    def delete_generation(self, key: str, generation: int) -> None:
        path = self._key_to_path.pop((key, generation), None)
        if path is None:
            return
        with self._lock:
            if path in self._files_lru:
                size, _, _ = self._files_lru.pop(path)
                self.current_bytes -= size
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def shutdown(self) -> None:
        self._stop = True
        if hasattr(self, "_writer_thread"):
            self._writer_thread.join(timeout=3.0)
        if hasattr(self, "_reader_thread"):
            self._reader_thread.join(timeout=3.0)

# ----------------------------------------------------------------------
# PinnedPool
# ----------------------------------------------------------------------
class PinnedPool:
    def __init__(self, max_bytes: int, pool_depth: int = POOL_DEPTH) -> None:
        self.max_bytes   = max_bytes
        self._pool_depth = pool_depth
        self._free: Dict[Tuple[int, torch.dtype], List[torch.Tensor]] = {}
        self._used_bytes = 0
        self._lock       = threading.RLock()
        self._numa_node  = 0

    def clear(self) -> None:
        with self._lock:
            self._free.clear()
            self._used_bytes = 0

    def set_numa_node(self, node: int) -> None:
        self._numa_node = node
        _pin_thread_to_numa_node(node)

    @staticmethod
    def _bucket(numel: int) -> int:
        if numel <= 0:
            return 1
        return 1 << (numel - 1).bit_length()

    def _allocate_backing(self, bucket: int, dtype: torch.dtype) -> torch.Tensor:
        if _HAS_NUMA and _ENABLE_NUMA and _numa_mod is not None:
            try:
                if hasattr(_numa_mod, "run_on_node"):
                    _numa_mod.run_on_node(self._numa_node)
            except Exception:
                pass
        if torch.cuda.is_available():
            return torch.empty((bucket,), dtype=dtype, pin_memory=True)
        return torch.empty((bucket,), dtype=dtype)

    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        numel     = math.prod(shape)
        bucket    = self._bucket(numel)
        key       = (bucket, dtype)
        candidate = None
        with self._lock:
            lst = self._free.get(key)
            if lst:
                for i in range(len(lst) - 1, -1, -1):
                    if lst[i].numel() >= numel:
                        candidate = lst.pop(i)
                        self._used_bytes -= candidate.numel() * candidate.element_size()
                        break
        if candidate is not None:
            return candidate[:numel].view(shape).zero_()
        backing = self._allocate_backing(bucket, dtype)
        return backing[:numel].view(shape).zero_()

    def release(self, buf: torch.Tensor) -> None:
            if buf is None:
                return
            numel      = buf.numel()
            bucket     = self._bucket(numel)
            key        = (bucket, buf.dtype)
            size_bytes = numel * buf.element_size()
            with self._lock:
                if key not in self._free:
                    self._free[key] = []
                if len(self._free[key]) < self._pool_depth:
                    self._free[key].append(buf)
                    self._used_bytes += size_bytes
                if self._used_bytes > self.max_bytes:
                    self._evict(self._used_bytes - self.max_bytes)

    def _evict(self, needed_bytes: int) -> None:
        freed = 0
        items = sorted(
            ((k, lst) for k, lst in self._free.items() if lst),
            key=lambda x: x[0][0],
            reverse=True,
        )
        for (bucket, dtype), lst in items:
            while lst and freed < needed_bytes:
                backing        = lst.pop()
                sz             = backing.numel() * backing.element_size()
                self._used_bytes -= sz
                freed         += sz
                del backing
            if freed >= needed_bytes:
                break

# ----------------------------------------------------------------------
# AsyncCopyEngine
# ----------------------------------------------------------------------
class AsyncCopyEngine:
    def __init__(self) -> None:
        self._streams: Dict[torch.device, torch.cuda.Stream] = {}
        self._lock = threading.RLock()

    def get_stream(self, device: torch.device) -> torch.cuda.Stream:
        with self._lock:
            if device not in self._streams:
                self._streams[device] = torch.cuda.Stream(device=device)
            return self._streams[device]

    def copy_async(self, src: torch.Tensor, dst: torch.Tensor, retries: int = 2) -> Optional[torch.cuda.Event]:
        if not torch.cuda.is_available():
            dst.copy_(src)
            return None
        for attempt in range(retries + 1):
            try:
                stream = self.get_stream(dst.device)
                with torch.cuda.stream(stream):
                    dst.copy_(src, non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
                return event
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    continue
                raise
        return None

    def wait_event(self, event: torch.cuda.Event) -> None:
        event.wait()

# ----------------------------------------------------------------------
# IndexedHeap
# ----------------------------------------------------------------------
class _IndexedHeap:
    __slots__ = ("_heap", "_versions")

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, str]] = []
        self._versions: Dict[str, int] = {}

    def push(self, key: str, score: float) -> None:
        ver = self._versions.get(key, 0) + 1
        self._versions[key] = ver
        heapq.heappush(self._heap, (score, ver, key))

    def pop_valid(self) -> Optional[str]:
        while self._heap:
            score, ver, key = heapq.heappop(self._heap)
            if self._versions.get(key) == ver:
                return key
        return None

    def peek_valid(self) -> Optional[str]:
        while self._heap:
            score, ver, key = self._heap[0]
            if self._versions.get(key) == ver:
                return key
            heapq.heappop(self._heap)
        return None

    def invalidate(self, key: str) -> None:
        if key in self._versions:
            self._versions[key] = self._versions[key] + 1

    def __len__(self) -> int:
        return len(self._heap)

# ----------------------------------------------------------------------
# EvictionPolicy
# ----------------------------------------------------------------------
class EvictionPolicy:
    def __init__(self, policy: str = "lru_freq") -> None:
        self.policy  = policy
        self._order: OrderedDict = OrderedDict()
        self._heap   = _IndexedHeap()
        self._hlock  = threading.RLock()

    def touch(self, key: str) -> None:
        if self.policy == "lru":
            if key in self._order:
                self._order.move_to_end(key)
            else:
                self._order[key] = True
        else:
            now = time.monotonic()
            if self.policy == "lru_size":
                score = now
            elif self.policy == "lru_freq":
                score = now
            else:  
                score = 0.0
            with self._hlock:
                self._heap.push(key, score)

    def select_victim(self, live: Dict[str, "BufferState"], needed_bytes: int) -> Optional[str]:
        def _eligible(state: "BufferState") -> bool:
            return (
                not state.is_free
                and state.tensor is not None
                and not state.on_disk
            )

        if self.policy == "lru":
            for key in self._order:
                state = live.get(key)
                if state and _eligible(state):
                    return key
            return None

        if self.policy == "lru_size":
            best_key, best_score = None, float("-inf")
            for key, state in live.items():
                if _eligible(state):
                    score = state.size_bytes
                    if score > best_score:
                        best_score, best_key = score, key
            return best_key

        if self.policy == "lru_freq":
            now = time.monotonic()
            best_key, best_score = None, float("inf")
            for key, state in live.items():
                if not _eligible(state):
                    continue
                recency = 1.0 / (now - state.last_access + 1e-9)
                score   = state.access_count * 0.7 + recency * 0.3
                if score < best_score:
                    best_score, best_key = score, key
            return best_key

        if self.policy == "largest_first":
            best_key, best_score = None, -1
            for key, state in live.items():
                if _eligible(state) and state.size_bytes > best_score:
                    best_score, best_key = state.size_bytes, key
            return best_key

        return None

    def remove(self, key: str) -> None:
        self._order.pop(key, None)
        with self._hlock:
            self._heap.invalidate(key)

# ----------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------
class Telemetry:
    _FIELDS = (
        "allocations", "allocation_hits", "allocation_misses",
        "spills", "loads", "evictions",
        "bytes_spilled", "bytes_loaded", "bytes_transfer_h2d",
        "prefetch_issued", "load_failures", "metadata_evictions",
        "defrag_bytes",
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            for f in self._FIELDS:
                setattr(self, f, 0)

    def inc(self, attr: str, delta: int = 1) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + delta)

    def add(self, attr: str, value: int) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + value)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {f: getattr(self, f) for f in self._FIELDS}

# ----------------------------------------------------------------------
# MemoryManager
# ----------------------------------------------------------------------
class MemoryManager:
    def __init__(
        self,
        gpu_fraction:       float = GPU_FRACTION,
        cpu_pressure_high:  float = CPU_PRESSURE_HIGH,
        cpu_pressure_low:   float = CPU_PRESSURE_LOW,
        spill_dir:          str   = SPILL_DIR,
        max_spill_bytes:    int   = MAX_SPILL_BYTES,
        prefetch_depth:     int   = PREFETCH_DEPTH,
        eviction_policy:    str   = EVICTION_POLICY,
        max_live_entries:   int   = MAX_LIVE_ENTRIES,
        backpressure_queue: int   = BACKPRESSURE_QUEUE,
        pool_depth:         int   = POOL_DEPTH,
    ) -> None:
        self._live_lock     = threading.RLock()
        self._pool_lock     = threading.RLock()
        self._prefetch_lock = threading.RLock()

        self.gpu_fraction      = gpu_fraction
        self.cpu_pressure_high = cpu_pressure_high
        self.cpu_pressure_low  = cpu_pressure_low
        self.max_live_entries  = max_live_entries
        self._pool_depth       = pool_depth
        self._stop             = False

        self._pending_writes_cond = threading.Condition(self._live_lock)

        _numa_node = 0
        if _HAS_NUMA and _ENABLE_NUMA and _numa_mod is not None:
            try:
                _numa_node = _numa_mod.get_current_node()
            except Exception:
                pass

        total_ram = psutil.virtual_memory().total if _HAS_PSUTIL else 16 * 1024 ** 3
        self.pinned_pool = PinnedPool(int(total_ram * cpu_pressure_high), pool_depth=pool_depth)
        if _HAS_NUMA and _ENABLE_NUMA:
            self.pinned_pool.set_numa_node(_numa_node)

        self.spill_mgr = SpillManager(
            spill_dir, max_spill_bytes, backpressure_queue,
            self._on_spill_file_deleted, self._on_spill_write_complete,
            numa_node=_numa_node,
        )
        self.prefetch_depth = prefetch_depth
        self.eviction       = EvictionPolicy(eviction_policy)
        self._copy_engine   = AsyncCopyEngine()
        self._telemetry     = Telemetry() if TELEMETRY else None

        self._device_pools: Dict[torch.device, Dict[Tuple[int, torch.dtype, torch.device, torch.memory_format], List[torch.Tensor]]] = {}
        self._live:           Dict[str, BufferState] = {}
        self._pending_writes: Dict[str, int]         = {}

        self._prefetch_queue:    deque = deque()
        self._prefetch_inflight: set   = set()
        self._prefetch_stop            = False
        self._prefetch_cond            = threading.Condition(self._prefetch_lock)
        self._prefetch_thread          = threading.Thread(target=self._prefetch_loop, daemon=True, name="mm-prefetch")
        self._prefetch_thread.start()

        self._cleaner_stop = False
        self._cleaner_cond = threading.Condition()
        self._cleaner_thread = threading.Thread(target=self._metadata_cleaner_loop, daemon=True, name="mm-cleaner")
        self._cleaner_thread.start()

        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._stop:
            return
        self._stop = True

        self._prefetch_stop = True
        with self._prefetch_cond:
            self._prefetch_cond.notify_all()
        self._prefetch_thread.join(timeout=3.0)

        self._cleaner_stop = True
        with self._cleaner_cond:
            self._cleaner_cond.notify_all()
        self._cleaner_thread.join(timeout=3.0)

        if hasattr(self, "spill_mgr") and self.spill_mgr is not None:
            self.spill_mgr.shutdown()

        with self._pool_lock:
            if hasattr(self, "_device_pools"):
                self._device_pools.clear()

        if hasattr(self, "_live"):
            self._live.clear()

    # ------------------------------------------------------------------
    # Spill callbacks
    # ------------------------------------------------------------------
    def _on_spill_file_deleted(self, key: str, generation: int) -> None:
        with self._live_lock:
            state = self._live.get(key)
            if state and state.spill_generation == generation and state.on_disk:
                state.load_failed   = True
                state.last_modified = time.monotonic()
                warnings.warn(f"Spill file for '{key}' gen={generation} deleted by quota — buffer lost")

    def _on_spill_write_complete(self, key: str, generation: int) -> None:
        with self._live_lock:
            self._pending_writes.pop(key, None)
            self._pending_writes_cond.notify_all()

    # ------------------------------------------------------------------
    # Metadata cleaner
    # ------------------------------------------------------------------
    def _metadata_cleaner_loop(self) -> None:
        while not self._cleaner_stop:
            with self._cleaner_cond:
                interval = 10 if len(self._live) > 500 else 30
                self._cleaner_cond.wait_for(lambda: self._cleaner_stop, timeout=interval)
            if self._cleaner_stop:
                break
            now = time.monotonic()
            to_delete: List[str] = []
            with self._live_lock:
                for key, state in list(self._live.items()):
                    if state.is_evicted and not state.on_disk:
                        to_delete.append(key)
                    elif state.load_failed and (now - state.last_modified > 300):
                        to_delete.append(key)
                    elif state.is_free and not state.on_disk and (now - state.last_modified > 600):
                        to_delete.append(key)
                for key in to_delete:
                    del self._live[key]
                    if self._telemetry:
                        self._telemetry.inc("metadata_evictions")

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------
    def _prefetch_loop(self) -> None:
        while not self._prefetch_stop:
            with self._prefetch_cond:
                while not self._prefetch_queue and not self._prefetch_stop:
                    self._prefetch_cond.wait()
                if self._prefetch_stop:
                    return
                batch: List[Tuple[str, torch.device]] = []
                limit = min(self.prefetch_depth, len(self._prefetch_queue))
                for _ in range(limit):
                    key, dev, _ = self._prefetch_queue.popleft()
                    if key not in self._prefetch_inflight:
                        self._prefetch_inflight.add(key)
                        batch.append((key, dev))
            for key, dev in batch:
                try:
                    self._do_prefetch(key, dev)
                except Exception as exc:
                    warnings.warn(f"Prefetch failed for {key}: {exc}")
                finally:
                    with self._prefetch_cond:
                        self._prefetch_inflight.discard(key)
                        if len(self._prefetch_inflight) > 4096:
                            self._prefetch_inflight = set(list(self._prefetch_inflight)[-2048:])

    def _do_prefetch(self, key: str, target_dev: torch.device) -> None:
        event_to_wait: Optional[torch.cuda.Event] = None
        with self._live_lock:
            state = self._live.get(key)
            if state is None:
                return
            if state.tensor is not None and state.tensor.device == target_dev:
                return
            if state.on_disk and state.tensor is None:
                current_gen = state.spill_generation

                def _on_load(tensor: Optional[torch.Tensor], _key: str = key, _gen: int = current_gen) -> None:
                    if tensor is None:
                        with self._live_lock:
                            s = self._live.get(_key)
                            if s:
                                s.load_failed = True
                                s.last_modified = time.monotonic()
                            if self._telemetry:
                                self._telemetry.inc("load_failures")
                        return
                    with self._live_lock:
                        s = self._live.get(_key)
                        if s is None or s.spill_generation != _gen:
                            return
                        if s.tensor is not None and not s.is_evicted:
                            return
                        s.tensor      = tensor
                        s.on_disk     = False
                        s.load_failed = False
                    if target_dev.type == "cuda":
                        try:
                            gpu_t = self._allocate_device_buffer(s.shape, target_dev, s.dtype)
                            event = self._copy_engine.copy_async(tensor, gpu_t)
                            if event is None:
                                return
                            with self._live_lock:
                                s2 = self._live.get(_key)
                                if s2 is None:
                                    return
                                old_t = s2.tensor
                                s2.tensor = gpu_t
                                s2.copy_event = event
                            if old_t is not None:
                                self._release_device_buffer(old_t, _key)
                            if self._telemetry:
                                self._telemetry.add("bytes_transfer_h2d", s.size_bytes)
                        except RuntimeError as exc:
                            warnings.warn(f"OOM during prefetch copy for {_key}: {exc}")

                self.spill_mgr.load_async(key, current_gen, _on_load)
                return

            if state.tensor is not None and state.tensor.device.type == "cpu" and target_dev.type == "cuda":
                old_tensor = state.tensor
                try:
                    gpu_t = self._allocate_device_buffer(state.shape, target_dev, state.dtype)
                    event = self._copy_engine.copy_async(old_tensor, gpu_t)
                    if event is None:
                        return
                    state.tensor = gpu_t
                    state.copy_event = event
                    event_to_wait = event
                    if self._telemetry:
                        self._telemetry.add("bytes_transfer_h2d", state.size_bytes)
                except RuntimeError as exc:
                    warnings.warn(f"OOM during prefetch copy for {key}: {exc}")
                    state.tensor = old_tensor
                    return
            else:
                return

        if event_to_wait is not None:
            self._copy_engine.wait_event(event_to_wait)
            with self._live_lock:
                s = self._live.get(key)
                if s:
                    s.copy_event = None

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------
    def get_buffer(
        self,
        shape:  Tuple[int, ...],
        dtype:  torch.dtype,
        device: torch.device,
        key:    Optional[str]       = None,
        layout: torch.memory_format = torch.contiguous_format,
    ) -> torch.Tensor:
        key = key or f"buf_{uuid.uuid4().hex}"
        needed_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()

        return_tensor: Optional[torch.Tensor] = None
        event_to_wait: Optional[torch.cuda.Event] = None
        disk_gen: Optional[int] = None

        with self._live_lock:
            if len(self._live) >= self.max_live_entries:
                free_candidates = sorted(
                    ((s.last_access, k) for k, s in self._live.items() if s.is_free and not s.on_disk)
                )
                if free_candidates:
                    for _, old_key in free_candidates[:10]:
                        self._live.pop(old_key, None)
                        if self._telemetry:
                            self._telemetry.inc("metadata_evictions")
                else:
                    victim = self.eviction.select_victim(self._live, needed_bytes)
                    if victim:
                        self._evict_buffer(victim, keep_metadata=True)

            state = self._live.get(key)
            if state is not None and state.tensor is not None:
                if state.shape != tuple(shape) or state.dtype != dtype or state.device != device:
                    warnings.warn(
                        f"get_buffer: key '{key}' reused with different "
                        f"shape/dtype/device ({state.shape} {state.dtype} {state.device} "
                        f"→ {tuple(shape)} {dtype} {device}). Evicting old buffer."
                    )
                    self._evict_buffer(key, keep_metadata=False)
                else:
                    deadline = time.monotonic() + 5.0
                    while key in self._pending_writes:
                        remaining = max(0.0, deadline - time.monotonic())
                        if remaining == 0.0:
                            break
                        self._pending_writes_cond.wait(timeout=remaining)
                    state.last_access   = time.monotonic()
                    state.access_count += 1
                    self.eviction.touch(key)
                    if self._telemetry:
                        self._telemetry.inc("allocation_hits")
                    event_to_wait    = state.copy_event
                    state.copy_event = None
                    return_tensor    = state.tensor
            elif state is not None and state.on_disk:
                disk_gen = state.spill_generation
            else:
                if state is not None and state.load_failed:
                    del self._live[key]
                    state = None

        if disk_gen is not None and return_tensor is None:
            tensor = self.spill_mgr._load_sync(key, disk_gen)
            if tensor is None:
                raise DataLossError(f"Spill file missing for '{key}' gen={disk_gen}")
            with self._live_lock:
                state = self._live.get(key)
                if state is None or state.spill_generation != disk_gen:
                    del tensor
                else:
                    state.tensor        = tensor
                    state.on_disk       = False
                    state.is_evicted    = False
                    state.last_access   = time.monotonic()
                    state.access_count += 1
                    state.last_modified = time.monotonic()
                    self.eviction.touch(key)
                    if self._telemetry:
                        self._telemetry.inc("loads")
                        self._telemetry.add("bytes_loaded", state.size_bytes)
                    return_tensor = tensor

        if event_to_wait is not None:
            self._copy_engine.wait_event(event_to_wait)

        if return_tensor is not None:
            return return_tensor

        if device.type == "cuda":
            total_gpu   = torch.cuda.get_device_properties(device).total_memory
            max_allowed = int(total_gpu * self.gpu_fraction)
            for attempt in range(3):
                with self._live_lock:
                    allocated = torch.cuda.memory_allocated(device)
                    while allocated + needed_bytes > max_allowed:
                        victim = self.eviction.select_victim(self._live, needed_bytes)
                        if victim is None:
                            break
                        self._evict_buffer(victim, keep_metadata=True)
                        allocated = torch.cuda.memory_allocated(device)
                if torch.cuda.memory_allocated(device) + needed_bytes <= max_allowed:
                    break
                torch.cuda.empty_cache()
                if attempt == 2:
                    raise MemoryError(f"Cannot allocate {shape} dtype={dtype} on {device} — GPU exhausted")
                time.sleep(0.1)

        if device.type == "cpu" and _HAS_PSUTIL:
            while True:
                with self._live_lock:
                    if self.cpu_memory_pressure() <= self.cpu_pressure_high:
                        break
                    victim = self.eviction.select_victim(self._live, needed_bytes)
                    if victim is None:
                        break
                    self._evict_buffer(victim, keep_metadata=True)
                time.sleep(0.01)

        numel    = math.prod(shape)
        bucket   = self._bucket_size(numel)
        pool_key = (bucket, dtype, device, layout)
        with self._pool_lock:
            dev_pool = self._device_pools.setdefault(device, {})
            pool_lst = dev_pool.get(pool_key, [])
            if pool_lst:
                tensor = pool_lst.pop()
                if tensor.numel() != numel:
                    tensor = tensor[:numel].reshape(shape)
                tensor.zero_()
                if self._telemetry:
                    self._telemetry.inc("allocation_hits")
            else:
                if device.type == "cpu":
                    tensor = self.pinned_pool.allocate(shape, dtype)
                else:
                    tensor = torch.zeros(shape, dtype=dtype, device=device).to(memory_format=layout)
                if self._telemetry:
                    self._telemetry.inc("allocation_misses")
                    self._telemetry.inc("allocations")

        with self._live_lock:
            new_state = BufferState(key, tensor, is_pinned=(device.type == "cpu" and tensor.is_pinned()))
            self._live[key] = new_state
            self.eviction.touch(key)

        tensor._mm_key = key 
        return tensor

    @staticmethod
    def _bucket_size(numel: int) -> int:
        if numel <= 0:
            return 1
        return 1 << (numel - 1).bit_length()

    def _allocate_device_buffer(self, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype, layout: torch.memory_format = torch.contiguous_format) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device).to(memory_format=layout)

    def _release_device_buffer(self, tensor: torch.Tensor, key: str) -> None:
        if tensor is None:
            return
        numel    = tensor.numel()
        bucket   = self._bucket_size(numel)
        pool_key = (bucket, tensor.dtype, tensor.device, torch.contiguous_format)
        with self._pool_lock:
            dev_pool = self._device_pools.setdefault(tensor.device, {})
            lst      = dev_pool.setdefault(pool_key, [])
            flat     = tensor.reshape(-1)
            if flat.numel() == bucket:
                stored = flat
            else:
                stored = torch.empty((bucket,), dtype=tensor.dtype, device=tensor.device)
                stored[: flat.numel()] = flat
            if len(lst) < self._pool_depth:
                lst.append(stored)
            else:
                lst.pop(0)
                lst.append(stored)

    # ------------------------------------------------------------------
    # Eviction and spilling
    # ------------------------------------------------------------------
    def _evict_buffer(self, key: str, keep_metadata: bool = True) -> None:
        tensor_to_release: Optional[torch.Tensor] = None
        spill_args: Optional[tuple] = None
        with self._live_lock:
            state = self._live.get(key)
            if state is None or (state.is_free and state.tensor is None):
                return
            if state.tensor is not None:
                dev_type = state.tensor.device.type
                pressure = self.memory_pressure(state.tensor.device) if dev_type == "cuda" else 0.0
                if dev_type == "cuda" and pressure > 0.7:
                    spill_args = self._spill_locked(key, state)
                else:
                    tensor_to_release = state.tensor
                    state.tensor      = None
            state.is_free    = True
            state.is_evicted = keep_metadata
            if not keep_metadata:
                del self._live[key]
            self.eviction.remove(key)
            if self._telemetry:
                self._telemetry.inc("evictions")
        if tensor_to_release is not None:
            self._release_device_buffer(tensor_to_release, key)
        if spill_args is not None:
            self.spill_mgr.spill_async(*spill_args)

    def _spill_locked(self, key: str, state: BufferState) -> Optional[tuple]:
        if key in self._pending_writes or state.tensor is None or state.on_disk:
            return None
        old_gen                = state.spill_generation
        new_gen                = old_gen + 1
        state.spill_generation = new_gen
        self._pending_writes[key] = new_gen
        state.on_disk          = True
        spill_tensor           = state.tensor
        state.tensor           = None
        if self._telemetry:
            self._telemetry.inc("spills")
            self._telemetry.add("bytes_spilled", state.size_bytes)
        return (key, new_gen, spill_tensor, old_gen)

    # ------------------------------------------------------------------
    # Public release and spilling
    # ------------------------------------------------------------------
    def release_buffer(self, tensor: torch.Tensor, key: Optional[str] = None) -> None:
        if tensor is None:
            return
        key = getattr(tensor, "_mm_key", key)
        if key is None:
            return
        high_pressure = False
        with self._live_lock:
            state = self._live.get(key)
            if state is None:
                return
            state.is_free       = True
            state.last_modified = time.monotonic()
            high_pressure = (
                self.memory_pressure() > 0.8
                or (self.cpu_memory_pressure() > self.cpu_pressure_high)
            )
        if high_pressure:
            self.spill_to_disk(key)
        else:
            owned = False
            with self._live_lock:
                state = self._live.get(key)
                if state and state.tensor is tensor:
                    state.tensor = None
                    owned        = True
            if owned:
                self._release_device_buffer(tensor, key)

    def spill_to_disk(self, key: str) -> None:
        spill_args: Optional[tuple] = None
        with self._live_lock:
            state = self._live.get(key)
            if state is None or state.on_disk:
                return
            if state.tensor is not None:
                spill_args = self._spill_locked(key, state)
        if spill_args is not None:
            self.spill_mgr.spill_async(*spill_args)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------
    def prefetch(self, key: str, target_device: torch.device) -> None:
        with self._live_lock:
            if key not in self._live:
                return
        with self._prefetch_cond:
            self._prefetch_queue.append((key, target_device, time.monotonic()))
            self._prefetch_cond.notify()
        if self._telemetry:
            self._telemetry.inc("prefetch_issued")

    # ------------------------------------------------------------------
    # GPU defragmentation
    # ------------------------------------------------------------------
    def defragment_gpu(self, device: Optional[torch.device] = None, pressure_threshold: float = 0.70) -> int:
        if not torch.cuda.is_available():
            return 0
        dev = device or torch.device("cuda", 0)
        if self.memory_pressure(dev) < pressure_threshold:
            return 0

        moved_bytes = 0
        candidates: List[Tuple[str, torch.Tensor]] = []
        with self._live_lock:
            for key, state in self._live.items():
                if (
                    state.tensor is not None
                    and state.tensor.device == dev
                    and not state.is_pinned
                    and not state.on_disk
                    and not state.is_free
                    and state.copy_event is None
                ):
                    candidates.append((key, state.tensor))

        for key, old_t in candidates:
            try:
                new_t = torch.empty_like(old_t, memory_format=torch.contiguous_format)
                new_t.copy_(old_t)
                with self._live_lock:
                    state = self._live.get(key)
                    if state is not None and state.tensor is old_t:
                        state.tensor = new_t
                        moved_bytes += old_t.numel() * old_t.element_size()
                del old_t
            except Exception:
                pass

        torch.cuda.empty_cache()
        if self._telemetry:
            self._telemetry.add("defrag_bytes", moved_bytes)
        return moved_bytes

    # ------------------------------------------------------------------
    # Memory pressure metrics
    # ------------------------------------------------------------------
    def memory_pressure(self, device: Optional[torch.device] = None) -> float:
        if not torch.cuda.is_available():
            return 0.0
        dev = device or torch.device("cuda", 0)
        return torch.cuda.memory_allocated(dev) / torch.cuda.get_device_properties(dev).total_memory

    def cpu_memory_pressure(self) -> float:
        if not _HAS_PSUTIL:
            return 0.0
        return psutil.virtual_memory().percent / 100.0

    # ------------------------------------------------------------------
    # Compatibility with ops.py
    # ------------------------------------------------------------------
    def should_manage(self, shape: Tuple[int, ...]) -> bool:
        return math.prod(shape) >= 1024

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.get_buffer(shape, dtype, device)

    def allocate(self, shape: Tuple[int, ...], device: torch.device, key: Optional[str] = None,
                 layout: torch.memory_format = torch.contiguous_format, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return self.get_buffer(shape=shape, dtype=dtype, device=device, key=key, layout=layout)

    def release(self, tensor: torch.Tensor, key: Optional[str] = None) -> None:
        self.release_buffer(tensor, key)

    def owns(self, tensor: torch.Tensor) -> bool:
        key = getattr(tensor, "_mm_key", None)
        if key is None:
            return False
        with self._live_lock:
            return key in self._live

    def pin_buffer(self, host_array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(host_array)
        if torch.cuda.is_available():
            tensor = tensor.pin_memory()
        return tensor

    def make_field(self, tensor: torch.Tensor, spacing: Tuple[float, ...],
                   origin: Optional[Tuple[float, ...]] = None, key: Optional[str] = None):
        from stenpy import Field
        key = key or f"field_{id(tensor)}"
        with self._live_lock:
            if key not in self._live:
                st = BufferState(key, tensor, is_pinned=tensor.is_pinned())
                self._live[key] = st
                self.eviction.touch(key)
                tensor._mm_key = key
        return Field(tensor=tensor, spacing=spacing, origin=origin, mm=self, key=key)

    def allocate_field(self, shape: Tuple[int, ...], spacing: Tuple[float, ...],
                       origin: Optional[Tuple[float, ...]] = None,
                       device: torch.device = torch.device("cpu"),
                       key: Optional[str] = None,
                       layout: torch.memory_format = torch.contiguous_format):
        from stenpy import Field
        key = key or f"field_{'_'.join(map(str, shape))}"
        tensor = self.get_buffer(shape, torch.float64, device, key, layout)
        return Field(tensor=tensor, spacing=spacing, origin=origin, mm=self, key=key)

    def halo_exchange(self, shard: torch.Tensor, radius: int, dims: Optional[List[int]] = None) -> torch.Tensor:
        return shard

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def stats(self) -> Dict[str, int]:
        if self._telemetry:
            return self._telemetry.snapshot()
        return {}

    def clear_pool(self) -> None:
        with self._pool_lock:
            self._device_pools.clear()
        try:
            self.pinned_pool.clear()
        except Exception as e:
            warnings.warn(f"Error clearing pinned pool: {e}")
        total_ram = psutil.virtual_memory().total if _HAS_PSUTIL else 16 * 1024 ** 3
        self.pinned_pool = PinnedPool(
            int(total_ram * self.cpu_pressure_high),
            pool_depth=self._pool_depth,
        )

    def __repr__(self) -> str:
        with self._live_lock:
            live = len(self._live)
        with self._pool_lock:
            pools = sum(len(lst) for dev in self._device_pools.values() for lst in dev.values())
        return f"MemoryManager(live={live}, pooled={pools}, policy={self.eviction.policy}, gpu_frac={self.gpu_fraction}, compress={_COMPRESS_NAME})"

# ----------------------------------------------------------------------
# patching of sten.MemoryManager
# ----------------------------------------------------------------------
def patch_ops() -> None:
    try:
        import stenpy
        stenpy.MemoryManager = MemoryManager
    except ImportError:
        pass

if os.environ.get("OPS_MM_PATCH_OPS", "0").lower() in ("1", "true", "yes"):
    patch_ops()
