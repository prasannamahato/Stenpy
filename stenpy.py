# stenpy.py
from __future__ import annotations
import contextlib
import datetime
import inspect                       
import logging
import math
import os
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False

import stenpy_engine as engine
from stenpy_engine import (
    Field, Graph, GraphNode, Runtime,
    register_operator, OP_REGISTRY, OP_METADATA,
    dist_init, is_distributed, dist_rank, dist_size, decompose_field,
    capability_report, validate_fft_consistency, validate_operator_identity,
    compile_expression, parse_expression, UserVar,
    LazyFieldProtocol, is_lazy_field, set_lazy_field_factory,
)
from memory_manager import (
    MemoryManager, HardwareProfile, AdaptiveThresholds,
    get_default_thresholds, set_hardware_profile, use_advanced_mm,
)

_DTYPE = torch.float64
_MAX_LAZY_FIELD_HANDLES = int(os.environ.get("OPS_MAX_LAZY_FIELD_HANDLES", "8"))
_SPACING_HINTS = ("spacing", "dx", "voxel_size", "resolution")


def _assert_fp64(t: torch.Tensor, where: str) -> None:
    if t.dtype not in (_DTYPE, torch.complex128):
        raise TypeError(f"{where}: expected float64 or complex128, got {t.dtype}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_ENABLED = os.environ.get("STENPY_LOG", "1").lower() in ("1", "true", "yes")
_LOG_DIR = os.environ.get("STENPY_LOG_DIR", ".")
_logger = logging.getLogger("stenpy")
_logger.setLevel(logging.INFO)
_logger.propagate = False


def _log_path_for_today() -> str:
    return os.path.join(_LOG_DIR, f"stenpy_{datetime.date.today().isoformat()}.log")


def _ensure_log_handler() -> None:
    if not _LOG_ENABLED:
        return
    today_path = _log_path_for_today()
    current = getattr(_logger, "_stenpy_log_path", None)
    if current == today_path and _logger.handlers:
        return
    for h in list(_logger.handlers):
        _logger.removeHandler(h)
        h.close()
    handler = logging.FileHandler(today_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(handler)
    _logger._stenpy_log_path = today_path


def log_event(msg: str, level: str = "info") -> None:
    if not _LOG_ENABLED:
        return
    _ensure_log_handler()
    getattr(_logger, level, _logger.info)(msg)


def set_logging(enabled: bool) -> None:
    global _LOG_ENABLED
    _LOG_ENABLED = enabled


# ---------------------------------------------------------------------------
# LazyField 
# ---------------------------------------------------------------------------

class LazyField(LazyFieldProtocol):
    def __init__(self, path: str, dataset_name: str = "field") -> None:
        self.path = path
        self.dataset_name = dataset_name
        self._lock = threading.RLock()
        self._local = threading.local()
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
                with contextlib.suppress(Exception):
                    old.close()

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

    @property
    def nbytes(self) -> int:
        return int(math.prod(self.shape) * 8)

    def __getitem__(self, idx: Any) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            return self._local.dset[idx]

    def close(self) -> None:
        with self._lock:
            for f in self._all_files:
                try:
                    f.close()
                except Exception:
                    pass
            self._all_files.clear()
        if getattr(self._local, "file", None) is not None:
            self._local.file = None
            self._local.dset = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        try:
            sh = self.shape
        except Exception:
            sh = "?"
        return f"LazyField(path={self.path}, shape={sh}, fp64)"


set_lazy_field_factory(LazyField)


@contextlib.contextmanager
def _open_hdf5_field(path: str) -> Iterator[Tuple["h5py.Dataset", Tuple[int, ...]]]:
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


# ---------------------------------------------------------------------------
# I/O: load_tensor / save_tensor / normalize
# ---------------------------------------------------------------------------

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

        shape = dset.shape
        total_gb = float(np.prod(shape)) * 8 / 1024 ** 3

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
                    arr = f[cand][:]
                    break
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
    tensor: torch.Tensor,
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
    path: str,
    chunks: bool = True,
) -> None:
    _assert_fp64(tensor, "save_tensor")
    arr = tensor.detach().cpu().numpy()
    chunk = tuple(min(128, s) for s in arr.shape) if chunks else None
    with h5py.File(path, "w") as f:
        f.create_dataset("field", data=arr, chunks=chunk)
        f.create_dataset("spacing", data=np.array(spacing, dtype=np.float64))
        f.create_dataset("origin", data=np.array(origin, dtype=np.float64))


def _normalize_hdf5(
    path: str,
    field_key: str = "field",
    spacing_key: str = "spacing",
    origin_key: str = "origin",
    spacing_value: Optional[Tuple[float, ...]] = None,
    output_path: Optional[str] = None,
) -> str:
    import tempfile
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".h5", prefix="stenpy_norm_")
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
        fo.create_dataset("field", data=data, chunks=chunk)

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

        origin = (
            fi[origin_key][:].tolist()
            if origin_key in fi
            else list(fi.attrs.get("origin", [0.0] * data.ndim))
        )
        fo.create_dataset("origin", data=np.array(origin, dtype=np.float64))

    return output_path


# ---------------------------------------------------------------------------
# Performance reporting
# ---------------------------------------------------------------------------

class PerfStats:
    def __init__(self) -> None:
        self.t0: Optional[float] = None
        self.t1: Optional[float] = None
        self.bytes_read = 0
        self.bytes_written = 0
        self.flops = 0
        self.n_chunks = 0

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def stop(self) -> None:
        self.t1 = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        if self.t0 is None:
            return 0.0
        return (self.t1 or time.perf_counter()) - self.t0

    def bandwidth_gbps(self) -> float:
        total = self.bytes_read + self.bytes_written
        return (total / 1024 ** 3) / max(self.elapsed_s, 1e-9)

    def gflops(self) -> float:
        return (self.flops / 1e9) / max(self.elapsed_s, 1e-9)


def _print_perf_report(stats: PerfStats, op_name: str, device: str) -> None:
    print(
        "\n\033[1m─── stenpy perf report ───\033[0m\n"
        f"  op            : {op_name}\n"
        f"  device        : {device}\n"
        f"  elapsed       : {stats.elapsed_s:.4f} s\n"
        f"  bandwidth     : {stats.bandwidth_gbps():.3f} GB/s\n"
        f"  chunks        : {stats.n_chunks}\n"
        f"  GFLOPs/s (est): {stats.gflops():.3f}\n"
        "\033[1m──────────────────────────\033[0m"
    )


def _progress(iterable, total: Optional[int] = None, desc: str = ""):
    if _HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


# ---------------------------------------------------------------------------
# Streaming execution over HDF5 
# ---------------------------------------------------------------------------

_MIN_CHUNK_ROWS = 4
_RECHECK_EVERY_CHUNKS = 4
_VRAM_SHRINK_TRIGGER = 0.85
_RAM_SHRINK_TRIGGER = 0.90


def _tensor_gb_stream(shape: Tuple[int, ...]) -> float:
    return math.prod(shape) * 8 / 1024 ** 3


def _compute_chunk_rows(full_shape, device, output_multiplier: float) -> int:
    thresholds = get_default_thresholds()
    per_row_bytes = 8 * (math.prod(full_shape[1:]) if len(full_shape) > 1 else 1)
    per_row_bytes *= (1.0 + max(output_multiplier, 0.0))
    budget = thresholds.row_stream_threshold_bytes()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free, _total = torch.cuda.mem_get_info(device)
            budget = min(budget, int(free * 0.5))
        except Exception:
            pass
    elif _HAS_PSUTIL:
        budget = min(budget, int(_psutil.virtual_memory().available * 0.5))
    rows = max(1, int(budget / max(per_row_bytes, 1)))
    return max(_MIN_CHUNK_ROWS, min(rows, full_shape[0]))


def _adapt_chunk_rows(current_rows, full_shape, device, output_multiplier, halo=0) -> int:
    min_halo_rows = max(_MIN_CHUNK_ROWS, 2 * halo + 1)
    if device.type == "cuda" and torch.cuda.is_available():
        used = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        if used / max(total, 1) >= _VRAM_SHRINK_TRIGGER:
            new_rows = max(min_halo_rows, _compute_chunk_rows(full_shape, device, output_multiplier))
            if new_rows < current_rows:
                return new_rows
    if _HAS_PSUTIL and _psutil.virtual_memory().percent / 100.0 >= _RAM_SHRINK_TRIGGER:
        new_rows = max(min_halo_rows, _compute_chunk_rows(full_shape, device, output_multiplier))
        if new_rows < current_rows:
            return new_rows
    return current_rows


def _read_halo_rows(ds, row_start, row_end, halo, periodic):
    total = ds.shape[0]
    if halo <= 0:
        return ds[row_start:row_end], 0
    if periodic:
        wanted = row_end - row_start + 2 * halo
        pos = row_start - halo
        if pos >= 0 and pos + wanted <= total:
            return ds[pos: pos + wanted], halo
        parts = [ds[i % total: i % total + 1] for i in range(pos, pos + wanted)]
        return np.concatenate(parts, axis=0), halo
    read_start = max(0, row_start - halo)
    read_end = min(total, row_end + halo)
    return ds[read_start:read_end], row_start - read_start


def _filter_op_kwargs(op_fn, **kwargs):
    try:
        sig = inspect.signature(op_fn)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (ValueError, TypeError):
        return kwargs


def stream_execute(
    op_name: str,
    input_path: str,
    out_path: str,
    device: Union[str, torch.device] = "cpu",
    dx: float = 1.0,
    boundary: str = "periodic",
    dataset_key: Optional[str] = None,
    verbose: bool = True,
    **params,
) -> Dict[str, Any]:
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unknown operator '{op_name}' (known: {list(OP_REGISTRY)})")

    dev = torch.device(device)
    mm = MemoryManager()
    rt = Runtime(mm, device=str(dev))
    stats = PerfStats()
    stats.start()
    log_event(f"stream_execute start op={op_name} in={input_path} out={out_path}")

    f_in = h5py.File(input_path, "r")
    try:
        key = dataset_key or next(
            (k for k in ("data", "field") if k in f_in and isinstance(f_in[k], h5py.Dataset)),
            next(k for k in f_in if isinstance(f_in[k], h5py.Dataset)),
        )
        ds = f_in[key]
        spacing = tuple(float(v) for v in f_in.attrs.get("spacing", [dx]))
        origin = tuple(float(v) for v in f_in.attrs.get("origin", [0.0] * len(spacing)))
        total_rows = ds.shape[0]
        in_shape = tuple(ds.shape)

        meta = OP_METADATA.get(op_name, {})
        halo = int(meta.get("stencil_radius", 0))
        periodic = (boundary == "periodic")
        probe_rows = min(max(2 * halo + 1, 3), total_rows)
        probe_np = ds[:probe_rows].astype(np.float64, copy=False)
        probe_t = torch.from_numpy(probe_np).to(_DTYPE).to(dev)

        g = Graph()
        src = g.add("_constant", (), {"value": probe_t})
        op_fn = OP_REGISTRY[op_name]
        op_params = _filter_op_kwargs(op_fn, dx=dx, boundary=boundary, **params)
        sink = g.add(op_name, (src,), op_params)
        with torch.no_grad():
            probe_out = rt.run(g)[sink]
        out_trailing = tuple(probe_out.shape[1:])
        del probe_out, probe_t
        mm.clear_pool(full=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        out_full_shape = (total_rows,) + out_trailing
        out_multiplier = (math.prod(out_trailing) if out_trailing else 1) / max(
            math.prod(in_shape[1:]) if len(in_shape) > 1 else 1, 1
        )

        if verbose:
            print(
                f"stream_execute('{op_name}')  in={in_shape} ({_tensor_gb_stream(in_shape):.3f} GB)"
                f"  out={out_full_shape} ({_tensor_gb_stream(out_full_shape):.3f} GB)  halo={halo}"
            )

        f_out = h5py.File(out_path, "w")
        try:
            chunk_rows_hint = max(1, min(64, total_rows))
            out_chunk_shape = (chunk_rows_hint,) + out_trailing
            out_ds = f_out.create_dataset("data", shape=out_full_shape, dtype=np.float64, chunks=out_chunk_shape)
            f_out.attrs["spacing"] = list(spacing)
            f_out.attrs["origin"] = list(origin) if origin else [0.0] * len(spacing)

            min_rows = max(_MIN_CHUNK_ROWS, 2 * halo + 1)
            batch_rows = max(min_rows, _compute_chunk_rows(in_shape, dev, out_multiplier))

            row_pos = 0
            n_chunks = 0
            out_min, out_max, out_sum, out_count = float("inf"), float("-inf"), 0.0, 0

            while row_pos < total_rows:
                if n_chunks % _RECHECK_EVERY_CHUNKS == 0:
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                        torch.cuda.empty_cache()
                    batch_rows = _adapt_chunk_rows(batch_rows, in_shape, dev, out_multiplier, halo=halo)

                remaining = total_rows - row_pos
                this_batch = max(min_rows, min(remaining, batch_rows)) if remaining >= min_rows else remaining
                row_start, row_end = row_pos, min(row_pos + max(this_batch, 1), total_rows)

                padded_np, halo_left = _read_halo_rows(ds, row_start, row_end, halo, periodic)
                padded_np = padded_np.astype(np.float64, copy=False)
                n_rows_chunk = row_end - row_start

                chunk_t = torch.from_numpy(np.ascontiguousarray(padded_np)).to(_DTYPE).to(dev)
                stats.bytes_read += chunk_t.numel() * 8

                g = Graph()
                src = g.add("_constant", (), {"value": chunk_t})
                sink = g.add(op_name, (src,), op_params) 
                with torch.no_grad():
                    out_t = rt.run(g)[sink]

                if halo > 0:
                    out_t = out_t[halo_left: halo_left + n_rows_chunk]
                if not out_t.is_contiguous():
                    out_t = out_t.contiguous()
                out_cpu = out_t.detach().cpu().numpy()
                stats.bytes_written += out_cpu.nbytes

                out_ds[row_start:row_end] = out_cpu
                out_min = min(out_min, float(out_cpu.min()))
                out_max = max(out_max, float(out_cpu.max()))
                out_sum += float(out_cpu.sum())
                out_count += out_cpu.size

                del chunk_t, out_t, out_cpu
                mm.clear_pool(full=True)

                row_pos = row_end
                n_chunks += 1
                if verbose and n_chunks % 10 == 0:
                    print(f"  {row_pos}/{total_rows} rows  batch={this_batch}")
        finally:
            f_out.close()
    finally:
        f_in.close()

    stats.stop()
    stats.n_chunks = n_chunks
    log_event(f"stream_execute done op={op_name} elapsed={stats.elapsed_s:.3f}s chunks={n_chunks}")

    return {
        "shape_out": list(out_full_shape),
        "min": out_min if out_min != float("inf") else 0.0,
        "max": out_max if out_max != float("-inf") else 0.0,
        "mean": out_sum / out_count if out_count else 0.0,
        "elapsed_s": stats.elapsed_s,
        "n_chunks": n_chunks,
        "out_path": out_path,
        "_perf_stats": stats,
    }

# ---------------------------------------------------------------------------
# Chainable configuration syntax
# ---------------------------------------------------------------------------

_KNOWN_SETTINGS = {"stream_i", "stream_o", "device", "log", "perf_report", "output", "boundary", "dx", "chunk_mb"}
_INTERNAL_PREFIX = "_"


def _resolve_device(device_str: Union[str, torch.device]) -> torch.device:
    if isinstance(device_str, torch.device):
        device_str = str(device_str)
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() "
                "is False -- no GPU visible to this process. Check your "
                "CUDA/driver install, or use device='cpu' explicitly."
            )
        return torch.device("cuda:0")
    dev = torch.device(device_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device_str!r} was requested but torch.cuda.is_available() "
            f"is False -- no GPU visible to this process."
        )
    return dev


class Pipeline:
    def __init__(self, op_name: str, expr_str: str) -> None:
        object.__setattr__(self, "_op_name", op_name)
        object.__setattr__(self, "_expr_str", expr_str)
        object.__setattr__(self, "_stream_i", False)
        object.__setattr__(self, "_stream_o", False)
        object.__setattr__(self, "_device", "cpu")
        object.__setattr__(self, "_chunk_mb", None) 
        object.__setattr__(self, "_log", True)
        object.__setattr__(self, "_perf_report", False)
        object.__setattr__(self, "_boundary", "periodic")
        object.__setattr__(self, "_dx", 1.0)
        object.__setattr__(self, "_field_paths", {}) 
        object.__setattr__(self, "_constants", {})    
        object.__setattr__(self, "_output", None)
        object.__setattr__(self, "_result", None)
        object.__setattr__(self, "_ran", False)


    def __setattr__(self, name: str, value: Any) -> None:
        if name in _KNOWN_SETTINGS:
            object.__setattr__(self, f"_{name}", value)
            if name == "output":
                self.run()
            return
        if name.startswith(_INTERNAL_PREFIX):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, str):
            self._field_paths[name] = value
        else:
            self._constants[name] = value

    def set(self, **kwargs) -> "Pipeline":
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def run(self) -> Any:
        if self._ran:
            return self._result
        object.__setattr__(self, "_ran", True)

        dev = _resolve_device(self._device)
        stats = PerfStats()
        stats.start()
        if self._log:
            log_event(f"pipeline start op={self._op_name} expr={self._expr_str!r} device={self._device}")

        field_map: Dict[str, torch.Tensor] = {}
        lazy_map: Dict[str, LazyField] = {}
        spacing = origin = None

        for var, path in self._field_paths.items():
            if self._stream_i:
                lf = LazyField(path)
                lazy_map[var] = lf
                with h5py.File(path, "r") as f:
                    spacing = tuple(float(v) for v in f.attrs.get("spacing", [self._dx]))
                    origin = tuple(float(v) for v in f.attrs.get("origin", [0.0] * len(spacing)))
                probe = torch.from_numpy(np.asarray(lf[:1])).to(_DTYPE)
                field_map[var] = probe  
            else:
                tensor, sp, org = load_tensor(path, device=dev, return_mode="eager")
                field_map[var] = tensor
                spacing, origin = sp, org

        for var, val in self._constants.items():
            field_map.setdefault(var, torch.tensor(float(val), dtype=_DTYPE))

        graph, sink_id, warns, resolved_map, sympy_expr = compile_expression(
            self._expr_str, dx=self._dx, boundary=self._boundary, field_map=field_map,
        )
        for w in warns:
            warnings.warn(w)

        mm = MemoryManager()
        rt = Runtime(mm, device=str(dev))

        if self._stream_i or self._stream_o:
            if len(lazy_map) != 1:
                raise NotImplementedError(
                    "stream_i/stream_o currently support a single streamed "
                    "input field per pipeline; use stream_execute() directly "
                    "for multi-field streamed graphs."
                )
            (var, lf), = lazy_map.items()
            if not self._output:
                raise ValueError("stream_o requires .output to be set to a path.")
            result_meta = self._run_streamed_graph(
                graph=graph,
                sink_id=sink_id,
                streamed_field=lf,
                streamed_node_name=resolved_map[var],   
                out_path=self._output,
                rt=rt,
                stats=stats,
            )
            object.__setattr__(self, "_result", LazyField(self._output))
            stats = result_meta.pop("_perf_stats", stats)
        else:
            with torch.no_grad():
                out = rt.run(graph)[sink_id]
            if self._output:
                save_tensor(out, spacing or (1.0,), origin or (0.0,), self._output)
                object.__setattr__(self, "_result", LazyField(self._output))
            else:
                object.__setattr__(self, "_result", out)

        stats.stop()
        if self._log:
            log_event(f"pipeline done op={self._op_name} elapsed={stats.elapsed_s:.3f}s")
        if self._perf_report:
            _print_perf_report(stats, self._op_name, str(dev))

        return self._result

    def _run_streamed_graph(
        self,
        graph: Graph,
        sink_id: str,
        streamed_field: LazyField,
        streamed_node_name: str,
        out_path: str,
        rt: Runtime,
        stats: PerfStats,
    ) -> Dict[str, Any]:
        import queue as _queue

        dev = rt.device
        total_rows = streamed_field.shape[0]
        halo = self._graph_stencil_radius(graph)
        boundary = self._boundary
        periodic = (boundary == "periodic")

        probe_rows = min(max(2 * halo + 1, 3), total_rows)
        arr_probe, _, _ = self._read_halo_chunk(streamed_field, 0, 0, probe_rows, total_rows, halo, boundary)
        probe_t = torch.from_numpy(arr_probe.astype(np.float64)).to(_DTYPE).to(dev)
        g_probe = graph.clone_with_replacement(streamed_node_name, probe_t)
        with torch.no_grad():
            r_probe = rt.run(g_probe)
        out_trailing = tuple(r_probe[sink_id].shape[1:])
        del r_probe, g_probe, probe_t
        rt.mm.clear_pool(full=True)

        out_full_shape = (total_rows,) + out_trailing
        min_chunk_rows = max(1, 2 * halo + 1) if halo > 0 else 1
        chunk_size = max(min_chunk_rows, self._auto_chunk_size(streamed_field, dev, 0))

        bytes_per_row = math.prod(streamed_field.shape[1:]) * 8 if len(streamed_field.shape) > 1 else 8
        print(
            f"stenpy stream: device={dev}  total_rows={total_rows}  "
            f"initial_chunk={chunk_size} rows (~{chunk_size * bytes_per_row / 1024**2:.1f} MB/chunk)  "
            f"buffer_multiplier={self._STREAM_BUFFER_MULTIPLIER}"
        )

        f_out = h5py.File(out_path, "w")
        out_ds = f_out.create_dataset(
            "data", shape=out_full_shape, dtype=np.float64,
            chunks=(min(64, total_rows),) + out_trailing,
        )
        with h5py.File(streamed_field.path, "r") as fin:
            if "spacing" in fin.attrs:
                f_out.attrs["spacing"] = list(fin.attrs["spacing"])
            if "origin" in fin.attrs:
                f_out.attrs["origin"] = list(fin.attrs["origin"])

        prefetch_q = _queue.Queue(maxsize=1)
        write_q = _queue.Queue(maxsize=2)
        out_min, out_max, out_sum, out_count = float("inf"), float("-inf"), 0.0, 0
        RECHECK_EVERY = 4

        def reader():
            nonlocal chunk_size
            start = 0
            i = 0
            while start < total_rows:
                if i > 0 and i % RECHECK_EVERY == 0:
                    new_size = max(min_chunk_rows, self._auto_chunk_size(streamed_field, dev, 0))
                    if new_size != chunk_size:
                        chunk_size = new_size
                end = min(start + chunk_size, total_rows)
                arr, lp, rp = self._read_halo_chunk(streamed_field, 0, start, end, total_rows, halo, boundary)
                prefetch_q.put((start, end, arr, lp, rp))
                start = end
                i += 1
            prefetch_q.put(None)

        def writer():
            while True:
                item = write_q.get()
                if item is None:
                    break
                w_start, w_end, arr_cpu = item
                sl = [slice(None)] * len(out_full_shape)
                sl[0] = slice(w_start, w_end)
                out_ds[tuple(sl)] = arr_cpu
                del arr_cpu

        reader_thread = threading.Thread(target=reader, daemon=True)
        writer_thread = threading.Thread(target=writer, daemon=True)
        reader_thread.start()
        writer_thread.start()
        n_chunks = 0
        while True:
            item = prefetch_q.get()
            if item is None:
                break
            start, end, arr, lp, rp = item
            chunk_t = torch.from_numpy(arr.astype(np.float64)).to(_DTYPE).to(dev)
            cg = graph.clone_with_replacement(streamed_node_name, chunk_t)
            with torch.no_grad():
                cr = rt.run(cg)
            cout = cr[sink_id]
            if halo > 0:
                cout = cout[lp: lp + (end - start)]
            cout_cpu = cout.cpu().numpy()
            write_q.put((start, end, cout_cpu))
            out_min = min(out_min, float(cout_cpu.min()))
            out_max = max(out_max, float(cout_cpu.max()))
            out_sum += float(cout_cpu.sum())
            out_count += cout_cpu.size
            del chunk_t, cg, cr, cout, cout_cpu
            n_chunks += 1
            rt.mm.clear_pool(full=True)

        write_q.put(None)
        writer_thread.join()
        reader_thread.join()
        f_out.close()

        return {
            "shape_out": list(out_full_shape),
            "min": out_min,
            "max": out_max,
            "mean": out_sum / out_count if out_count else 0.0,
            "n_chunks": n_chunks,
            "out_path": out_path,
            "_perf_stats": stats,
        }

    def _graph_stencil_radius(self, graph: Graph) -> int:
        r = 0
        for node in graph._nodes.values():
            meta = OP_METADATA.get(node.op_name, {})
            r = max(r, int(meta.get("stencil_radius", 0) or 0))
        return r
    _STREAM_BUFFER_MULTIPLIER = int(os.environ.get("OPS_STREAM_BUFFER_MULTIPLIER", "6"))
    _STREAM_SAFETY_FRACTION = float(os.environ.get("OPS_STREAM_SAFETY_FRACTION", "0.35"))
    _STREAM_MIN_CHUNK_MB = float(os.environ.get("OPS_STREAM_MIN_CHUNK_MB", "8"))
    _STREAM_MAX_CHUNK_MB = float(os.environ.get("OPS_STREAM_MAX_CHUNK_MB", "512"))

    def _stream_chunk_budget_bytes(self, dev: torch.device) -> int:
        if self._chunk_mb is not None:
            return int(self._chunk_mb * 1024 ** 2)

        if dev.type == "cuda" and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(dev)
            avail = free
        elif _HAS_PSUTIL:
            avail = _psutil.virtual_memory().available
        else:
            avail = 2 * 1024 ** 3 

        budget = int(avail * self._STREAM_SAFETY_FRACTION)
        budget = max(budget, int(self._STREAM_MIN_CHUNK_MB * 1024 ** 2))
        budget = min(budget, int(self._STREAM_MAX_CHUNK_MB * 1024 ** 2))
        return budget

    def _auto_chunk_size(self, lazy_field: LazyField, dev: torch.device, chunk_dim: int = 0) -> int:
        shape = lazy_field.shape
        slice_elements = math.prod(s for i, s in enumerate(shape) if i != chunk_dim)
        bytes_per_row = max(slice_elements * 8, 1)

        total_budget = self._stream_chunk_budget_bytes(dev)
        per_buffer_budget = max(total_budget // self._STREAM_BUFFER_MULTIPLIER, bytes_per_row)
        rows = max(1, int(per_buffer_budget / bytes_per_row))
        return min(rows, shape[chunk_dim])

    def _read_halo_chunk(self, lazy_field, chunk_dim, start, end, total, radius, boundary):
        ndim = len(lazy_field.shape)
        def _slice_read(a, b):
            idx = [slice(None)] * ndim
            idx[chunk_dim] = slice(a, b)
            return lazy_field[tuple(idx)]
        if radius <= 0:
            return _slice_read(start, end), 0, 0
        if boundary != "periodic":
            lo = max(0, start - radius)
            hi = min(total, end + radius)
            return _slice_read(lo, hi), start - lo, hi - end
        pieces = []
        if start - radius >= 0:
            pieces.append(_slice_read(start - radius, start))
        else:
            pieces.append(_slice_read(total - (radius - start), total))
            if start > 0:
                pieces.append(_slice_read(0, start))
        pieces.append(_slice_read(start, end))
        if end + radius <= total:
            pieces.append(_slice_read(end, end + radius))
        else:
            if end < total:
                pieces.append(_slice_read(end, total))
            pieces.append(_slice_read(0, (end + radius) - total))
        arr = np.concatenate(pieces, axis=chunk_dim) if len(pieces) > 1 else pieces[0]
        return arr, radius, radius


    @property
    def value(self) -> Any:
        return self.run()

    def __getitem__(self, idx):
        return self.run()[idx]

    def __repr__(self) -> str:
        if self._result is not None:
            return f"Pipeline({self._op_name}, result={self._result!r})"
        return f"Pipeline({self._op_name}({self._expr_str}), unconfigured)"


class _OperatorDispatch:
    def __init__(self, op_name: str) -> None:
        self._op_name = op_name

    def __call__(self, expr, *args, **kwargs) -> Pipeline:
        expr_str = expr if isinstance(expr, str) else str(expr)
        p = Pipeline(self._op_name, expr_str)
        if kwargs:
            p.set(**kwargs)
        return p


def __getattr__(name: str):
    if name in OP_REGISTRY:
        return _OperatorDispatch(name)
    if hasattr(engine, name):
        return getattr(engine, name)
    raise AttributeError(f"module 'stenpy' has no attribute {name!r}")


__all__ = [
    "Field", "Graph", "GraphNode", "Runtime", "MemoryManager",
    "register_operator", "OP_REGISTRY", "OP_METADATA",
    "dist_init", "is_distributed", "dist_rank", "dist_size", "decompose_field",
    "capability_report", "validate_fft_consistency", "validate_operator_identity",
    "use_advanced_mm", "HardwareProfile", "AdaptiveThresholds",
    "get_default_thresholds", "set_hardware_profile",
    "compile_expression", "parse_expression", "UserVar",
    "LazyField", "load_tensor", "save_tensor", "stream_execute", "_open_hdf5_field",
    "Pipeline", "PerfStats", "log_event", "set_logging",
]
