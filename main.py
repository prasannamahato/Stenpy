# mains.py  
from __future__ import annotations
import argparse
import datetime
import json
import logging
import math
import os
import re
import shutil
import sys
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from queue import Full as _QueueFull, Queue, Empty as _QueueEmpty
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import gc
import psutil
import time as _time_module

DEADLINE_SECONDS = 15 * 3600    
PIPELINE_START_TIME = _time_module.perf_counter()
_LOG_DIR = Path("outputs") / f"logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "processing.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(_LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
_logger = logging.getLogger("ops")
_logger.info(f"Logging initialized to {_LOG_FILE}")

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import sympy as sp
    from sympy import Function, Symbol, Add, Mul, Pow
    from sympy.core.function import AppliedUndef, UndefinedFunction
except ImportError:
    sys.exit("SymPy is required: pip install sympy")

def _dbg(msg: str) -> None:
    if os.environ.get("OPS_DEBUG", "0") == "1":
        print(f"[DEBUG] {msg}")

def _log_memory_state(label: str, device: torch.device) -> None:
    try:
        ram_info = psutil.virtual_memory()
        ram_pct = ram_info.percent
        ram_used_gb = ram_info.used / 1024**3
        ram_total_gb = ram_info.total / 1024**3
        ram_msg = f"RAM: {ram_used_gb:.2f}/{ram_total_gb:.1f} GB ({ram_pct:.1f}%)"
        
        if device.type == "cuda" and torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(device) / 1024**3
            vram_peak = torch.cuda.max_memory_allocated(device) / 1024**3
            vram_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            vram_pct = (vram_used / vram_total) * 100 if vram_total > 0 else 0
            vram_msg = f"VRAM: {vram_used:.2f}/{vram_total:.1f} GB ({vram_pct:.1f}%) [peak: {vram_peak:.2f} GB]"
            msg = f"{label} | {ram_msg} | {vram_msg}"
        else:
            msg = f"{label} | {ram_msg}"
        
        _logger.info(msg)
        _rank0_print(f"  [LOG] {msg}")
    except Exception as e:
        _logger.error(f"Error logging memory state: {e}")



def _is_cuda_oom(exc: Exception) -> bool:
    try:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg or "oom" in msg


try:
    import stenpy
except ImportError:
    sys.exit("sten.py not found — place it in the same directory as main.py")

import h5py as _h5py

_CHUNK_VRAM_FRACTION  = float(os.environ.get("OPS_CHUNK_VRAM_FRAC",  "0.0005"))  
_CHUNK_RAM_FRACTION   = float(os.environ.get("OPS_CHUNK_RAM_FRAC",   "0.001"))  

_VRAM_SHRINK_TRIGGER  = float(os.environ.get("OPS_VRAM_SHRINK",      "0.10"))   
_RAM_SHRINK_TRIGGER   = float(os.environ.get("OPS_RAM_SHRINK",       "0.10")) 

_MIN_CHUNK_ROWS = int(os.environ.get("OPS_MIN_CHUNK_ROWS", "1"))              
_MAX_CHUNK_FRAC = float(os.environ.get("OPS_MAX_CHUNK_FRAC", "0.01"))       
_HDF5_RDCC_NBYTES  = int(os.environ.get("OPS_HDF5_RDCC_NBYTES",  str(32 * 1024 * 1024)))
_HDF5_RDCC_NSLOTS  = int(os.environ.get("OPS_HDF5_RDCC_NSLOTS",  "10007"))
_HDF5_RDCC_W0      = float(os.environ.get("OPS_HDF5_RDCC_W0",    "0.75"))
_HDF5_MAX_CHUNK_BYTES = int(os.environ.get("OPS_HDF5_MAX_CHUNK_BYTES", str(64 * 1024 * 1024)))
_HDF5_FLUSH_EVERY_ROWS = int(os.environ.get("OPS_HDF5_FLUSH_ROWS", "64"))
_PIPE_DEPTH        = int(os.environ.get("OPS_PIPE_DEPTH", "1"))                 
_ROW_STREAM_THRESHOLD = int(os.environ.get("OPS_ROW_STREAM_THRESHOLD", str(16 * 1024 * 1024)))  

def _is_rocm() -> bool:
    try:
        import torch.version
        if getattr(torch.version, "hip", None) is not None:
            return True
    except Exception:
        pass
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = torch.cuda.get_device_properties(0).name
            if "AMD" in name or "Radeon" in name or "MI" in name:
                return True
    except Exception:
        pass
    return False

if _is_rocm():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    print("  ○  ROCm/HIP detected (AMD GPU) — using HIP-safe allocator config")
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

def _harmonize_memory_settings(device: torch.device) -> None:
    global _VRAM_SHRINK_TRIGGER, _RAM_SHRINK_TRIGGER
    
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        total_vram_gb = props.total_memory / 1024**3
        
        if total_vram_gb >= 24:
            _VRAM_SHRINK_TRIGGER = max(_VRAM_SHRINK_TRIGGER, 0.85)
        elif total_vram_gb >= 12:
            _VRAM_SHRINK_TRIGGER = max(_VRAM_SHRINK_TRIGGER, 0.80)
        else:
            _VRAM_SHRINK_TRIGGER = max(_VRAM_SHRINK_TRIGGER, 0.70)
        _rank0_print(f"  ○  VRAM harmonized: {total_vram_gb:.1f}GB detected. Shrink trigger set to {_VRAM_SHRINK_TRIGGER*100:.0f}%.")
            
    try:
        total_ram_gb = psutil.virtual_memory().total / 1024**3
        if total_ram_gb >= 64:
            _RAM_SHRINK_TRIGGER = max(_RAM_SHRINK_TRIGGER, 0.85)
        elif total_ram_gb >= 32:
            _RAM_SHRINK_TRIGGER = max(_RAM_SHRINK_TRIGGER, 0.80)
        else:
            _RAM_SHRINK_TRIGGER = max(_RAM_SHRINK_TRIGGER, 0.75)
        _rank0_print(f"  ○  RAM harmonized: {total_ram_gb:.1f}GB detected. Shrink trigger set to {_RAM_SHRINK_TRIGGER*100:.0f}%.")
    except Exception:
        pass

# ----------------------------------------------------------------------
# HPC environment detection
# ----------------------------------------------------------------------
_HPC_RANK_VARS = (
    "SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK",
    "MPI_RANK", "MV2_COMM_WORLD_RANK", "JSM_NAMESPACE_RANK",
)
_IS_HPC: bool = any(k in os.environ for k in _HPC_RANK_VARS)

def _env_rank() -> int:
    for k in _HPC_RANK_VARS:
        v = os.environ.get(k)
        if v is not None:
            try: return int(v)
            except ValueError: pass
    return 0

def _env_world() -> int:
    for k in ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE",
              "MPI_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"):
        v = os.environ.get(k)
        if v is not None:
            try: return max(1, int(v))
            except ValueError: pass
    return 1

_HPC_RANK:  int = _env_rank()
_HPC_WORLD: int = _env_world()

def _rank0_print(*args, **kwargs) -> None:
    if not _IS_HPC or _HPC_RANK == 0:
        print(*args, **kwargs)

def _hpc_scratch() -> str:
    for var in ("SCRATCH", "TMPDIR", "LUSTRE_SCRATCH", "WORK"):
        p = os.environ.get(var)
        if p and os.path.isdir(p):
            return p
    return "."

# ----------------------------------------------------------------------
# tqdm fallback
# ----------------------------------------------------------------------
class _FallbackBar:
    def __init__(self, total=None, desc="", unit="", **kw):
        self.total = total or 0; self.n = 0; self.desc = desc
        self._t0 = time.perf_counter()
        _rank0_print(f"  {desc} …")

    def update(self, n=1):
        self.n += n
        pct = 100 * self.n / self.total if self.total else 0
        elapsed = time.perf_counter() - self._t0
        if not _IS_HPC or _HPC_RANK == 0:
            print(f"\r  {self.desc}  {pct:5.1f}%  [{elapsed:.1f}s]", end="", flush=True)

    def set_postfix_str(self, s, **kw): pass
    def set_postfix(self, **kw): pass
    def close(self):
        if not _IS_HPC or _HPC_RANK == 0: print()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()

def _make_bar(total, desc="", unit="chunk", colour=None):
    if HAS_TQDM and (not _IS_HPC or _HPC_RANK == 0):
        return _tqdm(total=total, desc=f"  {desc}", unit=unit, dynamic_ncols=True,
                     colour=colour or "cyan",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")
    return _FallbackBar(total=total, desc=desc, unit=unit)

# ----------------------------------------------------------------------
# Nerd mode
# ----------------------------------------------------------------------
_NERD = os.environ.get("OPS_NERD", "0").lower() in ("1", "true", "yes")
_T_CRIT = 90
_T_WARN = 72
_T_NOTE = 45

def _sev_label(pct: float) -> str:
    if pct >= _T_CRIT: return "  ██ CRIT"
    if pct >= _T_WARN: return "  ▲▲ WARN"
    if pct >= _T_NOTE: return "  ●  NOTE"
    return                    "  ○  OK  "

def _vram_bar(device: torch.device, width: int = 28) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "CPU — no VRAM gauge"
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory
    if total <= 0: return "VRAM total unavailable"
    used  = torch.cuda.memory_allocated(device)
    peak  = torch.cuda.max_memory_allocated(device)
    frac  = max(0.0, min(1.0, used / total))
    pfrac = max(0.0, min(1.0, peak / total))
    fi    = min(int(frac  * width), width)
    pi    = min(int(pfrac * width), width)
    bar   = "█" * fi + "▒" * max(0, pi - fi) + "░" * max(0, width - max(fi, pi))
    pct   = frac * 100
    return (f"[{bar}] {used/1024**3:.2f}/{total/1024**3:.2f} GB  "
            f"{pct:.0f}%{_sev_label(pct)}")

def _hdf5_dissect(path: str) -> None:
    if not _NERD: return
    try: disk_mb = os.path.getsize(path) / 1024**2
    except OSError: disk_mb = 0.0
    _rank0_print(f"\n┌─── HDF5 dissection ─────────────────────────────────────────")
    _rank0_print(f"  │  file      : {Path(path).name}")
    _rank0_print(f"  │  disk size : {disk_mb:.2f} MB")
    try:
        with _h5py.File(path, "r") as f:
            if dict(f.attrs):
                _rank0_print(f"  │  file attrs:")
                for k, v in f.attrs.items():
                    _rank0_print(f"  │    {k} = {v}")
            def _visit(name: str, obj: Any) -> None:
                if not isinstance(obj, _h5py.Dataset): return
                nbytes = obj.dtype.itemsize * int(np.prod(obj.shape))
                stored = obj.id.get_storage_size()
                ratio  = stored / max(nbytes, 1)
                comp   = obj.compression or "none"
                comp_lvl = f"/{obj.compression_opts}" if obj.compression_opts is not None else ""
                ram_gb = nbytes / 1024**3
                _rank0_print(f"  │  /{name}")
                _rank0_print(f"  │    shape      : {tuple(obj.shape)}")
                _rank0_print(f"  │    dtype      : {obj.dtype}")
                _rank0_print(f"  │    ram        : {ram_gb:.3f} GB  ({nbytes:,} bytes uncompressed)")
                _rank0_print(f"  │    disk       : {stored/1024**2:.2f} MB  (ratio {ratio:.3f})")
                _rank0_print(f"  │    chunks     : {obj.chunks}")
                _rank0_print(f"  │    compress   : {comp}{comp_lvl}")
                if dict(obj.attrs):
                    for k, v in obj.attrs.items():
                        _rank0_print(f"  │    attr '{k}' : {v}")
            f.visititems(_visit)
    except Exception as exc:
        _rank0_print(f"  │  [error reading HDF5: {exc}]")
    _rank0_print(f"  └─────────────────────────────────────────────────────────────")

def _nerd(msg: str) -> None:
    if _NERD and (not _IS_HPC or _HPC_RANK == 0):
        print(f"  ○  {msg}")

def _budget_explain(
    full_shape:  Tuple[int, ...],
    device:      torch.device,
    n_fields:    int,
    chunk_rows:  int,
    n_chunks:    int,
) -> None:
    if not _NERD: return
    row_bytes = math.prod(full_shape[1:]) * 8
    dim0      = full_shape[0]
    cap       = max(1, int(dim0 * _MAX_CHUNK_FRAC))
    _rank0_print(f"\n┌─── Chunk budget ─────────────────────────────────────────────")
    if device.type == "cuda" and torch.cuda.is_available():
        free_vram  = _free_vram_bytes(device)
        free_ram   = _free_ram_bytes()
        _rank0_print(f"  │  free VRAM          : {free_vram/1024**3:.3f} GB")
        _rank0_print(f"  │  free RAM           : {free_ram/1024**3:.3f} GB")
        _rank0_print(f"  │  VRAM shrink trigger: {_VRAM_SHRINK_TRIGGER:.0%}")
        _rank0_print(f"  │  RAM  shrink trigger: {_RAM_SHRINK_TRIGGER:.0%}")
        _rank0_print(f"  │  row size (input)   : {row_bytes/1024**2:.3f} MB  "
                     f"({math.prod(full_shape[1:])} elements × 8 B)")
        _rank0_print(f"  │  dim-0 cap ({_MAX_CHUNK_FRAC:.0%})    : {cap}")
    else:
        free_ram = _free_ram_bytes()
        _rank0_print(f"  │  device             : CPU")
        _rank0_print(f"  │  free RAM           : {free_ram/1024**3:.3f} GB")
        _rank0_print(f"  │  row size           : {row_bytes/1024**2:.3f} MB")
    _rank0_print(f"  │  ──────────────────────────────────────────────────────────")
    _rank0_print(f"  │  chunk_rows (initial): {chunk_rows}  "
                 f"({chunk_rows * row_bytes * n_fields / 1024**3:.3f} GB input / iter)")
    _rank0_print(f"  │  n_chunks            : {n_chunks}  (dim-0 {dim0} rows)")
    _rank0_print(f"  │  adaptive shrinking  : ON (triggers at "
                 f"VRAM>{_VRAM_SHRINK_TRIGGER:.0%} / RAM>{_RAM_SHRINK_TRIGGER:.0%})")
    _rank0_print(f"  │  pipe_depth          : {_PIPE_DEPTH}  (triple-buffer)")
    _rank0_print(f"  └─────────────────────────────────────────────────────────────")

# ----------------------------------------------------------------------
# Performance instrumentation
# ----------------------------------------------------------------------
_FLOP_PER_ELEMENT: Dict[str, float] = {
    "add": 1.0, "sub": 1.0, "mul": 1.0, "div": 4.0, "neg": 1.0, "clamp": 2.0,
    "exp": 20.0, "log": 20.0, "sqrt": 4.0, "sin": 15.0, "tanh": 25.0,
    "sum": 1.0, "mean": 2.0, "norm_l2": 3.0, "variance": 4.0, "entropy": 6.0,
    "integrate": 3.0, "cumulative_integral": 3.0, "gradient": 6.0, "divergence": 6.0,
    "laplacian": 14.0, "curl": 18.0, "spectral_gradient": 0.0, "spectral_laplacian": 0.0,
    "fft": 0.0, "ifft": 0.0, "trace": 2.0, "determinant": 6.0, "eigenvalues": 12.0,
    "inverse": 10.0, "deviatoric": 5.0, "moving_average": 3.0, "min_max": 2.0,
    "covariance": 4.0, "correlation": 6.0, "surface_integral": 4.0,
    "distance_transform": 0.0, "stack_components": 0.0, "select_component": 0.0,
    "norm_last": 3.0, "div_last": 1.0, "neg_stack3": 4.0, "scale_eye": 1.0,
    "velocity_dot_grad": 1.0,
}

_HEAVY_OP_OVERHEAD: Dict[str, float] = {
    "gradient_nd":          5.0,   
    "mean_curvature":       7.0,   
    "surface_normals":      5.0,   
    "material_derivative":  5.0,  
    "hessian":             10.0,  
    "curl":                 3.5,   
    "spectral_gradient":    3.0,   
    "spectral_laplacian":   3.0,
    "eigenvalues":          4.0,
    "inverse":              4.0,
    "determinant":          3.0,
}

def _expr_peak_overhead(expr_str: str) -> float:
    peak = 1.0
    for op, factor in _HEAVY_OP_OVERHEAD.items():
        if re.search(rf'\b{re.escape(op)}\b', expr_str):
            peak = max(peak, factor)
    return peak

def _ast_flops(expr: sp.Expr, n_elem: int) -> float:
    if n_elem <= 0: return 0.0
    if isinstance(expr, (sp.Symbol, sp.Number, sp.NumberSymbol, sp.core.numbers.ImaginaryUnit)):
        return 0.0
    if isinstance(expr, sp.core.function.AppliedUndef):
        func_name = type(expr).__name__
        canonical = _ALIAS.get(func_name.lower(), func_name.lower())
        fpe       = _FLOP_PER_ELEMENT.get(canonical, 1.0)
        op_flops = 5.0 * n_elem * max(1.0, math.log2(max(n_elem, 2))) if fpe == 0.0 else fpe * n_elem
        child_flops = sum(_ast_flops(a, n_elem) for a in expr.args)
        return op_flops + child_flops
    if isinstance(expr, sp.Add):
        add_flops   = max(0, len(expr.args) - 1) * _FLOP_PER_ELEMENT["add"] * n_elem
        child_flops = sum(_ast_flops(a, n_elem) for a in expr.args)
        return add_flops + child_flops
    if isinstance(expr, sp.Mul):
        if sp.Integer(-1) in expr.args and len(expr.args) == 2:
            other = next(a for a in expr.args if a != sp.Integer(-1))
            return (_FLOP_PER_ELEMENT["neg"] * n_elem + _ast_flops(other, n_elem))
        mul_flops   = max(0, len(expr.args) - 1) * _FLOP_PER_ELEMENT["mul"] * n_elem
        child_flops = sum(_ast_flops(a, n_elem) for a in expr.args)
        return mul_flops + child_flops
    if isinstance(expr, sp.Pow):
        base, exp_s  = expr.args
        exp_val      = float(exp_s)
        child_flops  = _ast_flops(base, n_elem)
        if abs(exp_val - 1.0) < 1e-9: return child_flops
        if abs(exp_val - 0.5) < 1e-9:
            return _FLOP_PER_ELEMENT.get("sqrt", 4.0) * n_elem + child_flops
        exp_is_int = abs(exp_val - round(exp_val)) < 1e-9
        exp_int    = int(round(abs(exp_val))) if exp_is_int else None
        if exp_is_int and exp_int is not None and 2 <= exp_int <= 8:
            mul_cost = (exp_int - 1) * _FLOP_PER_ELEMENT["mul"] * n_elem
            if exp_val < 0: mul_cost += _FLOP_PER_ELEMENT["div"] * n_elem
            return mul_cost + child_flops
        return ((_FLOP_PER_ELEMENT.get("log",  20.0) +
                 _FLOP_PER_ELEMENT.get("mul",   1.0) +
                 _FLOP_PER_ELEMENT.get("exp",  20.0)) * n_elem + child_flops)
    return sum(_ast_flops(a, n_elem) for a in getattr(expr, "args", ()))

def _gpu_peak_bw_gbs(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available(): return float("nan")
    props = torch.cuda.get_device_properties(device)
    if not hasattr(props, 'memory_clock_rate') or not hasattr(props, 'memory_bus_width'):
        return float("nan")
    return (2.0 * props.memory_clock_rate * 1e3 * (props.memory_bus_width / 8)) / 1e9

def _gpu_name(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available(): return "CPU"
    p = torch.cuda.get_device_properties(device)
    return f"{p.name}  ({p.total_memory/1024**3:.0f} GB VRAM)"

@dataclass
class PerfStats:
    expr_str:   str
    mode:       str
    input_gb:   float = 0.0
    output_gb:  float = 0.0
    t_compile:  float = 0.0
    t_read:     float = 0.0
    t_h2d:      float = 0.0
    t_compute:  float = 0.0
    t_d2h:      float = 0.0
    t_write:    float = 0.0
    t_total:    float = 0.0
    chunk_read_s:    List[float] = dc_field(default_factory=list)
    chunk_h2d_s:     List[float] = dc_field(default_factory=list)
    chunk_compute_s: List[float] = dc_field(default_factory=list)
    chunk_dh_s:      List[float] = dc_field(default_factory=list)
    chunk_write_s:   List[float] = dc_field(default_factory=list)
    n_chunks:        int   = 0
    chunk_rows:      int   = 0
    peak_vram_gb:    float = 0.0
    vram_total_gb:   float = 0.0
    est_flops:       float = 0.0

    def read_bw_gbs(self) -> float: return self.input_gb  / self.t_read    if self.t_read    > 1e-9 else 0.0
    def write_bw_gbs(self) -> float: return self.output_gb / self.t_write   if self.t_write   > 1e-9 else 0.0
    def h2d_bw_gbs(self) -> float: return self.input_gb  / self.t_h2d     if self.t_h2d     > 1e-9 else 0.0
    def d2h_bw_gbs(self) -> float: return self.output_gb / self.t_d2h     if self.t_d2h     > 1e-9 else 0.0
    def compute_bw_gbs(self) -> float: return self.input_gb  / self.t_compute if self.t_compute > 1e-9 else 0.0
    def tflops(self) -> float: return (self.est_flops / 1e12) / self.t_compute if self.t_compute > 1e-9 else 0.0
    def overall_bw_gbs(self) -> float: return (self.input_gb + self.output_gb) / self.t_total if self.t_total > 1e-9 else 0.0

def _arr_stats(vals: List[float]) -> Tuple[float, float, float, float]:
    if not vals: return 0.0, 0.0, 0.0, 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

def _fmt_gb(v: float) -> str:
    if v >= 1.0:   return f"{v:.3f} GB"
    if v >= 1e-3:  return f"{v*1024:.1f} MB"
    return f"{v*1024**2:.0f} KB"

def _fmt_s(v: float) -> str:
    if v >= 60.0:  return f"{v/60:.1f} min"
    if v >= 1.0:   return f"{v:.3f} s"
    return f"{v*1000:.1f} ms"

def _fmt_bw(v: float) -> str:
    if math.isnan(v) or v == 0.0: return "  —  "
    return f"{v:.2f} GB/s"

def _fmt_tf(v: float) -> str:
    if v == 0.0:   return "  —  "
    if v >= 1.0:   return f"{v:.3f} TFLOP/s"
    if v >= 1e-3:  return f"{v*1000:.2f} GFLOP/s"
    return f"{v*1e6:.1f} MFLOP/s"

def _pct_bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)

def _truncate(s: str, n: int) -> str:
    if len(s) <= n: return s
    return s[:n - 1] + "…"

def _print_perf_report(
    stats_list:   List[PerfStats],
    device:       torch.device,
    total_wall_s: float,
    verbose:      bool = True,
) -> None:
    if not verbose or (_IS_HPC and _HPC_RANK != 0) or not stats_list: return
    W = 76
    def _hdr(title: str) -> None: print(f"  ┌─── {title} {'─' * max(0, W - 6 - len(title))}┐")
    def _ftr() -> None: print(f"  └{'─' * W}┘")
    def _row(label: str, value: str, note: str = "") -> None:
        note_s = f"  {note}" if note else ""
        inner  = f"  {label:<30}{value:<20}{note_s}"
        pad    = W - len(inner)
        print(f"  │{inner}{' ' * max(0, pad)}│")
    def _divider() -> None: print(f"  │{'─' * W}│")
    def _blank() -> None: print(f"  │{' ' * W}│")

    agg_input_gb  = sum(s.input_gb   for s in stats_list)
    agg_output_gb = sum(s.output_gb  for s in stats_list)
    agg_flops     = sum(s.est_flops  for s in stats_list)
    agg_t_read    = sum(s.t_read     for s in stats_list)
    agg_t_h2d     = sum(s.t_h2d      for s in stats_list)
    agg_t_compile = sum(s.t_compile  for s in stats_list)
    agg_t_compute = sum(s.t_compute  for s in stats_list)
    agg_t_d2h     = sum(s.t_d2h      for s in stats_list)
    agg_t_write   = sum(s.t_write    for s in stats_list)
    agg_chunks    = sum(s.n_chunks   for s in stats_list)
    peak_vram_gb  = max((s.peak_vram_gb  for s in stats_list), default=0.0)
    vram_total_gb = max((s.vram_total_gb for s in stats_list), default=0.0)
    gpu_peak_bw  = _gpu_peak_bw_gbs(device)
    agg_tflops_s = (agg_flops / 1e12) / agg_t_compute if agg_t_compute > 1e-9 else 0.0
    agg_read_bw  = agg_input_gb  / agg_t_read    if agg_t_read    > 1e-9 else 0.0
    agg_write_bw = agg_output_gb / agg_t_write   if agg_t_write   > 1e-9 else 0.0
    agg_h2d_bw   = agg_input_gb  / agg_t_h2d     if agg_t_h2d     > 1e-9 else 0.0
    agg_d2h_bw   = agg_output_gb / agg_t_d2h     if agg_t_d2h     > 1e-9 else 0.0
    compute_data_gb = agg_input_gb + agg_output_gb
    mem_util_pct = (
        (compute_data_gb / agg_t_compute) / gpu_peak_bw * 100.0
        if (agg_t_compute > 1e-9 and not math.isnan(gpu_peak_bw) and gpu_peak_bw > 0)
        else float("nan")
    )
    total_stages = (agg_t_read + agg_t_h2d + agg_t_compile
                    + agg_t_compute + agg_t_d2h + agg_t_write)
    def _pct(t: float) -> str:
        if total_stages < 1e-9: return "  —"
        return f"{t / total_stages * 100:4.0f}%"

    all_read    = [v for s in stats_list for v in s.chunk_read_s]
    all_h2d     = [v for s in stats_list for v in s.chunk_h2d_s]
    all_compute = [v for s in stats_list for v in s.chunk_compute_s]
    all_dh      = [v for s in stats_list for v in s.chunk_dh_s]
    all_write   = [v for s in stats_list for v in s.chunk_write_s]
    has_chunks  = bool(all_compute)
    n_exprs     = len(stats_list)
    device_name = _gpu_name(device)

    print()
    title = f"  PERFORMANCE REPORT  ─  {n_exprs} expression(s)  ─  {device_name}"
    print(f"  {'═' * W}")
    print(f"{title}")
    print(f"  {'═' * W}")
    _hdr("OVERALL THROUGHPUT")
    _row("Wall time (total):",       _fmt_s(total_wall_s))
    _row("Data read from disk:",     _fmt_gb(agg_input_gb), f"→ {_fmt_bw(agg_read_bw)} read")
    _row("Data written to disk:",    _fmt_gb(agg_output_gb), f"→ {_fmt_bw(agg_write_bw)} write")
    _row("Aggregate data moved:",    _fmt_gb(agg_input_gb + agg_output_gb))
    if agg_flops > 0:
        if agg_flops >= 1e12:   flop_str = f"{agg_flops/1e12:.3f} TFLOP (AST-estimated)"
        elif agg_flops >= 1e9:  flop_str = f"{agg_flops/1e9:.2f} GFLOP (AST-estimated)"
        else:                   flop_str = f"{agg_flops/1e6:.1f} MFLOP (AST-estimated)"
        _row("Estimated FLOPs:", flop_str)
        _row("Achieved compute rate:", _fmt_tf(agg_tflops_s))
    if not math.isnan(gpu_peak_bw):
        _row("GPU peak mem bandwidth:", f"{gpu_peak_bw:.0f} GB/s  (theoretical)")
    if not math.isnan(mem_util_pct):
        bar = _pct_bar(mem_util_pct / 100, width=16)
        _row("GPU BW utilisation:", f"{mem_util_pct:.1f}%  [{bar}]")
    _ftr()

    _hdr("STAGE BREAKDOWN  (cumulative over all runs)")
    inner = f"  {'Stage':<30}{'Time':>10}   {'Share':>5}   {'Bandwidth / Rate'}"
    print(f"  │{inner}{' ' * max(0, W - len(inner))}│")
    _divider()
    def _stage_row(label: str, t: float, bw_label: str = "") -> None:
        if t < 1e-9: return
        frac  = t / total_stages if total_stages > 1e-9 else 0.0
        bar   = _pct_bar(frac, width=10)
        inner = (f"  {label:<30}{_fmt_s(t):>10}   {_pct(t):>5}"
                 f"   [{bar}]  {bw_label}")
        pad   = W - len(inner)
        print(f"  │{inner}{' ' * max(0, pad)}│")
    _stage_row("HDF5 read  (disk → CPU)",   agg_t_read,    _fmt_bw(agg_read_bw))
    _stage_row("H→D transfer  (CPU → GPU)", agg_t_h2d,    _fmt_bw(agg_h2d_bw) + " (PCIe/NVLink)")
    _stage_row("Compile / graph trace",     agg_t_compile)
    _stage_row("GPU compute",               agg_t_compute, _fmt_tf(agg_tflops_s))
    _stage_row("D→H transfer  (GPU → CPU)", agg_t_d2h,    _fmt_bw(agg_d2h_bw) + " (PCIe/NVLink)")
    _stage_row("HDF5 write  (CPU → disk)",  agg_t_write,  _fmt_bw(agg_write_bw))
    _blank()
    stage_times = {
        "Disk read":    agg_t_read,
        "H→D transfer": agg_t_h2d,
        "GPU compute":  agg_t_compute,
        "D→H transfer": agg_t_d2h,
        "Disk write":   agg_t_write,
    }
    if any(v > 1e-9 for v in stage_times.values()):
        bottleneck = max(stage_times, key=stage_times.get)
        bpct = stage_times[bottleneck] / total_stages * 100 if total_stages > 1e-9 else 0.0
        inner = f"  ▶  Bottleneck: {bottleneck}  ({bpct:.0f}% of stage time)"
        print(f"  │{inner}{' ' * max(0, W - len(inner))}│")
    _ftr()

    if has_chunks:
        _hdr("CHUNK STATISTICS  (lazy / streaming mode)")
        _row("Total chunks processed:", str(agg_chunks))
        if agg_chunks and n_exprs:
            _row("Avg chunks / expression:", f"{agg_chunks / n_exprs:.1f}")
        def _chunk_stat_row(label: str, vals: List[float]) -> None:
            if not vals: return
            mean, std, mn, mx = _arr_stats(vals)
            inner = (f"  {label:<30}"
                     f"  mean {_fmt_s(mean):<12}"
                     f"  ±{_fmt_s(std):<10}"
                     f"  min {_fmt_s(mn):<10}"
                     f"  max {_fmt_s(mx)}")
            pad = W - len(inner)
            print(f"  │{inner}{' ' * max(0, pad)}│")
        _chunk_stat_row("Disk read  (per chunk):", all_read)
        _chunk_stat_row("H→D transfer:", all_h2d)
        _chunk_stat_row("GPU compute:", all_compute)
        _chunk_stat_row("D→H transfer:", all_dh)
        _chunk_stat_row("HDF5 write:", all_write)
        if len(all_compute) >= 2:
            mean_c, std_c, _, _ = _arr_stats(all_compute)
            cv = std_c / mean_c if mean_c > 0 else 0.0
            _blank()
            cv_note = ("uniform" if cv < 0.05 else "moderate variance" if cv < 0.15 else "HIGH variance — check input regularity")
            inner = f"  Compute CV (std/mean): {cv:.3f}  →  {cv_note}"
            print(f"  │{inner}{' ' * max(0, W - len(inner))}│")
        _ftr()

    _hdr("PER-EXPRESSION BREAKDOWN")
    col_e = 28
    hdr   = (f"  {'#':<3}  {'Expression':<{col_e}}"
             f"  {'Mode':<6}  {'Time':>9}  {'In':>8}  {'Out':>8}"
             f"  {'BW':>9}  {'TFLOP/s':>10}")
    print(f"  │{hdr}{' ' * max(0, W - len(hdr))}│")
    _divider()
    for i, s in enumerate(stats_list, 1):
        expr_short = _truncate(s.expr_str, col_e)
        bw   = s.overall_bw_gbs()
        tf   = s.tflops()
        bw_s = f"{bw:.2f}" if bw > 0 else "—"
        tf_s = _fmt_tf(tf) if tf > 0 else "—"
        row  = (f"  {i:<3}  {expr_short:<{col_e}}"
                f"  {s.mode:<6}  {_fmt_s(s.t_total):>9}"
                f"  {_fmt_gb(s.input_gb):>8}  {_fmt_gb(s.output_gb):>8}"
                f"  {bw_s:>9}  {tf_s:>10}")
        pad = W - len(row)
        print(f"  │{row}{' ' * max(0, pad)}│")
    _ftr()

    if device.type == "cuda" and peak_vram_gb > 0:
        _hdr("GPU MEMORY  (peak across all runs)")
        vram_pct = peak_vram_gb / vram_total_gb * 100 if vram_total_gb > 0 else 0.0
        bar = _pct_bar(vram_pct / 100, width=24)
        _row("Peak VRAM used:", f"{peak_vram_gb:.3f} / {vram_total_gb:.1f} GB", f"({vram_pct:.1f}%)  [{bar}]")
        if has_chunks and stats_list:
            s0 = stats_list[0]
            if s0.n_chunks and s0.input_gb:
                chunk_gb = s0.input_gb / s0.n_chunks
                _row("Est. chunk working-set:", f"{_fmt_gb(chunk_gb)} × {s0.n_chunks} chunks")
        _ftr()
    print(f"  {'═' * W}")
    print()

# ----------------------------------------------------------------------
# Symbolic operator vocabulary
# ----------------------------------------------------------------------
_ALIAS: Dict[str, str] = {
    "gradient": "gradient", "grad": "gradient", "divergence": "divergence", "div": "divergence",
    "laplacian": "laplacian", "laplace": "laplacian", "lap": "laplacian", "curl": "curl",
    "spectral_gradient": "spectral_gradient", "spectral_laplacian": "spectral_laplacian",
    "fft": "fft", "ifft": "ifft", "add": "add", "sub": "sub", "mul": "mul", "div_op": "div",
    "neg": "neg", "clamp": "clamp", "exp": "exp", "log": "log", "sqrt": "sqrt", "sin": "sin",
    "tanh": "tanh", "sum": "sum", "mean": "mean", "norm_l2": "norm_l2", "variance": "variance",
    "entropy": "entropy", "integrate": "integrate", "cumulative_integral": "cumulative_integral",
    "trace": "trace", "determinant": "determinant", "eigenvalues": "eigenvalues", "inverse": "inverse",
    "deviatoric": "deviatoric", "var": "variance", "moving_average": "moving_average",
    "min_max": "min_max", "minmax": "min_max", "covariance": "covariance", "cov": "covariance",
    "correlation": "correlation", "corr": "correlation", "surface_integral": "surface_integral",
    "distance_transform": "distance_transform",
}
_NO_STENCIL_OPS: Set[str] = {
    "add", "sub", "mul", "div", "neg", "clamp", "exp", "log", "sqrt", "sin", "tanh",
    "sum", "mean", "norm_l2", "min_max", "variance", "entropy", "trace", "determinant",
    "eigenvalues", "inverse", "deviatoric", "fft", "ifft", "covariance", "correlation",
    "moving_average", "min_max", "covariance", "correlation", "surface_integral",
    "distance_transform", "stack_components", "select_component", "norm_last",
    "div_last", "neg_stack3", "scale_eye", "velocity_dot_grad",
    "integrate", "cumulative_integral", "spectral_gradient", "spectral_laplacian",
}
_SHORT_OP: Dict[str, str] = {
    "spectral_gradient": "spec_grad", "spectral_laplacian": "spec_lap",
    "cumulative_integral": "cumintg", "determinant": "det", "eigenvalues": "eig",
    "divergence": "div", "laplacian": "lap", "gradient": "grad", "variance": "var",
    "entropy": "ent", "integrate": "intg", "inverse": "inv", "deviatoric": "dev",
    "norm_l2": "norm", "trace": "tr", "clamp": "clamp", "curl": "curl", "tanh": "tanh",
    "sqrt": "sqrt", "mean": "mean", "neg": "neg", "exp": "exp", "log": "log", "sin": "sin",
    "sum": "sum", "fft": "fft", "add": "add", "sub": "sub", "mul": "mul", "ifft": "ifft",
}
_SIMPLIFY_RULES: Dict[Tuple[str, str], str] = {
    ("divergence", "gradient"): "laplacian",
}
_RESERVED_SYMBOLS: frozenset = frozenset({"x", "y", "z", "t"})

# ----------------------------------------------------------------------
# Short-form filename helper
# ----------------------------------------------------------------------
def _expr_to_filename(expr_str: str, src_stem: str) -> str:
    s = expr_str.strip()
    for long_name in sorted(_SHORT_OP, key=len, reverse=True):
        s = re.sub(rf'\b{re.escape(long_name)}\b', _SHORT_OP[long_name], s)
    s = re.sub(r',\s*\w+=[\w.]+', '', s)
    s = re.sub(r'\b([fguvh])\b', src_stem, s)
    s = s.replace(' ', '')
    s = re.sub(r'[^\w()+\-]', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:120]

# ----------------------------------------------------------------------
# User-defined symbolic variables
# ----------------------------------------------------------------------
@dataclass
class UserVar:
    name:    str
    start:   float
    end:     float
    steps:   int
    current: float = 0.0

    @property
    def values(self) -> np.ndarray: return np.linspace(self.start, self.end, self.steps)
    @property
    def step_size(self) -> float: return 0.0 if self.steps < 2 else (self.end - self.start) / (self.steps - 1)

_USER_VARS: Dict[str, UserVar] = {}

def _define_user_var(pipeline: Optional["Pipeline"] = None) -> Optional[UserVar]:
    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │         DEFINE A SYMBOLIC VARIABLE               │")
    print("  │  Examples:  t (time)   omega (frequency)  k      │")
    print("  └──────────────────────────────────────────────────┘")
    try: name = input("  Variable name  ❯ ").strip()
    except (EOFError, KeyboardInterrupt): print(); return None
    if not name or not re.match(r'^[A-Za-z_]\w*$', name): print("  ✗ Invalid name."); return None
    if pipeline is not None and name in pipeline.fields:
        print(f"  ✗ '{name}' is already a loaded field — choose a different name.")
        return None
    if name in _USER_VARS: print(f"  ℹ  '{name}' already defined — overwriting.")
    try:
        start = float(input(f"  Start value for {name}  ❯ ").strip())
        end   = float(input(f"  End   value for {name}  ❯ ").strip())
        raw_s = input(f"  Number of sample points [1 = single value]  ❯ ").strip()
        steps = max(1, int(raw_s) if raw_s else 1)
    except (ValueError, EOFError, KeyboardInterrupt): print("  ✗ Aborted."); return None
    uv = UserVar(name=name, start=start, end=end, steps=steps, current=start)
    _USER_VARS[name] = uv
    print(f"\n✓ '{name}'  range [{start}, {end}]  steps={steps}")
    if steps > 1: print(f"     step size = {uv.step_size:.6g}")
    return uv

def _show_user_vars() -> None:
    if not _USER_VARS: print("  (no user-defined variables yet — use [6] to add one)"); return
    print(f"\n{'Name':<12}  {'Start':>10}  {'End':>10}  {'Steps':>6}  {'Last used':>12}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*12}")
    for uv in _USER_VARS.values():
        print(f"  {uv.name:<12}  {uv.start:>10.4g}  {uv.end:>10.4g}  {uv.steps:>6}  {uv.current:>12.6g}")

# ----------------------------------------------------------------------
# Expression parser
# ----------------------------------------------------------------------
def _decode_kw_value(enc: str) -> Any:
    try: decoded = bytes.fromhex(enc).decode()
    except Exception: decoded = enc
    try: return int(decoded)
    except ValueError: pass
    try: return float(decoded)
    except ValueError: pass
    return decoded

def _encode_kwargs(s: str) -> str:
    pattern = re.compile(r'(\w+)\s*\(([^()]*?)\)', re.DOTALL)
    def _repl(m: re.Match) -> str:
        fname    = m.group(1)
        raw_args = m.group(2)
        pos_args, kw_parts = [], []
        for part in raw_args.split(","):
            part = part.strip()
            if "=" in part and not part.startswith("_kw_"):
                k, v = part.split("=", 1)
                clean_v = v.strip().strip("'\"")
                token = clean_v.encode().hex()
                kw_parts.append(f"_kw_{k.strip()}__{token}")
            else:
                pos_args.append(part)
        return f"{fname}({', '.join(pos_args + kw_parts)})"
    prev = None
    while prev != s: prev = s; s = pattern.sub(_repl, s)
    return s

def _build_sympy_namespace(field_names: Optional[Set[str]] = None) -> Dict[str, Any]:
    ns: Dict[str, Any] = {}
    field_names = field_names or set()
    shadows = _RESERVED_SYMBOLS & field_names
    if shadows:
        _rank0_print(
            f"  ⚠  Field name(s) {sorted(shadows)} shadow reserved coordinate symbols.\n"
            "     SymPy may simplify expressions involving these symbols unexpectedly.\n"
            "     Consider renaming the field to avoid ambiguity."
        )
    for sym in _RESERVED_SYMBOLS:
        if sym not in field_names: ns[sym] = sp.Symbol(sym)
    for name in field_names: ns[name] = sp.Symbol(name)
    for name in _USER_VARS:
        if name not in ns: ns[name] = sp.Symbol(name)
    seen: Set[str] = set()
    for alias, canonical in _ALIAS.items():
        if alias == "hessian" or canonical == "hessian":
            ns["_hessian"] = sp.Function("_hessian")
            seen.add("_hessian")
            continue
        if alias not in seen:
            ns[alias] = sp.Function(alias); seen.add(alias)
        if canonical not in seen:
            ns[canonical] = sp.Function(canonical); seen.add(canonical)
    ns["pi"] = sp.pi
    ns["E"]  = sp.E
    return ns

def parse_expression(expr_str: str, field_names: Optional[Set[str]] = None) -> sp.Expr:
    cleaned = _encode_kwargs(expr_str.strip())
    ns = _build_sympy_namespace(field_names=field_names)
    for kf in re.findall(r'_kw_\w+', cleaned): ns[kf] = sp.Symbol(kf)
    try: return sp.sympify(cleaned, locals=ns)
    except Exception as exc:
        raise ValueError(f"Could not parse '{expr_str}': {exc}\nProcessed form: '{cleaned}'") from exc

# ----------------------------------------------------------------------
# AST → ops.Graph compiler
# ----------------------------------------------------------------------
@dataclass
class _CompileCtx:
    graph:     stenpy.Graph
    dx:        float
    boundary:  str
    cse_cache: Dict[Tuple, str]  = dc_field(default_factory=dict)
    src_id:    str               = ""
    warnings:  List[str]         = dc_field(default_factory=list)
    field_map: Dict[str, str]    = dc_field(default_factory=dict)

def _decode_kwparams(func_name: str, args: Tuple) -> Tuple[str, Dict[str, Any], List]:
    real_args: List    = []
    kwargs:    Dict[str, Any] = {}
    for a in args:
        s = str(a)
        if s.startswith("_kw_"):
            rest = s[4:]
            sep  = rest.index("__")
            key      = rest[:sep]
            val_str  = rest[sep + 2:]
            kwargs[key] = _decode_kw_value(val_str)
        else:
            real_args.append(a)
    canonical = _ALIAS.get(func_name.lower(), func_name.lower())
    return canonical, kwargs, real_args

def _op_params(canonical: str, dx: float, boundary: str) -> Dict[str, Any]:
    if canonical in _NO_STENCIL_OPS: return {}
    return {"dx": dx, "boundary": boundary}

def _compile_node(expr: sp.Expr, ctx: _CompileCtx) -> str:
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name in ctx.field_map: return ctx.field_map[name]
        if name in ("f", "g", "h", "u", "v"): return ctx.src_id
        ctx.warnings.append(f"Unknown symbol '{name}' — falling back to primary field.")
        return ctx.src_id
    if isinstance(expr, (sp.Number, sp.Integer, sp.Float, sp.Rational)):
        val = float(expr)
        key = ("_scalar", val)
        if key in ctx.cse_cache: return ctx.cse_cache[key]
        nid = ctx.graph.add("_constant", (), {"value": torch.tensor(val, dtype=torch.float64)})
        ctx.cse_cache[key] = nid
        return nid
    if isinstance(expr, sp.core.function.AppliedUndef):
        func_name = type(expr).__name__
        if func_name == "_hessian":
            canonical = "hessian"
            extra_kw = {}
            real_args = list(expr.args)
        else:
            canonical, extra_kw, real_args = _decode_kwparams(func_name, expr.args)
            def _get_ndim(node_id: str, fallback: int = 3) -> int:
                visited = set()
                queue = [node_id]
                while queue:
                    nid = queue.pop()
                    if nid in visited: continue
                    visited.add(nid)
                    node = ctx.graph._nodes.get(nid)
                    if node is None: continue
                    if node.op_name == "_constant":
                        val = node.params.get("value")
                        if isinstance(val, torch.Tensor) and val.ndim > 0: return val.ndim
                    queue.extend(node.input_ids)
                return fallback
            def _grad_components(input_id: str, ndim: int, dx: float, boundary: str) -> List[str]:
                ids = []
                for d in range(ndim):
                    p      = {"dim": d, "dx": float(dx), "boundary": boundary}
                    frozen = tuple(sorted(p.items()))
                    key    = ("gradient", (input_id,), frozen)
                    if key in ctx.cse_cache: ids.append(ctx.cse_cache[key])
                    else:
                        gid = ctx.graph.add("gradient", (input_id,), p)
                        ctx.cse_cache[key] = gid
                        ids.append(gid)
                return ids
            child_ids = tuple(_compile_node(a, ctx) for a in real_args)
            params: Dict[str, Any] = _op_params(canonical, ctx.dx, ctx.boundary)
            params.update(extra_kw)
            if len(child_ids) == 1:
                child_node = ctx.graph._nodes.get(child_ids[0])
                if child_node:
                    rule_key = (canonical, child_node.op_name)
                    if rule_key in _SIMPLIFY_RULES:
                        replacement = _SIMPLIFY_RULES[rule_key]
                        ctx.warnings.append(f"Simplified {canonical}({child_node.op_name}(f)) → {replacement}(f)")
                        canonical = replacement
                        child_ids = child_node.input_ids
            frozen  = tuple(sorted((k, v) for k, v in params.items() if isinstance(v, (int, float, str, bool))))
            cse_key = (canonical, child_ids, frozen)
            if cse_key in ctx.cse_cache: return ctx.cse_cache[cse_key]
            if canonical not in stenpy.OP_REGISTRY:
                raise ValueError(f"Unknown operator '{canonical}' (from '{func_name}').\nAvailable: {sorted(stenpy.OP_REGISTRY.keys())}")
            nid = ctx.graph.add(canonical, child_ids, params)
            ctx.cse_cache[cse_key] = nid
            return nid
        if isinstance(expr, sp.Add):
            ids = [_compile_node(o, ctx) for o in expr.args]
            result_id = ids[0]
            for nxt in ids[1:]:
                cse_key = ("add", (result_id, nxt), ())
                if cse_key in ctx.cse_cache: result_id = ctx.cse_cache[cse_key]
                else:
                    result_id = ctx.graph.add("add", (result_id, nxt), {})
                    ctx.cse_cache[cse_key] = result_id
            return result_id
        if isinstance(expr, sp.Mul):
            operands = list(expr.args)
            if sp.Integer(-1) in operands and len(operands) == 2:
                other    = next(o for o in operands if o != sp.Integer(-1))
                child_id = _compile_node(other, ctx)
                cse_key  = ("neg", (child_id,), ())
                if cse_key in ctx.cse_cache: return ctx.cse_cache[cse_key]
                nid = ctx.graph.add("neg", (child_id,), {})
                ctx.cse_cache[cse_key] = nid
                return nid
            ids = [_compile_node(o, ctx) for o in operands]
            result_id = ids[0]
            for nxt in ids[1:]:
                cse_key = ("mul", (result_id, nxt), ())
                if cse_key in ctx.cse_cache: result_id = ctx.cse_cache[cse_key]
                else:
                    result_id = ctx.graph.add("mul", (result_id, nxt), {})
                    ctx.cse_cache[cse_key] = result_id
            return result_id
        if isinstance(expr, sp.Pow):
            base_id = _compile_node(expr.args[0], ctx)
            exp_val = float(expr.args[1])
            if abs(exp_val - 0.5) < 1e-9:
                cse_key = ("sqrt", (base_id,), ())
                if cse_key in ctx.cse_cache: return ctx.cse_cache[cse_key]
                nid = ctx.graph.add("sqrt", (base_id,), {})
                ctx.cse_cache[cse_key] = nid
                return nid
            n = int(exp_val)
            if abs(exp_val - n) < 1e-9 and abs(n) <= 16:
                if n == 0:
                    one = ctx.graph.add("_constant", (), {"value": torch.tensor(1.0, dtype=torch.float64)})
                    return one
                if n < 0:
                    pos_id = _compile_node(sp.Pow(expr.args[0], sp.Integer(-n)), ctx)
                    one = ctx.graph.add("_constant", (), {"value": torch.tensor(1.0, dtype=torch.float64)})
                    return ctx.graph.add("div", (one, pos_id), {})
                result_id = base_id
                for _ in range(n - 1):
                    cse_key = ("mul", (result_id, base_id), ())
                    if cse_key in ctx.cse_cache: result_id = ctx.cse_cache[cse_key]
                    else:
                        result_id = ctx.graph.add("mul", (result_id, base_id), {})
                        ctx.cse_cache[cse_key] = result_id
                return result_id
            log_id = ctx.graph.add("log", (base_id,), {})
            const_nid = _compile_node(sp.Float(exp_val), ctx)
            mul_id = ctx.graph.add("mul", (log_id, const_nid), {})
            return ctx.graph.add("exp", (mul_id,), {})
        raise ValueError(f"Unsupported SymPy node {type(expr).__name__}: {expr}\nUse recognised operators and arithmetic (+, *, **).")

def compile_expression(
    expr_str:  str,
    dx:        float = 1.0,
    boundary:  str   = "neumann",
    field_map: Optional[Dict[str, Any]] = None,
) -> Tuple[stenpy.Graph, str, List[str], Dict[str, str], sp.Expr]:
    if field_map is None: field_map = {"f": torch.zeros(1, dtype=torch.float64)}
    field_names = set(field_map.keys())
    sympy_expr  = parse_expression(expr_str, field_names=field_names)
    g   = stenpy.Graph()
    ctx = _CompileCtx(graph=g, dx=dx, boundary=boundary)
    for name, tensor in field_map.items():
        nid = g.add("_constant", (), {"value": tensor})
        ctx.field_map[name] = nid
    ctx.src_id = ctx.field_map.get("f", next(iter(ctx.field_map.values())))
    sink_id    = _compile_node(sympy_expr, ctx)
    return g, sink_id, ctx.warnings, dict(ctx.field_map), sympy_expr

# ----------------------------------------------------------------------
# HPC helpers
# ----------------------------------------------------------------------
def _tensor_gb(shape: Tuple) -> float:
    try: return math.prod(int(s) for s in shape) * 8 / 1024**3
    except Exception: return 0.0

def _gpu_free_gb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available(): return float("inf")
    return _free_vram_bytes(device) / 1024**3

def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        print("  Device: CPU  (no CUDA GPU found)"); return torch.device("cpu")
    n = torch.cuda.device_count()
    print(f"\n{'─'*62}\nSELECT COMPUTE DEVICE\n{'─'*62}")
    print("    [0]  CPU")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"    [{i+1}]  CUDA:{i}  {p.name}  {p.total_memory/1024**3:.1f} GB VRAM  {p.multi_processor_count} SMs")
    print(f"  {'─'*62}")
    while True:
        try: raw = input("  Choice [0 = CPU]: ").strip() or "0"
        except (EOFError, KeyboardInterrupt): return torch.device("cpu")
        try: idx = int(raw)
        except ValueError: print(f"  ✗ Enter 0–{n}"); continue
        if idx == 0: return torch.device("cpu")
        if 1 <= idx <= n: return torch.device(f"cuda:{idx - 1}")
        print(f"  ✗ Enter 0–{n}")

def _select_device_hpc() -> torch.device:
    if not torch.cuda.is_available(): return torch.device("cpu")
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0: return torch.device("cpu")
    local_rank = None
    for var_name in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "PMI_LOCAL_RANK", "JSM_NAMESPACE_LOCAL_RANK", "MPI_LOCALRANKID"):
        val = os.environ.get(var_name)
        if val is not None:
            try: local_rank = int(val.strip()); break
            except (ValueError, TypeError): continue
    if local_rank is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible not in ("-1", ""):
            try:
                ids = [x.strip() for x in visible.split(",") if x.strip()]
                if ids: local_rank = _HPC_RANK % len(ids)
            except Exception: pass
    if local_rank is None: local_rank = _HPC_RANK % n_gpus
    return torch.device(f"cuda:{local_rank % n_gpus}")

def _select_boundary() -> str:
    opts = {"1": "neumann", "2": "dirichlet", "3": "periodic", "4": "reflect"}
    print("\nBoundary condition:")
    print("    [1]  Neumann    — replicate edge  ← default")
    print("    [2]  Dirichlet  — zero pad")
    print("    [3]  Periodic   — wrap around")
    print("    [4]  Reflect")
    try: raw = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt): return "neumann"
    return opts.get(raw, "neumann")

# ----------------------------------------------------------------------
# VRAM management
# ----------------------------------------------------------------------
def _flush_vram(device: torch.device, verbose: bool = True) -> None:
    if device.type != "cuda" or not torch.cuda.is_available(): return
    before_alloc = torch.cuda.memory_allocated(device)
    before_res   = torch.cuda.memory_reserved(device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    after_alloc  = torch.cuda.memory_allocated(device)
    after_res    = torch.cuda.memory_reserved(device)
    freed_alloc  = (before_alloc - after_alloc) / 1024**3
    freed_res    = (before_res   - after_res)   / 1024**3
    if verbose and (not _IS_HPC or _HPC_RANK == 0):
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / 1024**3
        pct   = after_alloc / max(props.total_memory, 1) * 100
        print(f"\n♻  VRAM flush")
        print(f"     allocated  : freed {freed_alloc:.3f} GB  │  now {after_alloc/1024**3:.3f} GB")
        print(f"     reserved   : freed {freed_res:.3f} GB  │  now {after_res/1024**3:.3f} GB")
        print(f"     gauge      : {_vram_bar(device)}")

def _prompt_vram_flush(device: torch.device) -> None:
    if _IS_HPC or device.type != "cuda": return
    used_gb = torch.cuda.memory_allocated(device) / 1024**3
    if used_gb < 0.1: return
    props = torch.cuda.get_device_properties(device)
    pct = used_gb / max(props.total_memory / 1024**3, 1e-9) * 100
    sev = _sev_label(pct).strip()
    print(f"\n♻  VRAM holds {used_gb:.2f} GB  [{sev}]")
    try: ans = input("  Flush VRAM now? [Y/n] ❯ ").strip().lower()
    except (EOFError, KeyboardInterrupt): return
    if ans in ("", "y", "yes"): _flush_vram(device, verbose=True)
    else: print("  VRAM kept — remember to flush before large operations.")

# ----------------------------------------------------------------------
# VRAM-LEAK FIX B: aggressive VRAM release after each run
# ----------------------------------------------------------------------
def _release_graph_tensors(graph: stenpy.Graph, device: torch.device) -> None:
    if device.type != "cuda": return
    for nid in list(graph._nodes.keys()):
        node = graph._nodes.get(nid)
        if node is None: continue
        if node.op_name == "_constant":
            val = node.params.get("value")
            if isinstance(val, torch.Tensor) and val.device.type == "cuda":
                cpu_stub = torch.zeros(val.shape, dtype=val.dtype, device="cpu")
                new_node = stenpy.GraphNode(id=node.id, op_name=node.op_name, input_ids=node.input_ids, params={"value": cpu_stub})
                graph._nodes[nid] = new_node
                del val

def _full_vram_release(device: torch.device, graph: Optional[stenpy.Graph] = None) -> None:
    if device.type != "cuda":
        gc.collect()
        return
    if graph is not None: _release_graph_tensors(graph, device)
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------
# Multi-field domain merging
# ----------------------------------------------------------------------
def _merge_domain(
    spacings: Dict[str, Tuple[float, ...]],
    origins:  Dict[str, Tuple[float, ...]],
    shapes:   Dict[str, Tuple[int, ...]],
) -> Tuple[float, Tuple[float, ...], Tuple[float, ...]]:
    if not spacings: return 1.0, (0.0,), (1.0,)
    all_sp = [s for sp_tuple in spacings.values() for s in sp_tuple]
    dx     = float(max(all_sp)) if all_sp else 1.0
    ndims  = min(len(sp) for sp in spacings.values())
    merged_origin = tuple(max(origins[n][d] for n in origins if d < len(origins[n])) for d in range(ndims))
    ends: List[List[float]] = []
    for name, sp_tuple in spacings.items():
        sh   = shapes.get(name, ())
        n_sp = min(len(sp_tuple), len(sh), ndims)
        ends.append([origins[name][d] + (sh[d] - 1) * sp_tuple[d] for d in range(n_sp)])
    merged_end = tuple(min(e[d] for e in ends) for d in range(ndims))
    return dx, merged_origin, merged_end

# ----------------------------------------------------------------------
# NUMA-aware pinned buffer allocation
# ----------------------------------------------------------------------
def _numa_node_for_device(device: torch.device) -> Optional[int]:
    if device.type != "cuda" or not torch.cuda.is_available(): return None
    try:
        dev_idx = device.index or 0
        numa_path = f"/sys/bus/pci/devices/{torch.cuda.get_device_properties(dev_idx).pci_bus_id.lower()}/numa_node"
        with open(numa_path) as fh:
            node = int(fh.read().strip())
            return node if node >= 0 else None
    except Exception: return None

def _alloc_pinned_on_numa(shape: Tuple[int, ...], numa_node: Optional[int]) -> torch.Tensor:
    if numa_node is None:
        try: return torch.empty(shape, dtype=torch.float64, pin_memory=True)
        except Exception: return torch.empty(shape, dtype=torch.float64)
    try:
        buf = torch.empty(shape, dtype=torch.float64, pin_memory=True)
        try:
            import ctypes, ctypes.util
            libnuma = ctypes.CDLL(ctypes.util.find_library("numa") or "libnuma.so.1")
            nbytes  = buf.nbytes
            ptr     = buf.data_ptr()
            nodemask = ctypes.c_ulong(1 << numa_node)
            libnuma.mbind(ctypes.c_void_p(ptr), ctypes.c_ulong(nbytes), ctypes.c_int(2), ctypes.byref(nodemask), ctypes.c_ulong(64), ctypes.c_uint(2))
        except Exception: pass
        return buf
    except Exception: return torch.empty(shape, dtype=torch.float64)

# ----------------------------------------------------------------------
# HDF5 chunk-aligned read helpers
# ----------------------------------------------------------------------
def _hdf5_chunk_size(ds: "_h5py.Dataset") -> int:
    if ds.chunks is not None and len(ds.chunks) > 0: return int(ds.chunks[0])
    return 1

def _hdf5_output_chunks(shape: Tuple[int, ...], preferred_rows: int, dtype: Any = np.float64) -> Optional[Tuple[int, ...]]:
    if not shape:
        return None
    shape = tuple(int(s) for s in shape)
    total_rows = max(1, shape[0])
    row_elems = math.prod(shape[1:]) if len(shape) > 1 else 1
    row_bytes = row_elems * np.dtype(dtype).itemsize
    hdf5_hard_limit = (4 * 1024**3) - 1
    if row_bytes > hdf5_hard_limit:
        raise ValueError(
            f"One output row is {row_bytes/1024**3:.2f} GB, which exceeds HDF5's "
            "single-chunk limit. Reduce the output dimensionality or tile another axis."
        )
    max_chunk_bytes = max(row_bytes, min(_HDF5_MAX_CHUNK_BYTES, hdf5_hard_limit))
    max_rows_by_bytes = max(1, int(max_chunk_bytes // max(row_bytes, 1)))
    rows = max(1, min(int(preferred_rows or max_rows_by_bytes), total_rows, max_rows_by_bytes))
    return (rows,) + shape[1:]

def _ensure_output_capacity(out_path: str, shape: Tuple[int, ...], dtype: Any = np.float64) -> None:
    try:
        out_dir = Path(out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        needed = math.prod(int(s) for s in shape) * np.dtype(dtype).itemsize
        existing = Path(out_path).stat().st_size if Path(out_path).exists() else 0
        free = shutil.disk_usage(out_dir).free + existing
        required = int(needed * 1.03) + 256 * 1024**2
        _logger.info(
            f"Output capacity check: need {needed/1024**3:.2f} GB "
            f"(with margin {required/1024**3:.2f} GB), available {free/1024**3:.2f} GB"
        )
        if free < required:
            raise OSError(
                f"Not enough free disk space for output {out_path}: need about "
                f"{required/1024**3:.1f} GB, available {free/1024**3:.1f} GB."
            )
    except OSError:
        raise
    except Exception as exc:
        _logger.warning(f"Could not verify output disk capacity for {out_path}: {exc}")

def _align_to_chunk(row: int, chunk_size: int, total: int, align_up: bool) -> int:
    if chunk_size <= 1: return row
    if align_up: return min(total, int(math.ceil(row / chunk_size)) * chunk_size)
    else: return int(math.floor(row / chunk_size)) * chunk_size

def _compute_chunk_rows_aligned(target_rows: int, chunk_size: int, total_rows: int) -> int:
    if chunk_size <= 1 or target_rows <= 0: return max(1, target_rows)
    aligned = int(math.ceil(target_rows / chunk_size)) * chunk_size
    return min(aligned, total_rows)

# ----------------------------------------------------------------------
# Memory helpers
# ----------------------------------------------------------------------
def _free_ram_bytes() -> int:
    try: return int(psutil.virtual_memory().available)
    except Exception: pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"): return int(line.split()[1]) * 1024
    except Exception: pass
    return 512 * 1024 * 1024

def _free_vram_bytes(device: torch.device) -> int:
    if device.type != "cuda" or not torch.cuda.is_available(): return 0
    props = torch.cuda.get_device_properties(device)
    reserved   = torch.cuda.memory_reserved(device)
    allocated  = torch.cuda.memory_allocated(device)
    reclaimable = (reserved - allocated) // 2
    free = props.total_memory - reserved + reclaimable
    return max(0, free)

_STREAM_GLOBAL_OPS: Set[str] = {
    "norm_l2", "min_max", "covariance", "correlation", "surface_integral",
    "fft", "ifft", "spectral_gradient", "spectral_laplacian", "distance_transform",
}

_ROW_STREAM_HEAVY_OPS: Set[str] = {
    "gradient_nd", "hessian", "mean_curvature", "surface_normals",
    "material_derivative",
}

def _dim_mentions_chunk_dim(value: Any, chunk_dim: int = 0) -> bool:
    if value is None: return True
    if isinstance(value, (tuple, list, set)): return any(_dim_mentions_chunk_dim(v, chunk_dim) for v in value)
    try: return int(value) == chunk_dim
    except (TypeError, ValueError): return True

def _stream_unsafe_reason(node: Any, chunk_dim: int = 0) -> Optional[str]:
    op = node.op_name
    if op in _STREAM_GLOBAL_OPS: return f"{op} needs global data, not independent HDF5 chunks"
    if op in {"sum", "mean", "variance", "entropy"}:
        if _dim_mentions_chunk_dim(node.params.get("dim"), chunk_dim): return f"{op} reduces across the streamed dimension"
    if op == "integrate":
        if _dim_mentions_chunk_dim(node.params.get("dims"), chunk_dim): return "integrate reduces across the streamed dimension"
    if op == "cumulative_integral":
        if _dim_mentions_chunk_dim(node.params.get("dim", 0), chunk_dim): return "cumulative_integral needs state across chunks"
    return None

def _node_row_halo(node: Any, chunk_dim: int = 0) -> int:
    meta = stenpy.OP_METADATA.get(node.op_name, {})
    radius = int(meta.get("stencil_radius", 0) or 0)
    if radius <= 0: return 0
    exchange_dims = meta.get("exchange_dims")
    if exchange_dims is None or chunk_dim in exchange_dims: return radius
    return 0

def _graph_row_halo(graph: stenpy.Graph, sink_id: str, chunk_dim: int = 0) -> int:
    memo: Dict[str, int] = {}
    def _walk(node_id: str) -> int:
        if node_id in memo: return memo[node_id]
        node = graph._nodes[node_id]
        reason = _stream_unsafe_reason(node, chunk_dim=chunk_dim)
        if reason is not None:
            raise ValueError(f"Expression is not safe for direct chunk streaming: {reason}. Use an eager/small input path.")
        child_halo = max((_walk(dep) for dep in node.input_ids), default=0)
        halo = child_halo + _node_row_halo(node, chunk_dim=chunk_dim)
        memo[node_id] = halo
        return halo
    return _walk(sink_id)

def _graph_ops_from_sink(graph: stenpy.Graph, sink_id: str) -> Set[str]:
    seen: Set[str] = set()
    ops:  Set[str] = set()
    def _walk(node_id: str) -> None:
        if node_id in seen: return
        seen.add(node_id)
        node = graph._nodes[node_id]
        ops.add(node.op_name)
        for dep in node.input_ids: _walk(dep)
    _walk(sink_id)
    return ops

def _row_stream_reasons(
    graph: stenpy.Graph,
    sink_id: str,
    bytes_per_out_row: int,
    output_multiplier: float,
) -> List[str]:
    reasons: List[str] = []
    if bytes_per_out_row >= _ROW_STREAM_THRESHOLD:
        reasons.append(
            f"output row = {bytes_per_out_row/1024**2:.1f} MB >= "
            f"threshold {_ROW_STREAM_THRESHOLD/1024**2:.0f} MB"
        )
    if output_multiplier > 3.0:
        reasons.append(f"expansion {output_multiplier:.1f}x > 3.0x")
    heavy_ops = sorted(_graph_ops_from_sink(graph, sink_id) & _ROW_STREAM_HEAVY_OPS)
    if heavy_ops:
        reasons.append(f"heavy op(s): {', '.join(heavy_ops)}")
    return reasons

def _read_halo_rows(ds: "_h5py.Dataset", row_start: int, row_end: int, halo: int, periodic: bool) -> Tuple[np.ndarray, int]:
    total = int(ds.shape[0])
    if halo <= 0: return ds[row_start:row_end], 0
    if total <= 0: raise ValueError("Cannot stream an empty HDF5 dataset")
    if periodic:
        wanted = (row_end - row_start) + 2 * halo
        parts: List[np.ndarray] = []
        pos = row_start - halo
        remaining = wanted
        while remaining > 0:
            wrapped = pos % total
            run = min(remaining, total - wrapped)
            parts.append(ds[wrapped:wrapped + run])
            pos += run; remaining -= run
        arr = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        return arr, halo
    read_start = max(0, row_start - halo)
    read_end = min(total, row_end + halo)
    return ds[read_start:read_end], row_start - read_start

def _safe_mm_clear(mm: Any) -> None:
    try: mm.clear_pool()
    except Exception: pass
    try:
        _lock = getattr(mm, '_lock', None) or getattr(mm, '_live_lock', None)
        _live = getattr(mm, '_live', None)
        if _live is not None:
            if _lock is not None:
                with _lock: _live.clear()
            else: _live.clear()
    except Exception: pass

# ----------------------------------------------------------------------
# Robust Chunk Sizing with Huge Safety Buffers
# ----------------------------------------------------------------------
def _compute_chunk_rows(
    full_shape:        Tuple[int, ...],
    device:            torch.device,
    n_fields:          int = 1,
    output_multiplier: float = 1.0,
    graph_overhead:    float = 2.0,
) -> int:
    if len(full_shape) == 0: return 1
    elements_per_row = math.prod(full_shape[1:]) if len(full_shape) > 1 else 1
    bytes_per_row_in  = elements_per_row * 8
    bytes_per_row_out = int(bytes_per_row_in * max(1.0, output_multiplier))
    dim0     = full_shape[0]
    hard_cap = max(_MIN_CHUNK_ROWS, int(dim0 * _MAX_CHUNK_FRAC))
    VRAM_HEADROOM_BYTES = max(2 * 1024**3, int(torch.cuda.get_device_properties(device).total_memory * 0.20)) if device.type == "cuda" and torch.cuda.is_available() else 0
    RAM_HEADROOM_BYTES  = max(4 * 1024**3, int(psutil.virtual_memory().total * 0.15))

    if device.type == "cuda" and torch.cuda.is_available():
        free_vram = _free_vram_bytes(device)
        usable_vram = max(0, free_vram - VRAM_HEADROOM_BYTES)
        vram_per_row = (bytes_per_row_in * n_fields * (1.0 + graph_overhead) + bytes_per_row_out)
        vram_rows = max(_MIN_CHUNK_ROWS, int(usable_vram / max(vram_per_row, 1)))

        free_ram = _free_ram_bytes()
        usable_ram = max(0, free_ram - RAM_HEADROOM_BYTES)
        halo_est   = 3
        halo_fixed = 2 * halo_est * (_PIPE_DEPTH + n_fields) * bytes_per_row_in
        ram_per_row = (_PIPE_DEPTH + n_fields) * bytes_per_row_in + 3 * bytes_per_row_out
        ram_rows = max(_MIN_CHUNK_ROWS, int(max(0, usable_ram - halo_fixed) / max(ram_per_row, 1)))

        rows = min(vram_rows, ram_rows, hard_cap)
    else:
        free_ram = _free_ram_bytes()
        usable_ram = max(0, free_ram - RAM_HEADROOM_BYTES)
        halo_est    = 3
        halo_fixed  = 2 * halo_est * (1 + n_fields) * bytes_per_row_in
        ram_per_row = (1 + n_fields) * bytes_per_row_in + 3 * bytes_per_row_out
        rows = max(_MIN_CHUNK_ROWS, min(int(max(0, usable_ram - halo_fixed) / max(ram_per_row, 1)), hard_cap))

    if dim0 > 100 and output_multiplier > 2.0:
        rows = min(rows, 2)

    if _NERD:
        _rank0_print(f"  ○  chunk_rows: computed {rows} (vram_rows={locals().get('vram_rows', 'N/A')}, ram_rows={locals().get('ram_rows', 'N/A')}, hard_cap={hard_cap})")

    return max(_MIN_CHUNK_ROWS, rows)

def _adapt_chunk_rows(
    current_rows:   int,
    full_shape:     Tuple[int, ...],
    device:         torch.device,
    n_fields:       int,
    output_multiplier: float,
    halo:           int = 0,
) -> int:
    elements_per_row  = math.prod(full_shape[1:]) if len(full_shape) > 1 else 1
    bytes_per_row_in  = elements_per_row * 8
    
    if device.type == "cuda" and torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        pct   = used / max(total, 1)
        if pct >= _VRAM_SHRINK_TRIGGER:
            new_rows = _compute_chunk_rows(full_shape, device, n_fields, output_multiplier)
            min_halo_rows = max(1, 2 * halo + 1)
            new_rows = max(new_rows, min_halo_rows)
            if new_rows < current_rows:
                msg = f"VRAM pressure {pct*100:.0f}% (used {used/1024**3:.2f}/{total/1024**3:.1f} GB) — shrinking chunk_rows {current_rows} → {new_rows}"
                if _NERD or (not _IS_HPC or _HPC_RANK == 0):
                    _rank0_print(f"\n▲  {msg}")
                _logger.warning(msg)
                return new_rows
    try:
        pct = psutil.virtual_memory().percent / 100.0
        ram_available = psutil.virtual_memory().available / 1024**3
    except Exception:
        pct = 0.0
        ram_available = 0.0
    if pct >= _RAM_SHRINK_TRIGGER:
        new_rows = _compute_chunk_rows(full_shape, device, n_fields, output_multiplier)
        min_halo_rows = max(1, 2 * halo + 1)
        new_rows = max(new_rows, min_halo_rows)
        if new_rows < current_rows:
            msg = f"RAM pressure {pct*100:.0f}% (available {ram_available:.2f} GB) — shrinking chunk_rows {current_rows} → {new_rows}"
            if _NERD or (not _IS_HPC or _HPC_RANK == 0):
                _rank0_print(f"\n▲  {msg}")
            _logger.warning(msg)
            return new_rows
    return current_rows

# ----------------------------------------------------------------------
# Pre-compiled template graph
# ----------------------------------------------------------------------
@dataclass
class _TemplateGraph:
    graph:          stenpy.Graph
    sink_id:        str
    field_node_ids: Dict[str, str]

    def update_chunk_tensors(self, chunk_tensors: Dict[str, torch.Tensor]) -> None:
        for field_name, tensor in chunk_tensors.items():
            node_id = self.field_node_ids.get(field_name)
            if node_id is not None and node_id in self.graph._nodes:
                node = self.graph._nodes[node_id]
                old_params = node.params
                new_node = type(node)(id=node.id, op_name=node.op_name, input_ids=node.input_ids, params={"value": tensor})
                self.graph._nodes[node_id] = new_node
                old_tensor = old_params.get("value")
                del old_params
                if isinstance(old_tensor, torch.Tensor): del old_tensor

    def clear_gpu_tensors(self, device: torch.device) -> None:
        if device.type != "cuda": return
        _release_graph_tensors(self.graph, device)

    def make_chunk_graph(self, chunk_tensors: Dict[str, torch.Tensor]) -> stenpy.Graph:
        self.update_chunk_tensors(chunk_tensors)
        return self.graph

def _hdf5_open_cached(path: str, mode: str = "r") -> "_h5py.File":
    propfaid = _h5py.h5p.create(_h5py.h5p.FILE_ACCESS)
    propfaid.set_cache(0, _HDF5_RDCC_NSLOTS, _HDF5_RDCC_NBYTES, _HDF5_RDCC_W0)
    fid = _h5py.h5f.open(path.encode(), _h5py.h5f.ACC_RDONLY if mode == "r" else _h5py.h5f.ACC_RDWR, fapl=propfaid)
    return _h5py.File(fid)

def _pad_halo_boundary(arr: np.ndarray, pre_deficit: int, post_deficit: int, boundary: str) -> Tuple[np.ndarray, int]:
    orig = arr
    orig_nrows = orig.shape[0]
    extra_pre = 0
    if pre_deficit > 0:
        if boundary == "dirichlet": pre_pad = np.zeros((pre_deficit,) + orig.shape[1:], dtype=orig.dtype)
        elif boundary == "reflect" and orig_nrows > 1:
            n_avail = min(pre_deficit, orig_nrows - 1)
            mirrored = orig[1 : n_avail + 1][::-1].copy()
            short = pre_deficit - mirrored.shape[0]
            if short > 0:
                filler = np.zeros((short,) + orig.shape[1:], dtype=orig.dtype)
                pre_pad = np.concatenate([filler, mirrored], axis=0)
            else: pre_pad = mirrored
        else: pre_pad = np.broadcast_to(orig[:1], (pre_deficit,) + orig.shape[1:]).copy()
        arr = np.concatenate([pre_pad, orig], axis=0)
        extra_pre = pre_deficit
    if post_deficit > 0:
        if boundary == "dirichlet": post_pad = np.zeros((post_deficit,) + orig.shape[1:], dtype=orig.dtype)
        elif boundary == "reflect" and orig_nrows > 1:
            n_avail = min(post_deficit, orig_nrows - 1)
            start = -(n_avail + 1)
            end = -1
            mirrored = orig[start:end][::-1].copy() if n_avail > 0 else orig[-1:][::-1].copy()
            short = post_deficit - mirrored.shape[0]
            if short > 0:
                filler = np.zeros((short,) + orig.shape[1:], dtype=orig.dtype)
                post_pad = np.concatenate([mirrored, filler], axis=0)
            else: post_pad = mirrored
        else: post_pad = np.broadcast_to(orig[-1:], (post_deficit,) + orig.shape[1:]).copy()
        arr = np.concatenate([arr, post_pad], axis=0)
    return arr, extra_pre

def _validate_chunk_size(chunk_rows: int, halo: int, total_rows: int, row_start: int) -> int:
    min_halo_rows = max(1, 2 * halo + 1)
    if row_start + chunk_rows >= total_rows: return max(1, chunk_rows)
    if chunk_rows < min_halo_rows: return min_halo_rows
    return chunk_rows

def _hdf5_safe_row_stream(
    expr_str: str, field_paths: Dict[str, str], scalar_map: Dict[str, torch.Tensor],
    dx: float, boundary: str, out_path: str, spacing: Tuple, origin: Tuple,
    device: torch.device, out_trailing: Tuple[int, ...], halo: int, total_rows: int,
    primary_shape: Tuple[int, ...], verbose: bool = True,
) -> Dict[str, Any]:
    if verbose:
        out_full_gb = _tensor_gb((total_rows,) + out_trailing)
        in_gb       = _tensor_gb(primary_shape)
        print(f"\n▶ Row-stream mode  (output too large for chunk stream)")
        print(f"    input  : {primary_shape}  {in_gb:.3f} GB")
        print(f"    output : ({total_rows},)+{out_trailing}  {out_full_gb:.3f} GB")
        print(f"    halo   : {halo} rows")
        if device.type == "cuda": print(f"    {_vram_bar(device)}")

    handles:    Dict[str, "_h5py.Dataset"] = {}
    hdf5_files: List["_h5py.File"]        = []

    try:
        for name, path in field_paths.items():
            try: f = _hdf5_open_cached(path, "r")
            except Exception: f = _h5py.File(path, "r")
            hdf5_files.append(f)
            key = next((k for k in ("data", "field") if k in f and isinstance(f[k], _h5py.Dataset)),
                       next(k for k in f if isinstance(f[k], _h5py.Dataset)))
            handles[name] = f[key]
    except Exception:
        for _f in hdf5_files:
            try: _f.close()
            except Exception: pass
        raise

    primary_name = next(iter(field_paths))
    hdf5_chunk_sz = _hdf5_chunk_size(handles[primary_name])
    n_fields      = len(field_paths)

    probe_rows = min(max(2 * halo + 1, 7), total_rows)
    probe_fmap: Dict[str, torch.Tensor] = dict(scalar_map)
    for name, ds in handles.items():
        arr = ds[:probe_rows].astype(np.float64)
        probe_fmap[name] = torch.from_numpy(arr).to(device)

    t0_compile = time.perf_counter()
    base_graph, sink_id, warns, field_node_ids, _sympy_expr = compile_expression(
        expr_str, dx=dx, boundary=boundary, field_map=probe_fmap
    )
    t_compile = time.perf_counter() - t0_compile

    for name in list(probe_fmap.keys()):
        t = probe_fmap.pop(name)
        if isinstance(t, torch.Tensor) and t.device.type == "cuda": del t
    del probe_fmap
    _full_vram_release(device, base_graph)

    template = _TemplateGraph(graph=base_graph, sink_id=sink_id, field_node_ids=field_node_ids)

    out_full_shape = (total_rows,) + out_trailing
    _ensure_output_capacity(out_path, out_full_shape, np.float64)
    out_chunks = _hdf5_output_chunks(out_full_shape, preferred_rows=1, dtype=np.float64)
    _logger.info(
        f"Row-stream output chunks: {out_chunks} "
        f"({math.prod(out_chunks) * np.dtype(np.float64).itemsize / 1024**2:.1f} MB/chunk)"
    )
    out_f  = _h5py.File(out_path, "w")
    out_ds = out_f.create_dataset("data", shape=out_full_shape, dtype=np.float64, chunks=out_chunks)
    out_f.attrs["spacing"] = list(spacing) if spacing else [1.0]
    out_f.attrs["origin"]  = list(origin)  if origin  else [0.0]

    write_q: Queue = Queue(maxsize=2)
    _writer_stop = threading.Event()
    write_errors: List[Exception] = []
    write_times:  List[float]     = []
    out_min = float("inf"); out_max = float("-inf")
    out_sum = 0.0; out_count = 0

    def _writer() -> None:
        nonlocal out_min, out_max, out_sum, out_count
        while True:
            try:
                item = write_q.get(timeout=0.05)
            except _QueueEmpty:
                if _writer_stop.is_set():
                    break
                continue
            if item is None: write_q.task_done(); break
            r_start, r_end, arr_cpu = item
            try:
                t_w0 = time.perf_counter()
                out_ds[r_start:r_end] = arr_cpu
                if _HDF5_FLUSH_EVERY_ROWS > 0 and (r_end >= total_rows or r_end % _HDF5_FLUSH_EVERY_ROWS == 0):
                    out_f.flush()
                write_times.append(time.perf_counter() - t_w0)
                out_min    = min(out_min, float(arr_cpu.min()))
                out_max    = max(out_max, float(arr_cpu.max()))
                out_sum   += float(arr_cpu.sum())
                out_count += arr_cpu.size
            except Exception as e:
                write_errors.append(e)
            finally:
                del arr_cpu
            write_q.task_done()

    writer_t = threading.Thread(target=_writer, daemon=True, name="row-stream-writer")
    writer_t.start()

    chunk_mm = stenpy.MemoryManager()
    chunk_rt = stenpy.Runtime(chunk_mm, device=str(device), skip_pool=True)

    t0          = time.perf_counter()
    bytes_in    = 0
    row_pos     = 0
    n_iters     = 0
    read_times:    List[float] = []
    h2d_times:     List[float] = []
    compute_times: List[float] = []
    dh_times:      List[float] = []
    in_elements_per_row  = math.prod(primary_shape[1:]) if len(primary_shape) > 1 else 1
    out_elements_per_row = math.prod(out_trailing) if out_trailing else 1
    bytes_per_in_row     = in_elements_per_row * 8
    bytes_per_out_row    = out_elements_per_row * 8
    min_rows = max(1, 2 * halo + 1)

    try:
        with _make_bar(total_rows, desc="Row-streaming", unit="row", colour="yellow") as bar:
            while row_pos < total_rows:
                if time.perf_counter() - PIPELINE_START_TIME > DEADLINE_SECONDS:
                    raise TimeoutError("⏰ 15‑hour deadline reached – shutting down gracefully")

                if write_errors: raise RuntimeError(f"HDF5 write error: {write_errors[0]}")

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                gc.collect()

                remaining = total_rows - row_pos
                if remaining < min_rows:
                    row_start = max(0, total_rows - min_rows)
                    row_end   = total_rows
                else:
                    batch_rows = max(min_rows, min(remaining, min_rows))
                    row_start  = row_pos
                    row_end    = min(row_start + batch_rows, total_rows)

                n_rows_chunk = row_end - row_start
                ext_start   = max(0, row_start - halo)
                ext_end     = min(total_rows, row_end + halo)
                halo_left   = row_start - ext_start
                halo_right  = ext_end - row_end
                n_rows_ext  = ext_end - ext_start

                t_r0 = time.perf_counter()
                chunk_np: Dict[str, np.ndarray] = {}
                for name, ds in handles.items():
                    arr = ds[ext_start:ext_end]
                    if not arr.flags["C_CONTIGUOUS"]: arr = np.ascontiguousarray(arr)
                    chunk_np[name] = arr.astype(np.float64, copy=False)
                    bytes_in += chunk_np[name].nbytes
                read_times.append(time.perf_counter() - t_r0)

                t_h = time.perf_counter()
                chunk_tensors: Dict[str, torch.Tensor] = {}
                for name, arr in chunk_np.items():
                    chunk_tensors[name] = torch.from_numpy(arr).to(device)
                chunk_np.clear(); del chunk_np
                h2d_times.append(time.perf_counter() - t_h)

                template.update_chunk_tensors(chunk_tensors)
                try: chunk_mm.clear_pool()
                except Exception: pass

                t_c = time.perf_counter()
                with torch.no_grad():
                    chunk_results = chunk_rt.run(template.graph)
                if device.type == "cuda": torch.cuda.synchronize(device)
                compute_times.append(time.perf_counter() - t_c)

                out_gpu = chunk_results.get(sink_id)
                if halo > 0 and isinstance(out_gpu, torch.Tensor):
                    trim_start = halo_left
                    trim_end   = halo_left + n_rows_chunk
                    slices     = [slice(None)] * out_gpu.ndim
                    slices[0]  = slice(trim_start, trim_end)
                    out_gpu    = out_gpu[tuple(slices)]
                if not out_gpu.is_contiguous(): out_gpu = out_gpu.contiguous()

                t_d = time.perf_counter()
                if isinstance(out_gpu, torch.Tensor):
                    out_cpu = out_gpu.detach().cpu().numpy()
                else:
                    out_cpu = np.zeros((n_rows_chunk,) + out_trailing, dtype=np.float64)
                dh_times.append(time.perf_counter() - t_d)

                del chunk_tensors
                del chunk_results
                if isinstance(out_gpu, torch.Tensor): del out_gpu
                template.clear_gpu_tensors(device)
                _safe_mm_clear(chunk_mm)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                gc.collect()

                write_q.put((row_start, row_end, out_cpu), timeout=30.0)
                del out_cpu
                row_pos  = row_end
                n_iters += 1
                bar.update(n_rows_chunk)

                if verbose and device.type == "cuda":
                    used_gb  = torch.cuda.memory_allocated(device) / 1024**3
                    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
                    pct      = used_gb / max(total_gb, 1e-9) * 100
                    elapsed  = time.perf_counter() - t0
                    tput_gbs = (bytes_in / 1024**3) / max(elapsed, 1e-9)
                    bar.set_postfix_str(
                        f"{tput_gbs:.2f} GB/s  VRAM {used_gb:.2f}/{total_gb:.0f}GB ({pct:.0f}%)  batch={n_rows_chunk}r",
                        refresh=False
                    )

        template.clear_gpu_tensors(device)
        try: chunk_rt.flush_vram()
        except Exception: pass
        _safe_mm_clear(chunk_mm)
        _full_vram_release(device, template.graph)

        try:
            write_q.put(None, timeout=30.0)
        except _QueueFull:
            _logger.warning("Unable to enqueue writer shutdown sentinel; queue is full")
            _writer_stop.set()
        writer_t.join(timeout=60.0)
        if writer_t.is_alive():
            _logger.warning("Writer thread did not finish within 60 seconds after safe row-stream shutdown")
        write_q.join()
    except TimeoutError:
        raise
    except Exception as exc:
        _logger.error("Safe row-stream failed during execution")
        _logger.error(f"Exception: {type(exc).__name__}: {exc}")
        _logger.error(traceback.format_exc())
        _writer_stop.set()
        drained_sentinel = False
        while True:
            try:
                item = write_q.get_nowait()
                write_q.task_done()
                if item is None:
                    drained_sentinel = True
            except _QueueEmpty:
                break
        if not drained_sentinel:
            try: write_q.put_nowait(None)
            except Exception: pass
        writer_t.join(timeout=60.0)
        if writer_t.is_alive():
            _logger.warning("Writer thread did not finish within 60 seconds after safe row-stream exception cleanup")
        try: out_f.close()
        except Exception: pass
        for f in hdf5_files:
            try: f.close()
            except Exception: pass
        _full_vram_release(device)
        if device.type == "cuda" and _is_cuda_oom(exc):
            _logger.warning("Retrying safe row-stream on CPU due to GPU memory failure")
            return _hdf5_safe_row_stream(
                expr_str=expr_str, field_paths=field_paths, scalar_map=scalar_map, dx=dx, boundary=boundary,
                out_path=out_path, spacing=spacing, origin=origin, device=torch.device("cpu"),
                out_trailing=out_trailing, halo=halo, total_rows=total_rows, primary_shape=primary_shape, verbose=verbose,
            )
        raise

    try: out_f.close()
    except Exception: pass
    for f in hdf5_files:
        try: f.close()
        except Exception: pass
    _full_vram_release(device)

    if write_errors: raise RuntimeError(f"HDF5 write error: {write_errors[0]}")

    elapsed   = time.perf_counter() - t0
    out_shape = (total_rows,) + out_trailing
    out_gb    = _tensor_gb(out_shape)
    out_mean  = out_sum / out_count if out_count else 0.0
    peak_vram_gb  = 0.0
    vram_total_gb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_vram_gb  = torch.cuda.max_memory_allocated(device) / 1024**3
        vram_total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3

    if verbose:
        print(f"\n✓ Row-stream complete  {elapsed:.1f}s  ({out_gb/max(elapsed,1e-9):.3f} GB/s)")
        print(f"  Output shape  : {out_shape}  ({out_gb:.3f} GB)")
        print(f"  min/max/mean  : {out_min:.6g} / {out_max:.6g} / {out_mean:.6g}")
        print(f"  Saved → {out_path}")

    return {
        "shape_out": list(out_shape), "min": out_min if out_min != float("inf") else 0.0,
        "max": out_max if out_max != float("-inf") else 0.0, "mean": out_mean,
        "elapsed_s": elapsed, "throughput_gbs": out_gb / elapsed if elapsed > 0 else 0.0,
        "output_multiplier": out_elements_per_row / max(in_elements_per_row, 1),
        "t_compile": t_compile, "t_read": sum(read_times), "t_h2d": sum(h2d_times),
        "t_compute": sum(compute_times), "t_d2h": sum(dh_times), "t_write": sum(write_times),
        "chunk_read_s": read_times, "chunk_h2d_s": h2d_times, "chunk_compute_s": compute_times,
        "chunk_dh_s": dh_times, "chunk_write_s": write_times, "n_chunks": n_iters,
        "chunk_rows": 1, "peak_vram_gb": peak_vram_gb, "vram_total_gb": vram_total_gb,
        "input_gb": bytes_in / 1024**3, "output_gb": out_gb,
    }

# ----------------------------------------------------------------------
# GPU-saturating direct-HDF5 chunked executor
# ----------------------------------------------------------------------
def _hdf5_direct_chunked_run(
    expr_str: str, field_paths: Dict[str, str], scalar_map: Dict[str, torch.Tensor],
    dx: float, boundary: str, out_path: str, spacing: Tuple, origin: Tuple,
    device: torch.device, verbose: bool = True,
) -> Dict[str, Any]:
    _logger.info(f"=== START HDF5 Direct Chunked Run ===")
    _logger.info(f"Expression: {expr_str}")
    _logger.info(f"Fields: {list(field_paths.keys())}")
    _logger.info(f"Device: {device}")
    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _logger.info(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    except Exception:
        pass
    _log_memory_state("PRE-RUN", device)

    def _stage_log(stage: str) -> None:
        try:
            _logger.info(f"--- STAGE: {stage} ---")
            _log_memory_state(stage, device)
        except Exception as _e:
            _logger.error(f"Failed to log memory state for stage {stage}: {_e}")

    if _IS_HPC: print(f"[{_HPC_RANK}] Starting HDF5 direct chunked run", flush=True)
    _stage_log("start")

    handles:    Dict[str, "_h5py.Dataset"] = {}
    hdf5_files: List["_h5py.File"]        = []
    shapes:     Dict[str, Tuple[int, ...]] = {}
    try:
        for name, path in field_paths.items():
            _hdf5_dissect(path)
            try: f = _hdf5_open_cached(path, "r")
            except Exception: f = _h5py.File(path, "r")
            hdf5_files.append(f)
            key = next((k for k in ("data", "field") if k in f and isinstance(f[k], _h5py.Dataset)), next(k for k in f if isinstance(f[k], _h5py.Dataset)))
            ds = f[key]
            handles[name] = ds
            shapes[name]  = tuple(ds.shape)
    except Exception:
        for _f in hdf5_files:
            try: _f.close()
            except Exception: pass
        raise

    primary_name  = next(iter(field_paths))
    primary_ds    = handles[primary_name]
    primary_shape = shapes[primary_name]
    total_rows    = primary_shape[0]
    n_fields      = len(field_paths)

    for name in field_paths:
        other_shape = shapes[name]
        if primary_shape[1:] != other_shape[1:]:
            for _f in hdf5_files:
                try: _f.close()
                except Exception: pass
            raise ValueError(f"Field '{name}' has trailing dims {other_shape[1:]} but primary field '{primary_name}' has {primary_shape[1:]}.")

    hdf5_chunk_size = _hdf5_chunk_size(primary_ds)

    probe_rows = min(max(7, 2 * 3 + 1), total_rows)
    probe_fmap: Dict[str, torch.Tensor] = dict(scalar_map)
    for name, ds in handles.items():
        arr = ds[:probe_rows]
        if not arr.flags["C_CONTIGUOUS"]: arr = np.ascontiguousarray(arr)
        probe_fmap[name] = torch.from_numpy(arr.astype(np.float64, copy=False)).to(device)

    _nerd(f"probe: {probe_rows} rows, compiling expression ...")
    t_compile_start = time.perf_counter()
    base_graph, pre_sink, warns, field_node_ids, _sympy_expr = compile_expression(expr_str, dx=dx, boundary=boundary, field_map=probe_fmap)
    t_compile = time.perf_counter() - t_compile_start
    _stage_log("compiled / post-compile")

    try: halo = _graph_row_halo(base_graph, pre_sink, chunk_dim=0)
    except ValueError as _halo_err:
        for _hf in hdf5_files:
            try: _hf.close()
            except Exception: pass
        raise ValueError(f"Expression is not safe for HDF5 chunk streaming: {_halo_err}") from _halo_err
    _nerd(f"graph halo (dim-0): {halo} row(s)")
    _stage_log("post-halo")

    with torch.no_grad():
        probe_rt      = stenpy.Runtime(stenpy.MemoryManager(), device=str(device))
        probe_out_map = probe_rt.run(base_graph)
        probe_out    = probe_out_map.get(pre_sink)
        out_trailing = tuple(probe_out.shape[1:]) if isinstance(probe_out, torch.Tensor) else ()
        in_elements_per_row  = math.prod(primary_shape[1:]) if len(primary_shape) > 1 else 1
        out_elements_per_row = math.prod(out_trailing) if out_trailing else 1
        output_multiplier    = max(1.0, out_elements_per_row / max(in_elements_per_row, 1))
    _nerd(f"probe output trailing dims: {out_trailing}  output_multiplier: {output_multiplier:.2f}x")

    _release_graph_tensors(base_graph, device)
    for name in list(probe_fmap.keys()):
        t = probe_fmap.pop(name)
        if isinstance(t, torch.Tensor) and t.device.type == "cuda": del t
    del probe_fmap
    probe_rt.flush_vram()
    del probe_out_map, probe_out, probe_rt
    _full_vram_release(device)
    _stage_log("post-probe-cleanup")
    _nerd(f"post-probe VRAM after cleanup: {torch.cuda.memory_allocated(device)/1024**3:.3f} GB" if device.type == "cuda" else "post-probe cleanup done")

    for f in hdf5_files:
        try: f.close()
        except Exception: pass
    hdf5_files.clear()
    handles.clear()

    bytes_per_out_row = out_elements_per_row * 8
    peak_overhead = _expr_peak_overhead(expr_str)
    use_row_stream = (
        bytes_per_out_row >= _ROW_STREAM_THRESHOLD
        or output_multiplier >= 3.0         
        or peak_overhead >= 5.0             
    )
    if use_row_stream:
        if verbose:
            reason = []
            if bytes_per_out_row >= _ROW_STREAM_THRESHOLD:
                reason.append(f"output row = {bytes_per_out_row/1024**2:.1f} MB ≥ threshold {_ROW_STREAM_THRESHOLD/1024**2:.0f} MB")
            if output_multiplier >= 3.0:
                reason.append(f"expansion {output_multiplier:.1f}x ≥ 3.0x")
            if peak_overhead >= 5.0:
                reason.append(f"internal op overhead estimate {peak_overhead:.1f}x ≥ 5.0x")
            reason = ' and '.join(reason)
            print(f"\n⚠  {reason} → switching to safe row-stream mode")
        return _hdf5_safe_row_stream(
            expr_str=expr_str, field_paths=field_paths, scalar_map=scalar_map, dx=dx, boundary=boundary,
            out_path=out_path, spacing=spacing, origin=origin, device=device, out_trailing=out_trailing,
            halo=halo, total_rows=total_rows, primary_shape=primary_shape, verbose=verbose,
        )

    try:
        for name, path in field_paths.items():
            try: f = _hdf5_open_cached(path, "r")
            except Exception: f = _h5py.File(path, "r")
            hdf5_files.append(f)
            key = next((k for k in ("data", "field") if k in f and isinstance(f[k], _h5py.Dataset)), next(k for k in f if isinstance(f[k], _h5py.Dataset)))
            handles[name] = f[key]
    except Exception:
        for _f in hdf5_files:
            try: _f.close()
            except Exception: pass
        raise

    try:
        primary_ds = handles[primary_name]
        template = _TemplateGraph(graph=base_graph, sink_id=pre_sink, field_node_ids=field_node_ids)

        min_chunk_for_halo = max(1, 2 * halo + 1)
        chunk_rows_target = _compute_chunk_rows(
            primary_shape, device, n_fields, output_multiplier,
            graph_overhead=max(3.0, peak_overhead),
        )
        chunk_rows_target = max(chunk_rows_target, min_chunk_for_halo)
        
        available_ram_gb = psutil.virtual_memory().available / 1024**3
        max_safe_rows = min(
            int(available_ram_gb * 0.5 * 1024**3 / max(in_elements_per_row * 8, 1)),
            total_rows, int(total_rows * _MAX_CHUNK_FRAC),
        )
        max_safe_rows = max(max_safe_rows, min_chunk_for_halo)
        if chunk_rows_target > max_safe_rows:
            if verbose: print(f"  Reducing chunk from {chunk_rows_target} to {max_safe_rows} rows (RAM safe)")
            chunk_rows_target = max_safe_rows

        chunk_rows = _compute_chunk_rows_aligned(chunk_rows_target, hdf5_chunk_size, total_rows)
        chunk_rows = max(chunk_rows, min_chunk_for_halo)
        n_chunks_est = math.ceil(total_rows / max(chunk_rows, 1))
        _budget_explain(primary_shape, device, n_fields, chunk_rows, n_chunks_est)

        if verbose and (not _IS_HPC or _HPC_RANK == 0):
            chunk_gb_in  = _tensor_gb((chunk_rows,) + primary_shape[1:]) * n_fields
            chunk_gb_out = _tensor_gb((chunk_rows,) + out_trailing) if out_trailing else 0.0
            print(f"  Direct-HDF5 stream: {total_rows} rows  initial chunk={chunk_rows} rows  (hdf5_chunk={hdf5_chunk_size}  halo={halo})  ({chunk_gb_in:.3f} GB in / {chunk_gb_out:.3f} GB out  expansion {output_multiplier:.1f}x)")
            if device.type == "cuda": print(f"  {_vram_bar(device)}")

        numa_node = _numa_node_for_device(device)
        out_full_shape = (total_rows,) + out_trailing
        _ensure_output_capacity(out_path, out_full_shape, np.float64)
        out_chunks = _hdf5_output_chunks(out_full_shape, preferred_rows=chunk_rows, dtype=np.float64)
        _logger.info(
            f"Direct output chunks: {out_chunks} "
            f"({math.prod(out_chunks) * np.dtype(np.float64).itemsize / 1024**2:.1f} MB/chunk)"
        )
        _swmr_write = os.environ.get("OPS_HDF5_SWMR", "0").lower() in ("1", "true")
        if _swmr_write: out_f = _h5py.File(out_path, "w", libver="latest")
        else: out_f = _h5py.File(out_path, "w")
        out_ds = out_f.create_dataset("data", shape=out_full_shape, dtype=np.float64, chunks=out_chunks)
        out_f.attrs["spacing"] = list(spacing) if spacing else [1.0]
        out_f.attrs["origin"]  = list(origin)  if origin  else [0.0]

        WRITE_Q_BOUND = _PIPE_DEPTH
        write_q: Queue = Queue(maxsize=WRITE_Q_BOUND)
        _writer_stop   = threading.Event()
        write_errors: List[Exception] = []
        out_min   = float("inf")
        out_max   = float("-inf")
        out_sum   = 0.0
        out_count = 0
        chunk_write_times: List[float] = []

        def _writer() -> None:
            nonlocal out_min, out_max, out_sum, out_count
            while True:
                try: item = write_q.get(timeout=0.05)
                except _QueueEmpty:
                    if _writer_stop.is_set(): break
                    continue
                if item is None: write_q.task_done(); break
                row_start_w, row_end_w, out_cpu = item
                try:
                    t_w0 = time.perf_counter()
                    arr  = out_cpu.numpy() if isinstance(out_cpu, torch.Tensor) else out_cpu
                    out_ds[row_start_w:row_end_w] = arr
                    if _HDF5_FLUSH_EVERY_ROWS > 0 and (row_end_w >= total_rows or row_end_w % _HDF5_FLUSH_EVERY_ROWS == 0):
                        out_f.flush()
                    chunk_write_times.append(time.perf_counter() - t_w0)
                    out_min    = min(out_min, float(arr.min()))
                    out_max    = max(out_max, float(arr.max()))
                    out_sum   += float(arr.sum())
                    out_count += arr.size
                except Exception as e: write_errors.append(e)
                finally: del out_cpu
                write_q.task_done()

        writer_t = threading.Thread(target=_writer, daemon=True, name="hdf5-writer")
        writer_t.start()

        ext_slot_rows = chunk_rows + 2 * halo
        pipe_slots: List[Optional[torch.Tensor]] = []
        if device.type == "cuda":
            for _ in range(_PIPE_DEPTH):
                slot_shape = (ext_slot_rows,) + primary_shape[1:]
                try:
                    buf = _alloc_pinned_on_numa(slot_shape, numa_node)
                    pipe_slots.append(buf)
                except Exception: pipe_slots.append(None)
        else: pipe_slots = [None] * _PIPE_DEPTH

        if device.type == "cuda":
            _ev_start = [torch.cuda.Event(enable_timing=True) for _ in range(_PIPE_DEPTH)]
            _ev_h2d   = [torch.cuda.Event(enable_timing=True) for _ in range(_PIPE_DEPTH)]
            _ev_comp  = [torch.cuda.Event(enable_timing=True) for _ in range(_PIPE_DEPTH)]
            _ev_d2h   = [torch.cuda.Event(enable_timing=True) for _ in range(_PIPE_DEPTH)]

        t0        = time.perf_counter()
        bytes_in  = 0
        row_pos   = 0
        n_chunks_actual = 0
        chunk_read_times:    List[float] = []
        chunk_h2d_times:     List[float] = []
        chunk_compute_times: List[float] = []
        chunk_dh_times:      List[float] = []

        compute_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        h2d_streams    = [torch.cuda.Stream(device=device) for _ in range(_PIPE_DEPTH)] if device.type == "cuda" else [None] * _PIPE_DEPTH

        chunk_mm = stenpy.MemoryManager()
        chunk_rt = stenpy.Runtime(chunk_mm, device=str(device), skip_pool=True)

        try:
            with _make_bar(n_chunks_est, desc="Streaming", unit="chunk", colour="green") as bar:
                while row_pos < total_rows:
                    if write_errors: raise RuntimeError(f"HDF5 write error: {write_errors[0]}")
                    
                    # Aggressive cleanup at top of loop
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()
                    gc.collect()

                    slot_idx = n_chunks_actual % _PIPE_DEPTH
                    chunk_rows = _adapt_chunk_rows(chunk_rows, primary_shape, device, n_fields, output_multiplier, halo=halo)
                    chunk_rows = max(_compute_chunk_rows_aligned(chunk_rows, hdf5_chunk_size, total_rows - row_pos), min_chunk_for_halo)
                    
                    ext_slot_rows = chunk_rows + 2 * halo
                    row_start    = row_pos
                    row_end      = min(row_start + chunk_rows, total_rows)
                    n_rows_chunk = row_end - row_start
                    if halo > 0:
                        ext_start   = max(0, row_start - halo)
                        ext_end     = min(total_rows, row_end + halo)
                        halo_left   = row_start - ext_start
                        halo_right  = ext_end - row_end
                    else:
                        ext_start  = row_start
                        ext_end    = row_end
                        halo_left  = 0
                        halo_right = 0
                    n_rows_ext = ext_end - ext_start

                    t_r0 = time.perf_counter()
                    chunk_np: Dict[str, np.ndarray] = {}
                    for name, ds in handles.items():
                        staging = pipe_slots[slot_idx] if device.type == "cuda" else None
                        if (staging is not None and name == primary_name and n_rows_ext == staging.shape[0] and staging.shape[1:] == ds.shape[1:]):
                            try:
                                ds.read_direct(staging.numpy(), source_sel=np.s_[ext_start:ext_end], dest_sel=np.s_[:n_rows_ext])
                                chunk_np[name] = None
                                bytes_in += n_rows_ext * math.prod(ds.shape[1:]) * 8
                                continue
                            except Exception: pass
                        arr = ds[ext_start:ext_end]
                        if not arr.flags["C_CONTIGUOUS"]: arr = np.ascontiguousarray(arr)
                        chunk_np[name] = arr.astype(np.float64, copy=False)
                        bytes_in += chunk_np[name].nbytes
                    chunk_read_times.append(time.perf_counter() - t_r0)
                    t_read_chunk = chunk_read_times[-1]
                    _logger.debug(f"CHUNK {n_chunks_actual}: HDF5 read complete ({t_read_chunk*1000:.1f}ms, {n_rows_ext} rows)")

                    t_h2d = time.perf_counter()
                    chunk_tensors: Dict[str, torch.Tensor] = {}
                    h2d_stream = h2d_streams[slot_idx]
                    if device.type == "cuda":
                        try:
                            with torch.cuda.stream(h2d_stream):
                                for name, arr in chunk_np.items():
                                    if arr is None:
                                        pin_view = pipe_slots[slot_idx][:n_rows_ext]
                                        chunk_tensors[name] = pin_view.to(device, non_blocking=True)
                                    else:
                                        staging = pipe_slots[slot_idx]
                                        n_rows_arr = arr.shape[0]
                                        if (staging is not None and n_rows_arr <= staging.shape[0] and arr.shape[1:] == staging.shape[1:]):
                                            pin_view = staging[:n_rows_arr]
                                            pin_view.copy_(torch.from_numpy(arr))
                                            chunk_tensors[name] = pin_view.to(device, non_blocking=True)
                                        else:
                                            cpu_t = torch.from_numpy(arr.copy())
                                            try: 
                                                cpu_t_p = cpu_t.pin_memory()
                                            except Exception as e:
                                                _logger.warning(f"Failed to pin CPU tensor for {name}: {e}")
                                                cpu_t_p = cpu_t
                                            chunk_tensors[name] = cpu_t_p.to(device, non_blocking=True)
                                            del cpu_t, cpu_t_p
                                h2d_stream.synchronize()
                        except Exception as h2d_err:
                            _logger.error(f"H2D transfer failed for chunk {n_chunks_actual}: {h2d_err}")
                            raise
                    else:
                        for name, arr in chunk_np.items():
                            if arr is None: chunk_tensors[name] = pipe_slots[slot_idx][:n_rows_ext].clone()
                            else: chunk_tensors[name] = torch.from_numpy(arr).clone()
                    chunk_h2d_times.append(time.perf_counter() - t_h2d)
                    _logger.debug(f"CHUNK {n_chunks_actual}: H→D transfer complete ({chunk_h2d_times[-1]*1000:.1f}ms)")
                    _log_memory_state(f"CHUNK {n_chunks_actual} AFTER H2D", device)

                    template.update_chunk_tensors(chunk_tensors)
                    _safe_mm_clear(chunk_mm)

                    t_compute = time.perf_counter()
                    if device.type == "cuda":
                        with torch.cuda.stream(compute_stream):
                            _ev_start[slot_idx].record(compute_stream)
                            with torch.no_grad(): chunk_results = chunk_rt.run(template.graph)
                            _ev_comp[slot_idx].record(compute_stream)
                    else:
                        with torch.no_grad(): chunk_results = chunk_rt.run(template.graph)
                    chunk_compute_times.append(time.perf_counter() - t_compute)
                    _logger.debug(f"CHUNK {n_chunks_actual}: GPU compute complete ({chunk_compute_times[-1]*1000:.1f}ms)")
                    _log_memory_state(f"CHUNK {n_chunks_actual} AFTER COMPUTE", device)

                    out_gpu = chunk_results.get(pre_sink)
                    if halo > 0 and isinstance(out_gpu, torch.Tensor):
                        trim_start = halo_left
                        trim_end   = halo_left + n_rows_chunk
                        expected_ext_rows = halo_left + n_rows_chunk + halo_right
                        if out_gpu.shape[0] != expected_ext_rows:
                            raise RuntimeError(f"Halo trim mismatch at row {row_start}: out_gpu.shape[0]={out_gpu.shape[0]} but expected {expected_ext_rows}")
                        slices    = [slice(None)] * out_gpu.ndim
                        slices[0] = slice(trim_start, trim_end)
                        out_gpu   = out_gpu[tuple(slices)]
                    if not out_gpu.is_contiguous(): out_gpu = out_gpu.contiguous()

                    t_dh = time.perf_counter()
                    if isinstance(out_gpu, torch.Tensor):
                        if device.type == "cuda":
                            compute_stream.synchronize()
                        out_cpu = out_gpu.detach().cpu()
                    chunk_dh_times.append(time.perf_counter() - t_dh)
                    _logger.debug(f"CHUNK {n_chunks_actual}: D→H transfer complete ({chunk_dh_times[-1]*1000:.1f}ms)")

                    if isinstance(out_gpu, torch.Tensor): del out_gpu
                    del chunk_tensors
                    del chunk_results
                    template.clear_gpu_tensors(device)
                    _safe_mm_clear(chunk_mm)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()
                    gc.collect()

                    write_q.put((row_start, row_end, out_cpu), timeout=30.0)
                    del out_cpu
                    row_pos         = row_end
                    n_chunks_actual += 1
                    elapsed = time.perf_counter() - t0
                    tput    = (bytes_in / 1024**3) / elapsed if elapsed > 0 else 0.0
                    bar.update(1)
                    status_parts = [f"{tput:.2f} GB/s"]
                    if device.type == "cuda":
                        used_gb  = torch.cuda.memory_allocated(device) / 1024**3
                        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
                        pct      = used_gb / max(total_gb, 1e-9) * 100
                        status_parts.append(f"VRAM {used_gb:.2f}/{total_gb:.0f}GB ({pct:.0f}%)")
                        if pct >= _T_CRIT: status_parts.append("CRIT")
                        else: status_parts.append(f"chunk={chunk_rows}r halo={halo}")
                    else:
                        status_parts.append(f"chunk={chunk_rows}r halo={halo}")
                    bar.set_postfix_str("  ".join(status_parts), refresh=False)
                    _logger.info(f"CHUNK {n_chunks_actual} DONE: {row_pos}/{total_rows} ({(row_pos/total_rows)*100:.1f}%)")

            try: chunk_rt.flush_vram()
            except Exception as e:
                _logger.error(f"Error flushing chunk_rt VRAM: {e}")
            try:
                write_q.put(None, timeout=30.0)
            except _QueueFull:
                _logger.warning("Unable to enqueue writer shutdown sentinel; queue is full")
                _writer_stop.set()
            writer_t.join(timeout=60.0)
            if writer_t.is_alive():
                _logger.warning("Writer thread did not finish within 60 seconds after shutdown")
            write_q.join()
        except Exception:
            _writer_stop.set()
            drained_sentinel = False
            while True:
                try:
                    _item = write_q.get_nowait()
                    write_q.task_done()
                    if _item is None: drained_sentinel = True
                except _QueueEmpty: break
            if not drained_sentinel:
                try: write_q.put_nowait(None)
                except Exception: pass
            writer_t.join(timeout=60.0)
            if writer_t.is_alive():
                _logger.warning("Writer thread did not finish within 60 seconds after exception cleanup")
            raise
        finally:
            template.clear_gpu_tensors(device)
            _safe_mm_clear(chunk_mm)
            _full_vram_release(device, template.graph)
            pipe_slots.clear()
            try: out_f.close()
            except Exception: pass
            for f in hdf5_files:
                try: f.close()
                except Exception: pass

        if write_errors: raise RuntimeError(f"HDF5 write error: {write_errors[0]}")

        elapsed  = time.perf_counter() - t0
        out_gb   = _tensor_gb(out_full_shape)
        out_mean = out_sum / out_count if out_count else 0.0
        peak_vram_gb  = 0.0
        vram_total_gb = 0.0
        if device.type == "cuda" and torch.cuda.is_available():
            peak_vram_gb  = torch.cuda.max_memory_allocated(device) / 1024**3
            vram_total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3

        return {
            "shape_out": list(out_full_shape), "min": out_min if out_min != float("inf") else 0.0,
            "max": out_max if out_max != float("-inf") else 0.0, "mean": out_mean,
            "elapsed_s": elapsed, "throughput_gbs": out_gb / elapsed if elapsed > 0 else 0.0,
            "output_multiplier": output_multiplier, "t_compile": t_compile,
            "t_read": sum(chunk_read_times), "t_h2d": sum(chunk_h2d_times),
            "t_compute": sum(chunk_compute_times), "t_d2h": sum(chunk_dh_times), "t_write": sum(chunk_write_times),
            "chunk_read_s": chunk_read_times, "chunk_h2d_s": chunk_h2d_times, "chunk_compute_s": chunk_compute_times,
            "chunk_dh_s": chunk_dh_times, "chunk_write_s": chunk_write_times, "n_chunks": n_chunks_actual,
            "chunk_rows": chunk_rows, "peak_vram_gb": peak_vram_gb, "vram_total_gb": vram_total_gb,
            "input_gb": bytes_in / 1024**3, "output_gb": out_gb,
        }
    except Exception as exc:
        _logger.error("Direct HDF5 chunked run failed, falling back to safe row-stream mode")
        _logger.error(f"Exception: {type(exc).__name__}: {exc}")
        _logger.error(traceback.format_exc())
        try:
            _log_memory_state("ON-FAIL", device)
        except Exception as _le:
            _logger.error(f"Failed to log memory state on failure: {_le}")
        try:
            diag = _LOG_DIR / f"failure_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(diag, "w") as _df:
                _df.write("Exception:\n")
                _df.write(traceback.format_exc())
            _logger.info(f"Wrote failure diagnostics to {diag}")
        except Exception:
            _logger.warning("Unable to write failure diagnostics file")
        _full_vram_release(device)
        fallback_device = torch.device("cpu") if device.type == "cuda" else device
        if device.type == "cuda":
            _logger.warning("Retrying safe row-stream on CPU to avoid GPU/OOM issues")
        return _hdf5_safe_row_stream(
            expr_str=expr_str, field_paths=field_paths, scalar_map=scalar_map, dx=dx, boundary=boundary,
            out_path=out_path, spacing=spacing, origin=origin, device=fallback_device,
            out_trailing=out_trailing, halo=halo, total_rows=total_rows, primary_shape=primary_shape, verbose=verbose,
        )

# ----------------------------------------------------------------------
# Jacobian — true row-streaming
# ----------------------------------------------------------------------
def _run_jacobian(
    field_path: str, out_path: str, spatial_ndim: int, dx: float, boundary: str,
    device: torch.device, spacing: Tuple, origin: Tuple, verbose: bool = True,
) -> Dict[str, Any]:
    in_f = _h5py.File(field_path, "r")
    try:
        key = next((k for k in ("data", "field") if k in in_f and isinstance(in_f[k], _h5py.Dataset)), next(k for k in in_f if isinstance(in_f[k], _h5py.Dataset)))
        ds = in_f[key]
        in_shape = tuple(ds.shape)
        hdf5_chunk_sz = _hdf5_chunk_size(ds)
        total_rows = in_shape[0]
        trailing   = in_shape[1:]
        probe_rows = min(7, total_rows)
        probe_arr  = ds[:probe_rows].astype(np.float64)
        probe_t    = torch.from_numpy(probe_arr).to(device)
        with torch.no_grad(): probe_out = stenpy.gradient(probe_t, dim=0, dx=dx, boundary=boundary)
        per_comp_trailing = probe_out.shape[1:]
        del probe_t, probe_out, probe_arr
        _full_vram_release(device)
        out_trailing = per_comp_trailing + (spatial_ndim,)
        out_shape    = (total_rows,) + out_trailing
        in_gb        = _tensor_gb(in_shape)
        out_gb       = _tensor_gb(out_shape)
        if verbose:
            print(f"\nJacobian  {in_shape} → {out_shape}")
            print(f"  Strategy  : {spatial_ndim} streaming passes, direct HDF5 write")
            print(f"  Input     : {in_gb:.3f} GB   Output: {out_gb:.3f} GB")
        chunk_rows = _compute_chunk_rows(in_shape, device, n_fields=1, output_multiplier=1.0)
        chunk_rows = max(chunk_rows, 1)
        chunk_rows = _compute_chunk_rows_aligned(chunk_rows, hdf5_chunk_sz, total_rows)
        _ensure_output_capacity(out_path, out_shape, np.float64)
        chunk_h5 = _hdf5_output_chunks(out_shape, preferred_rows=min(64, total_rows), dtype=np.float64)
        _logger.info(
            f"Jacobian output chunks: {chunk_h5} "
            f"({math.prod(chunk_h5) * np.dtype(np.float64).itemsize / 1024**2:.1f} MB/chunk)"
        )
        with _h5py.File(out_path, "w") as out_f:
            out_ds = out_f.create_dataset("data", shape=out_shape, dtype=np.float64, chunks=chunk_h5)
            out_f.attrs["spacing"]      = list(spacing) if spacing else [1.0]
            out_f.attrs["origin"]       = list(origin)  if origin  else [0.0]
            out_f.attrs["type"]         = "jacobian"
            out_f.attrs["spatial_ndim"] = spatial_ndim
            t0 = time.perf_counter()
            total_bytes = 0
            for d in range(spatial_ndim):
                if verbose: print(f"\nPass {d+1}/{spatial_ndim}  — gradient(f, dim={d})")
                row_pos  = 0
                n_chunks = math.ceil(total_rows / chunk_rows)
                with _make_bar(n_chunks, desc=f"  grad dim={d}", unit="chunk", colour="cyan") as bar:
                    while row_pos < total_rows:
                        row_end = min(row_pos + chunk_rows, total_rows)
                        arr = ds[row_pos:row_end]
                        if not arr.flags["C_CONTIGUOUS"]: arr = np.ascontiguousarray(arr)
                        arr = arr.astype(np.float64, copy=False)
                        total_bytes += arr.nbytes
                        t_in = torch.from_numpy(arr).to(device)
                        del arr
                        with torch.no_grad(): grad_out = stenpy.gradient(t_in, dim=d, dx=dx, boundary=boundary)
                        del t_in
                        if device.type == "cuda": torch.cuda.synchronize()
                        grad_cpu = grad_out.detach().cpu().numpy()
                        del grad_out
                        out_ds[row_pos:row_end, ..., d] = grad_cpu
                        if _HDF5_FLUSH_EVERY_ROWS > 0 and (row_end >= total_rows or row_end % _HDF5_FLUSH_EVERY_ROWS == 0):
                            out_f.flush()
                        del grad_cpu
                        _full_vram_release(device)
                        gc.collect()
                        row_pos = row_end
                        bar.update(1)
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"\n✓ Jacobian complete  {elapsed:.1f}s  ({out_gb/max(elapsed,1e-9):.3f} GB/s)")
                print(f"  Output shape : {out_shape}  ({out_gb:.3f} GB)")
                print(f"  Saved → {out_path}")
            return {
                "shape_out": list(out_shape), "elapsed_s": elapsed,
                "throughput_gbs": out_gb / max(elapsed, 1e-9),
                "input_gb": total_bytes / 1024**3, "output_gb": out_gb,
            }
    finally: in_f.close()

# ----------------------------------------------------------------------
# Hessian — true row-streaming
# ----------------------------------------------------------------------
def _run_hessian(
    field_path: str, out_path: str, spatial_ndim: int, dx: float, boundary: str,
    device: torch.device, spacing: Tuple, origin: Tuple, verbose: bool = True,
) -> Dict[str, Any]:
    in_f = _h5py.File(field_path, "r")
    try:
        key = next((k for k in ("data", "field") if k in in_f and isinstance(in_f[k], _h5py.Dataset)), next(k for k in in_f if isinstance(in_f[k], _h5py.Dataset)))
        ds = in_f[key]
        in_shape = tuple(ds.shape)
        hdf5_chunk_sz = _hdf5_chunk_size(ds)
        total_rows = in_shape[0]
        trailing   = in_shape[1:]
        probe_rows = min(7, total_rows)
        probe_arr  = ds[:probe_rows].astype(np.float64)
        probe_t    = torch.from_numpy(probe_arr).to(device)
        with torch.no_grad():
            g1 = stenpy.gradient(probe_t, dim=0, dx=dx, boundary=boundary)
            g2 = stenpy.gradient(g1,      dim=0, dx=dx, boundary=boundary)
            per_comp_trailing = g2.shape[1:]
        del probe_t, g1, g2, probe_arr
        _full_vram_release(device)
        out_trailing = trailing + (spatial_ndim, spatial_ndim)
        out_shape    = (total_rows,) + out_trailing
        in_gb        = _tensor_gb(in_shape)
        out_gb       = _tensor_gb(out_shape)
        n_passes     = spatial_ndim * spatial_ndim
        if verbose:
            print(f"\nHessian  {in_shape} → {out_shape}")
            print(f"  Strategy : {n_passes} streaming passes, direct HDF5 write")
            print(f"  Input    : {in_gb:.3f} GB   Output: {out_gb:.3f} GB")
        chunk_rows = _compute_chunk_rows(in_shape, device, n_fields=1, output_multiplier=1.0)
        chunk_rows = max(chunk_rows, 1)
        chunk_rows = _compute_chunk_rows_aligned(chunk_rows, hdf5_chunk_sz, total_rows)
        _ensure_output_capacity(out_path, out_shape, np.float64)
        chunk_h5 = _hdf5_output_chunks(out_shape, preferred_rows=min(32, total_rows), dtype=np.float64)
        _logger.info(
            f"Hessian output chunks: {chunk_h5} "
            f"({math.prod(chunk_h5) * np.dtype(np.float64).itemsize / 1024**2:.1f} MB/chunk)"
        )
        with _h5py.File(out_path, "w") as out_f:
            out_ds = out_f.create_dataset("data", shape=out_shape, dtype=np.float64, chunks=chunk_h5)
            out_f.attrs["spacing"]      = list(spacing) if spacing else [1.0]
            out_f.attrs["origin"]       = list(origin)  if origin  else [0.0]
            out_f.attrs["type"]         = "hessian"
            out_f.attrs["spatial_ndim"] = spatial_ndim
            t0 = time.perf_counter()
            total_bytes = 0
            pass_num = 0
            for i in range(spatial_ndim):
                for j in range(spatial_ndim):
                    pass_num += 1
                    if verbose: print(f"\nPass {pass_num}/{n_passes} — ∂²f/∂x{i}∂x{j}")
                    row_pos  = 0
                    n_chunks = math.ceil(total_rows / chunk_rows)
                    with _make_bar(n_chunks, desc=f"  hess[{i},{j}]", unit="chunk", colour="magenta") as bar:
                        while row_pos < total_rows:
                            row_end = min(row_pos + chunk_rows, total_rows)
                            arr = ds[row_pos:row_end]
                            if not arr.flags["C_CONTIGUOUS"]: arr = np.ascontiguousarray(arr)
                            arr = arr.astype(np.float64, copy=False)
                            total_bytes += arr.nbytes
                            t_in = torch.from_numpy(arr).to(device)
                            del arr
                            with torch.no_grad():
                                g1 = stenpy.gradient(t_in, dim=j, dx=dx, boundary=boundary)
                                g2 = stenpy.gradient(g1,   dim=i, dx=dx, boundary=boundary)
                            del t_in, g1
                            if device.type == "cuda": torch.cuda.synchronize()
                            g2_cpu = g2.detach().cpu().numpy()
                            del g2
                            out_ds[row_pos:row_end, ..., i, j] = g2_cpu
                            if _HDF5_FLUSH_EVERY_ROWS > 0 and (row_end >= total_rows or row_end % _HDF5_FLUSH_EVERY_ROWS == 0):
                                out_f.flush()
                            del g2_cpu
                            _full_vram_release(device)
                            gc.collect()
                            row_pos = row_end
                            bar.update(1)
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"\n✓ Hessian complete  {elapsed:.1f}s  ({out_gb/max(elapsed,1e-9):.3f} GB/s)")
                print(f"  Output shape : {out_shape}  ({out_gb:.3f} GB)")
                print(f"  Saved → {out_path}")
            return {
                "shape_out": list(out_shape), "elapsed_s": elapsed,
                "throughput_gbs": out_gb / max(elapsed, 1e-9),
                "input_gb": total_bytes / 1024**3, "output_gb": out_gb,
            }
    finally: in_f.close()

# ----------------------------------------------------------------------
# Pipeline executor
# ----------------------------------------------------------------------
@dataclass
class RunResult:
    op_name:         str
    expr_str:        str
    output:          Any
    elapsed_ms:      float
    shape_in:        Tuple
    shape_out:       Optional[Tuple]
    simplifications: List[str]
    node_count:      int
    out_path:        Optional[str]  = None
    throughput_gbs:  float          = 0.0
    perf:            Optional[PerfStats] = None

def _print_banner(title, width=72):
    _rank0_print(f"\n{'═'*width}\n{title}\n{'═'*width}")

def _print_section(title, width=72):
    _rank0_print(f"\n{'─'*width}\n{title}\n{'─'*width}")

def _load_field(path, device, normalize):
    with _h5py.File(path, "r") as f:
        key = next((k for k in ("data", "field") if k in f), list(f.keys())[0])
        nbytes = f[key].size * 8
        mode = "lazy"
        loaded = stenpy.load_tensor(path, device=device, normalize=normalize, return_mode=mode, max_eager_gb=0.1)
        if isinstance(loaded, tuple) and len(loaded) == 3: return loaded
        return loaded, (1.0,) * loaded.ndim, (0.0,) * loaded.ndim

class Pipeline:
    def __init__(self, field_paths, project=None, device="cpu", dx=1.0, boundary="neumann", normalize=False, verbose=True):
        if not field_paths: raise ValueError("At least one field path must be supplied.")
        self.field_paths = field_paths
        self.device      = torch.device(device)
        self.dx          = dx
        self.boundary    = boundary
        self.normalize   = normalize
        self.verbose     = verbose and (not _IS_HPC or _HPC_RANK == 0)
        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        proj_name = project or f"run_{ts}"
        if _IS_HPC: self.out_dir = Path(_hpc_scratch()) / "ops_outputs" / proj_name
        else: self.out_dir = Path("outputs") / proj_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fields:   Dict[str, Any]               = {}
        self.spacings: Dict[str, Tuple[float, ...]] = {}
        self.origins:  Dict[str, Tuple[float, ...]] = {}
        self.shapes:   Dict[str, Tuple[int, ...]]   = {}
        self.spacing:  Optional[Tuple[float, ...]]  = None
        self.origin:   Optional[Tuple[float, ...]]  = None
        self.src_stem  = "field"
        for name, path in field_paths.items():
            if self.verbose: print(f"  Loading field '{name}' from : {path}")
            _hdf5_dissect(path)
            tensor, spacing, origin = _load_field(path, self.device, normalize)
            shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()
            self.fields[name]   = tensor
            self.spacings[name] = spacing
            self.origins[name]  = origin
            self.shapes[name]   = shape
            if self.spacing is None:
                self.spacing = spacing; self.origin = origin
                self.src_stem = Path(path).stem
        self.tensor = next(iter(self.fields.values()))
        if len(field_paths) > 1:
            merged_dx, merged_origin, merged_end = _merge_domain(self.spacings, self.origins, self.shapes)
            if self.verbose:
                print(f"\n── Multi-field domain merge ──────────────────────")
                print(f"  dx      : {merged_dx:.6g}  origin  : {merged_origin}  end: {merged_end}")
            if self.dx == 1.0: self.dx = merged_dx
            self.spacing = tuple(merged_dx for _ in merged_origin)
            self.origin  = merged_origin
        elif self.dx == 1.0 and self.spacing is not None:
            self.dx = float(min(self.spacing))
        if self.verbose:
            pshape = tuple(self.tensor.shape) if hasattr(self.tensor, "shape") else ("?",)
            print(f"\nFields loaded       : {list(self.fields.keys())}")
            print(f"  Primary shape       : {pshape}")
            print(f"  Field size          : {_tensor_gb(pshape):.3f} GB  (float64)")
            print(f"  Spacing             : {self.spacing}")
            print(f"  dx  (from HDF5)     : {self.dx:.6g}")
            print(f"  Device              : {self.device}")
            if self.device.type == "cuda": print(f"  {_vram_bar(self.device)}")
            bw = _gpu_peak_bw_gbs(self.device)
            if not math.isnan(bw): print(f"  GPU peak mem BW     : {bw:.0f} GB/s")
            print(f"  Output folder       : {self.out_dir}/")
            if _IS_HPC: print(f"  HPC mode            : rank {_HPC_RANK}/{_HPC_WORLD}")
            if _NERD:   print(f"  Nerd mode           : ON  (OPS_NERD=1)")
            
            _harmonize_memory_settings(self.device)
            
            free_vram = _free_vram_bytes(self.device) / 1024**3 if self.device.type == "cuda" else 0
            free_ram  = _free_ram_bytes() / 1024**3
            print(f"  Free VRAM           : {free_vram:.2f} GB" if self.device.type == "cuda" else f"  Free RAM            : {free_ram:.2f} GB")
            print(f"  Row-stream threshold: {_ROW_STREAM_THRESHOLD/1024**2:.0f} MB/row  (env: OPS_ROW_STREAM_THRESHOLD)")
            print(f"  HDF5 chunk cache    : {_HDF5_RDCC_NBYTES//1024//1024} MB  slots={_HDF5_RDCC_NSLOTS}  (env: OPS_HDF5_RDCC_NBYTES)")
            print(f"  Pipeline depth      : {_PIPE_DEPTH}  (env: OPS_PIPE_DEPTH)")
        self.mm = stenpy.MemoryManager()
        self.rt = stenpy.Runtime(self.mm, device=device)
        self._run_index = 0
        for name, tensor in self.fields.items():
            if isinstance(tensor, stenpy.LazyField): print(f"  ✓ Field '{name}' is LAZY (streaming from disk)")
            else: print(f"  ⚠ Field '{name}' is EAGER (loaded in memory) - shape {tensor.shape} = {tensor.nbytes/1e9:.2f} GB")

    def load_field(self, name, path):
        if self.verbose: print(f"  Loading field '{name}' from : {path}")
        _hdf5_dissect(path)
        tensor, spacing, origin = _load_field(path, self.device, self.normalize)
        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()
        self.fields[name] = tensor; self.spacings[name] = spacing
        self.origins[name] = origin; self.shapes[name] = shape
        self.field_paths[name] = path
        merged_dx, merged_origin, merged_end = _merge_domain(self.spacings, self.origins, self.shapes)
        self.dx = merged_dx
        self.spacing = tuple(merged_dx for _ in merged_origin)
        self.origin  = merged_origin
        if self.verbose:
            print(f"  '{name}' loaded — shape {shape}")
            print(f"  Domain re-merged  dx={merged_dx:.6g}  origin={merged_origin}  end={merged_end}")

    def _resolve_boundary_and_vars(self, expr_str):
        boundary = self.boundary
        spatial_ops = {"gradient", "gradient_nd", "divergence", "laplacian", "curl", "hessian", "mean_curvature", "surface_normals", "material_derivative", "spectral_gradient", "spectral_laplacian"}
        if (not _IS_HPC and any(op in expr_str for op in spatial_ops) and boundary == "neumann"):
            print(f"\nℹ  Expression uses a spatial operator.  Current BC: {boundary}")
            try: ans = input("  Keep it or change? [Enter = keep, c = change] ❯ ").strip()
            except (EOFError, KeyboardInterrupt): ans = ""
            if ans.lower() in ("c", "change"):
                boundary = _select_boundary(); self.boundary = boundary
        field_map = dict(self.fields)
        for name, uv in _USER_VARS.items():
            if not re.search(rf'\b{re.escape(name)}\b', expr_str): continue
            if uv.steps <= 1:
                uv.current = uv.start
                field_map[name] = torch.tensor(uv.current, dtype=torch.float64)
            else:
                if _IS_HPC:
                    uv.current = uv.start; field_map.pop(name, None)
                    field_map[f"__sweep_{name}"] = uv
                else:
                    print(f"\nVariable '{name}'  range [{uv.start}, {uv.end}]  {uv.steps} steps")
                    print("    [1]  Use start value only\n[2]  Enter a specific value\n[3]  Sweep all values")
                    try: ch = input("  Choice [2] ❯ ").strip() or "2"
                    except (EOFError, KeyboardInterrupt): ch = "1"
                    if ch == "1":
                        uv.current = uv.start
                        field_map[name] = torch.tensor(uv.current, dtype=torch.float64)
                    elif ch == "3":
                        uv.current = uv.start; field_map.pop(name, None)
                        field_map[f"__sweep_{name}"] = uv
                    else:
                        try: uv.current = float(input(f"  Value for {name} [{uv.current}] ❯ ").strip() or str(uv.current))
                        except ValueError: uv.current = uv.start
                        field_map[name] = torch.tensor(uv.current, dtype=torch.float64)
        return boundary, field_map

    def run_expr(self, expr_str: str) -> RunResult:
        self._run_index += 1
        if self.verbose: _print_section(f"Expression #{self._run_index}:  {expr_str}")
        boundary, field_map = self._resolve_boundary_and_vars(expr_str)
        sweep_keys = [k for k in field_map if k.startswith("__sweep_")]
        if sweep_keys: return self._run_sweep(expr_str, field_map, sweep_keys, boundary)
        short_name = _expr_to_filename(expr_str, self.src_stem)
        out_path   = str(self.out_dir / f"{short_name}.h5")
        primary    = self.tensor
        is_lazy    = isinstance(primary, stenpy.LazyField)
        if is_lazy:
            lazy_field_names = {n for n, v in field_map.items() if isinstance(v, stenpy.LazyField)}
            lazy_path_map    = {n: self.field_paths[n] for n in lazy_field_names if n in self.field_paths}
            scalar_map_for_chunk = {n: v for n, v in field_map.items() if n not in lazy_path_map and not n.startswith("__") and isinstance(v, torch.Tensor)}
            sympy_expr_for_flops = None
            warns: List[str] = []
            try:
                probe_fm = dict(scalar_map_for_chunk)
                for n, p in lazy_path_map.items():
                    with stenpy._open_hdf5_field(p) as (ds, _shape):
                        probe_fm[n] = torch.from_numpy(ds[:2].astype(np.float64))
                _, _, warns, _, sympy_expr_for_flops = compile_expression(expr_str, dx=self.dx, boundary=boundary, field_map=probe_fm)
                if self.verbose and warns:
                    for w in warns: print(f"  ℹ  {w}")
            except Exception: warns = []
            in_shape = tuple(primary.shape)
            if self.verbose:
                print(f"\nLazyField  {in_shape}  {_tensor_gb(in_shape):.3f} GB")
                print(f"  → Direct-HDF5 adaptive-chunked stream  (triple-buffer depth={_PIPE_DEPTH})")
                if self.device.type == "cuda": print(f"  {_vram_bar(self.device)}")
            t_total_start = time.perf_counter()
            stats = _hdf5_direct_chunked_run(
                expr_str=expr_str, field_paths=lazy_path_map, scalar_map=scalar_map_for_chunk, dx=self.dx, boundary=boundary,
                out_path=out_path, spacing=self.spacing, origin=self.origin, device=self.device, verbose=self.verbose,
            )
            t_total = time.perf_counter() - t_total_start
            _full_vram_release(self.device)
            shape_out  = tuple(stats["shape_out"])
            elapsed_ms = stats["elapsed_s"] * 1e3
            tput       = stats["throughput_gbs"]
            n_elem     = math.prod(int(s) for s in shape_out) if shape_out else 1
            est_flops  = (_ast_flops(sympy_expr_for_flops, n_elem) if sympy_expr_for_flops is not None else 0.0)
            if self.verbose:
                print(f"\n✓ {elapsed_ms:.0f} ms ({elapsed_ms/1e3:.1f} s)  │  {tput:.3f} GB/s")
                print(f"  Output shape  : {shape_out}  ({_tensor_gb(shape_out):.3f} GB)")
                print(f"  min / max     : {stats['min']:.6g}  /  {stats['max']:.6g}")
                print(f"  mean          : {stats['mean']:.6g}")
                if self.device.type == "cuda": print(f"  {_vram_bar(self.device)}")
                print(f"  Saved → {out_path}")
            perf = PerfStats(
                expr_str=expr_str, mode="lazy",
                input_gb=stats.get("input_gb", _tensor_gb(in_shape)),
                output_gb=stats.get("output_gb", _tensor_gb(shape_out)),
                t_compile=stats.get("t_compile", 0.0), t_read=stats.get("t_read", 0.0), t_h2d=stats.get("t_h2d", 0.0),
                t_compute=stats.get("t_compute", stats["elapsed_s"]), t_d2h=stats.get("t_d2h", 0.0), t_write=stats.get("t_write", 0.0),
                t_total=t_total, chunk_read_s=stats.get("chunk_read_s", []), chunk_h2d_s=stats.get("chunk_h2d_s", []),
                chunk_compute_s=stats.get("chunk_compute_s", []), chunk_dh_s=stats.get("chunk_dh_s", []),
                chunk_write_s=stats.get("chunk_write_s", []), n_chunks=stats.get("n_chunks", 0), chunk_rows=stats.get("chunk_rows", 0),
                peak_vram_gb=stats.get("peak_vram_gb", 0.0), vram_total_gb=stats.get("vram_total_gb", 0.0), est_flops=est_flops,
            )
            _flush_vram(self.device, verbose=False)
            return RunResult(
                op_name=expr_str, expr_str=expr_str, output=None, elapsed_ms=elapsed_ms, shape_in=in_shape, shape_out=shape_out,
                simplifications=warns, node_count=0, out_path=out_path, throughput_gbs=tput, perf=perf,
            )
        
        t_compile_start = time.perf_counter()
        try:
            graph, sink_id, warns, field_node_ids, sympy_expr = compile_expression(
                expr_str, dx=self.dx, boundary=boundary, field_map={k: v for k, v in field_map.items() if not k.startswith("__")})
        except ValueError as exc:
            print(f"  ✗ Parse/compile error: {exc}")
            raise
        t_compile = time.perf_counter() - t_compile_start
        if self.verbose and warns:
            for w in warns: print(f"  ℹ  {w}")
        if self.verbose:
            topo = graph.topological_sort()
            print(f"  Graph  ({len(topo)} nodes):")
            for n in topo:
                meta = stenpy.OP_METADATA.get(n.op_name, {})
                cost = meta.get("cost", "")
                cost_str = f"  [{cost}]" if cost else ""
                deps = f"← {n.input_ids}" if n.input_ids else "(source)"
                print(f"    {n.op_name:<22}  {deps}{cost_str}")
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
        try:
            t_compute_start = time.perf_counter()
            if self.verbose:
                with _make_bar(1, desc="Computing", unit="step") as bar:
                    with torch.no_grad(): results = self.rt.run(graph)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                        bar.update(1)
            else:
                with torch.no_grad(): results = self.rt.run(graph)
                if self.device.type == "cuda": torch.cuda.synchronize()
            t_compute  = time.perf_counter() - t_compute_start
            elapsed_ms = (t_compile + t_compute) * 1e3
            output     = results.get(sink_id)
            shape_out  = tuple(output.shape) if isinstance(output, torch.Tensor) else None
            out_gb     = _tensor_gb(shape_out) if shape_out else 0.0
            tput       = out_gb / t_compute if t_compute > 1e-9 else 0.0
            t_dh_start = time.perf_counter()
            out_cpu_save = None
            if isinstance(output, torch.Tensor) and output.ndim >= 1:
                out_cpu_save = output.detach().cpu()
                if self.device.type == "cuda": torch.cuda.synchronize()
            t_d2h = time.perf_counter() - t_dh_start
            peak_vram_gb  = 0.0
            vram_total_gb = 0.0
            if self.device.type == "cuda" and torch.cuda.is_available():
                peak_vram_gb  = torch.cuda.max_memory_allocated(self.device) / 1024**3
                vram_total_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            if self.verbose:
                print(f"  ✓ {elapsed_ms:.0f} ms  │  {tput:.3f} GB/s")
                if shape_out: print(f"  Output shape  : {shape_out}  ({out_gb:.3f} GB)")
                if isinstance(output, torch.Tensor):
                    flat = output.flatten()
                    print(f"  min / max     : {flat.min().item():.6g}  /  {flat.max().item():.6g}")
                    print(f"  mean          : {flat.mean().item():.6g}")
                    if torch.isnan(output).any(): print("  ██ CRIT  NaN detected in output!")
                    if torch.isinf(output).any(): print("  ▲▲ WARN  Inf detected in output!")
                if self.device.type == "cuda": print(f"  {_vram_bar(self.device)}")
            in_shape = tuple(self.tensor.shape) if hasattr(self.tensor, "shape") else ()
            in_gb    = _tensor_gb(in_shape)
            t_write_start = time.perf_counter()
            if out_cpu_save is not None:
                stenpy.save_tensor(out_cpu_save, self.spacing, self.origin, out_path)
                if self.verbose: print(f"  Saved → {out_path}")
            else:
                out_path = out_path.replace(".h5", ".json")
                scalar_val = output.item() if isinstance(output, torch.Tensor) else float(output)
                with open(out_path, "w") as jf:
                    json.dump({"expression": expr_str, "result": scalar_val, "elapsed_ms": elapsed_ms}, jf, indent=2)
                if self.verbose: print(f"  Scalar: {scalar_val:.6g}  → {out_path}")
            t_write = time.perf_counter() - t_write_start
            n_elem    = math.prod(int(s) for s in shape_out) if shape_out else 1
            est_flops = _ast_flops(sympy_expr, n_elem)
            perf = PerfStats(
                expr_str=expr_str, mode="eager", input_gb=in_gb, output_gb=out_gb,
                t_compile=t_compile, t_read=0.0, t_h2d=0.0, t_compute=t_compute, t_d2h=t_d2h, t_write=t_write,
                t_total=t_compile + t_compute + t_d2h + t_write, peak_vram_gb=peak_vram_gb, vram_total_gb=vram_total_gb, est_flops=est_flops,
            )
            return RunResult(
                op_name=expr_str, expr_str=expr_str, output=output, elapsed_ms=elapsed_ms, shape_in=in_shape, shape_out=shape_out,
                simplifications=warns, node_count=len(graph), out_path=out_path, throughput_gbs=tput, perf=perf,
            )
        finally: self.rt.flush_vram()

    def run_jacobian(self, spatial_ndim: int = 3) -> RunResult:
        self._run_index += 1
        primary_name = next(iter(self.field_paths))
        field_path   = self.field_paths[primary_name]
        short_name   = f"jacobian_{self.src_stem}"
        out_path     = str(self.out_dir / f"{short_name}.h5")
        _print_section(f"Expression #{self._run_index}:  jacobian(f)  spatial_ndim={spatial_ndim}")
        t0    = time.perf_counter()
        stats = _run_jacobian(field_path=field_path, out_path=out_path, spatial_ndim=spatial_ndim, dx=self.dx, boundary=self.boundary, device=self.device, spacing=self.spacing, origin=self.origin, verbose=self.verbose)
        t_total = time.perf_counter() - t0
        in_shape  = tuple(self.tensor.shape)
        shape_out = tuple(stats["shape_out"])
        perf = PerfStats(expr_str="jacobian(f)", mode="composite", input_gb=stats["input_gb"], output_gb=stats["output_gb"], t_compute=stats["elapsed_s"], t_total=t_total, est_flops=stats["input_gb"] * 1024**3 / 8 * 6 * spatial_ndim)
        return RunResult(op_name="jacobian", expr_str="jacobian(f)", output=None, elapsed_ms=t_total * 1e3, shape_in=in_shape, shape_out=shape_out, simplifications=[], node_count=0, out_path=out_path, throughput_gbs=stats["throughput_gbs"], perf=perf)

    def run_hessian(self, spatial_ndim: int = 3) -> RunResult:
        self._run_index += 1
        primary_name = next(iter(self.field_paths))
        field_path   = self.field_paths[primary_name]
        short_name   = f"hessian_{self.src_stem}"
        out_path     = str(self.out_dir / f"{short_name}.h5")
        _print_section(f"Expression #{self._run_index}:  hessian(f)  spatial_ndim={spatial_ndim}")
        t0    = time.perf_counter()
        stats = _run_hessian(field_path=field_path, out_path=out_path, spatial_ndim=spatial_ndim, dx=self.dx, boundary=self.boundary, device=self.device, spacing=self.spacing, origin=self.origin, verbose=self.verbose)
        t_total = time.perf_counter() - t0
        in_shape  = tuple(self.tensor.shape)
        shape_out = tuple(stats["shape_out"])
        perf = PerfStats(expr_str="hessian(f)", mode="composite", input_gb=stats["input_gb"], output_gb=stats["output_gb"], t_compute=stats["elapsed_s"], t_total=t_total, est_flops=stats["input_gb"] * 1024**3 / 8 * 6 * spatial_ndim**2)
        return RunResult(op_name="hessian", expr_str="hessian(f)", output=None, elapsed_ms=t_total * 1e3, shape_in=in_shape, shape_out=shape_out, simplifications=[], node_count=0, out_path=out_path, throughput_gbs=stats["throughput_gbs"], perf=perf)

    def _run_sweep(self, expr_str, field_map, sweep_keys, boundary):
        svars: List[UserVar] = [field_map.pop(k) for k in sweep_keys]
        clean_map = {k: v for k, v in field_map.items() if not k.startswith("__")}
        results_list: List[RunResult] = []
        uv = svars[0]
        vals = uv.values
        has_lazy = any(isinstance(v, stenpy.LazyField) for v in clean_map.values())
        sweep_sympy_expr = None
        try: sweep_sympy_expr = parse_expression(expr_str, field_names=set(clean_map.keys()) | {uv.name})
        except Exception: pass
        compiled_graph = compiled_sink = None
        compiled_warns = []
        compiled_fnids = {}
        if self.verbose:
            print(f"\nSweeping '{uv.name}' over {len(vals)} values …")
            with _make_bar(len(vals), desc=f"Sweep {uv.name}", unit="step", colour="green") as bar:
                for v in vals:
                    uv.current = float(v)
                    sweep_map = {**clean_map, uv.name: torch.tensor(uv.current, dtype=torch.float64)}
                    fname = _expr_to_filename(f"{expr_str}_{uv.name}{v:.4g}", self.src_stem)
                    op_path = str(self.out_dir / f"{fname}.h5")
                    try:
                        if has_lazy:
                            lazy_path_map = {n: self.field_paths[n] for n, val in clean_map.items() if isinstance(val, stenpy.LazyField) and n in self.field_paths}
                            scalar_map_for_chunk = {n: val for n, val in sweep_map.items() if n not in lazy_path_map and not n.startswith("__") and isinstance(val, torch.Tensor)}
                            stats = _hdf5_direct_chunked_run(
                                expr_str=expr_str, field_paths=lazy_path_map, scalar_map=scalar_map_for_chunk, dx=self.dx, boundary=boundary,
                                out_path=op_path, spacing=self.spacing, origin=self.origin, device=self.device, verbose=self.verbose,
                            )
                            _full_vram_release(self.device)
                            shape_out = tuple(stats["shape_out"])
                            in_shape = tuple(next(iter(self.shapes.values())))
                            n_elem = math.prod(int(s) for s in shape_out) if shape_out else 1
                            est_flops = (_ast_flops(sweep_sympy_expr, n_elem) if sweep_sympy_expr is not None else 0.0)
                            perf = PerfStats(
                                expr_str=expr_str, mode="lazy", input_gb=stats.get("input_gb", 0.0), output_gb=stats.get("output_gb", _tensor_gb(shape_out)),
                                t_compile=stats.get("t_compile", 0.0), t_read=stats.get("t_read", 0.0), t_h2d=stats.get("t_h2d", 0.0),
                                t_compute=stats.get("t_compute", stats["elapsed_s"]), t_d2h=stats.get("t_d2h", 0.0), t_write=stats.get("t_write", 0.0),
                                t_total=stats["elapsed_s"], chunk_read_s=stats.get("chunk_read_s", []), chunk_h2d_s=stats.get("chunk_h2d_s", []),
                                chunk_compute_s=stats.get("chunk_compute_s", []), chunk_dh_s=stats.get("chunk_dh_s", []), chunk_write_s=stats.get("chunk_write_s", []),
                                n_chunks=stats.get("n_chunks", 0), chunk_rows=stats.get("chunk_rows", 0), peak_vram_gb=stats.get("peak_vram_gb", 0.0),
                                vram_total_gb=stats.get("vram_total_gb", 0.0), est_flops=est_flops,
                            )
                            results_list.append(RunResult(
                                op_name=expr_str, expr_str=expr_str, output=None, elapsed_ms=stats["elapsed_s"] * 1e3,
                                shape_in=in_shape, shape_out=shape_out, simplifications=[], node_count=0,
                                out_path=op_path, throughput_gbs=stats["throughput_gbs"], perf=perf,
                            ))
                            if self.verbose: print(f"  {uv.name}={v:.4g}  →  {op_path}  ({stats['elapsed_s']:.1f}s  {stats['throughput_gbs']:.3f} GB/s)")
                        else:
                            if compiled_graph is None:
                                compiled_graph, compiled_sink, compiled_warns, compiled_fnids, _ = compile_expression(expr_str, dx=self.dx, boundary=boundary, field_map=sweep_map)
                            else:
                                scalar_node_id = compiled_fnids.get(uv.name)
                                if scalar_node_id is not None:
                                    compiled_graph = compiled_graph.clone_with_replacement(scalar_node_id, torch.tensor(uv.current, dtype=torch.float64))
                            with torch.no_grad(): res = self.rt.run(compiled_graph)
                            if self.device.type == "cuda": torch.cuda.synchronize()
                            out = res.get(compiled_sink)
                            if isinstance(out, torch.Tensor) and out.ndim >= 1:
                                stenpy.save_tensor(out, self.spacing, self.origin, op_path)
                            results_list.append(RunResult(
                                op_name=expr_str, expr_str=expr_str, output=out, elapsed_ms=0, shape_in=(),
                                shape_out=tuple(out.shape) if isinstance(out, torch.Tensor) else None,
                                simplifications=compiled_warns, node_count=len(compiled_graph), out_path=op_path,
                            ))
                    except Exception as exc:
                        _rank0_print(f"\n✗ {uv.name}={v:.4g}  {exc}")
                        compiled_graph = None
                    finally:
                        if not has_lazy: self.rt.flush_vram()
                    bar.update(1)
            uv.current = uv.start
        if results_list:
            if self.verbose: print(f"  Sweep complete — {len(results_list)} outputs in {self.out_dir}/")
            return results_list[-1]
        else:
            _rank0_print(f"  ✗ Sweep produced no outputs.")
            return RunResult(op_name=expr_str, expr_str=expr_str, output=None, elapsed_ms=0, shape_in=(), shape_out=None, simplifications=[], node_count=0)

    def run_batch(self, expressions):
        _print_banner(f"ops.py  Multi-Operator Pipeline  —  {len(expressions)} expr(s)")
        _rank0_print(f"  Fields : {list(self.fields.keys())}  dx={self.dx:.4g}  bc={self.boundary}  device={self.device}")
        if _IS_HPC: _rank0_print(f"  HPC    : rank {_HPC_RANK}/{_HPC_WORLD}  scratch={_hpc_scratch()}")
        if self.device.type == "cuda":
            _rank0_print(f"  {_vram_bar(self.device)}")
            torch.cuda.reset_peak_memory_stats(self.device)
        results_list: List[RunResult] = []
        total_t0 = time.perf_counter()
        for expr_str in expressions:
            try: results_list.append(self.run_expr(expr_str))
            except Exception as exc: _rank0_print(f"\n✗ FAILED: {expr_str}  →  {exc}")
            if self.device.type == "cuda":
                _full_vram_release(self.device)
                if self.device.type == "cuda" and _gpu_free_gb(self.device) < 0.5:
                    _rank0_print(f"  ▲▲ WARN  Low VRAM — flushing automatically.")
                    _flush_vram(self.device, verbose=self.verbose)
        total_s = time.perf_counter() - total_t0
        _print_section("Summary")
        col_w = 36
        _rank0_print(f"  {'#':<3}  {'Expression':<{col_w}}  {'ms':>9}  {'GB/s':>6}  Shape")
        _rank0_print(f"  {'─'*3}  {'─'*col_w}  {'─'*9}  {'─'*6}  {'─'*20}")
        for i, r in enumerate(results_list, 1):
            shape = str(r.shape_out) if r.shape_out else "scalar"
            expr  = _truncate(r.expr_str, col_w)
            _rank0_print(f"  {i:<3}  {expr:<{col_w}}  {r.elapsed_ms:>9.0f}  {r.throughput_gbs:>6.3f}  {shape}")
        _rank0_print(f"\nTotal  : {total_s*1e3:.0f} ms  ({total_s:.1f} s)")
        _rank0_print(f"  Output : {self.out_dir}/")
        perf_list = [r.perf for r in results_list if r.perf is not None]
        if perf_list and self.verbose: _print_perf_report(perf_list, self.device, total_s, verbose=True)
        manifest = {
            "fields": self.field_paths, "dx": self.dx, "boundary": self.boundary,
            "device": str(self.device), "total_ms": total_s * 1e3,
            "hpc_rank": _HPC_RANK if _IS_HPC else None, "hpc_world": _HPC_WORLD if _IS_HPC else None,
            "memory_config": {
                "chunk_vram_fraction": _CHUNK_VRAM_FRACTION, "chunk_ram_fraction": _CHUNK_RAM_FRACTION,
                "vram_shrink_trigger": _VRAM_SHRINK_TRIGGER, "ram_shrink_trigger": _RAM_SHRINK_TRIGGER,
                "row_stream_threshold_mb": _ROW_STREAM_THRESHOLD // 1024 // 1024,
                "hdf5_rdcc_nbytes": _HDF5_RDCC_NBYTES, "hdf5_rdcc_nslots": _HDF5_RDCC_NSLOTS, "pipe_depth": _PIPE_DEPTH,
            },
            "runs": [
                {
                    "index": i, "expr": r.expr_str, "elapsed_ms": r.elapsed_ms, "throughput_gbs": r.throughput_gbs,
                    "shape_in": list(r.shape_in), "shape_out": list(r.shape_out) if r.shape_out else None,
                    "nodes": r.node_count, "simplifications": r.simplifications, "output_file": r.out_path,
                    "perf": {
                        "mode": r.perf.mode, "input_gb": r.perf.input_gb, "output_gb": r.perf.output_gb, "t_compile_s": r.perf.t_compile,
                        "t_read_s": r.perf.t_read, "t_h2d_s": r.perf.t_h2d, "t_compute_s": r.perf.t_compute, "t_d2h_s": r.perf.t_d2h,
                        "t_write_s": r.perf.t_write, "t_total_s": r.perf.t_total, "est_flops": r.perf.est_flops,
                        "est_flops_note": "AST-derived upper bound; CSE may reduce actual ops", "tflops": r.perf.tflops(),
                        "peak_vram_gb": r.perf.peak_vram_gb, "n_chunks": r.perf.n_chunks,
                    } if r.perf else None,
                }
                for i, r in enumerate(results_list, 1)
            ],
        }
        if not _IS_HPC or _HPC_RANK == 0:
            with open(self.out_dir / "manifest.json", "w") as mf:
                json.dump(manifest, mf, indent=2)
        return results_list

    def cleanup(self) -> None:
        _safe_mm_clear(self.mm)
        _full_vram_release(self.device)
        _flush_vram(self.device, verbose=self.verbose)

# ----------------------------------------------------------------------
# Interactive REPL
# ----------------------------------------------------------------------
_OP_GROUPS: Dict[str, List[str]] = {
    "Arithmetic":    ["add", "sub", "mul", "div", "neg", "clamp"],
    "Elementwise":   ["exp", "log", "sqrt", "sin", "tanh"],
    "Reductions":    ["sum", "mean", "norm_l2", "variance", "entropy", "integrate", "cumulative_integral"],
    "Differential":  ["gradient", "gradient_nd", "divergence", "laplacian", "curl", "jacobian", "hessian", "mean_curvature", "surface_normals", "material_derivative"],
    "Spectral":      ["fft", "ifft", "spectral_gradient", "spectral_laplacian"],
    "Tensor/Matrix": ["trace", "determinant", "eigenvalues", "inverse", "deviatoric"],
}
_COMPILER_EXPANDED: Set[str] = {}
_OP_HINTS: Dict[str, str] = {
    "gradient": "∂f/∂x  (dim=N for other dims)", "gradient_nd": "∇f  all dims → (*shape, ndim)",
    "divergence": "∇·F  needs vector field (*shape, ndim)", "laplacian": "∇²f",
    "curl": "∇×F  3-D vector field (*shape, 3)", "spectral_gradient": "∂f/∂x via FFT  (periodic BC)",
    "spectral_laplacian": "∇²f via FFT  (spectrally accurate)", "fft": "N-D forward FFT  → real part",
    "ifft": "N-D inverse FFT  → real part", "jacobian": "∂fᵢ/∂xⱼ → (*spatial, C, ndim)  [streamed row-by-row]",
    "hessian": "∇∇f  → (*spatial, ndim, ndim)  [streamed row-by-row]", "mean_curvature": "H = ∇·(∇f/|∇f|)",
    "surface_normals": "unit normals of level set f=0", "integrate": "∫f dV  all dims, Simpson",
    "cumulative_integral": "running ∫ along dim 0", "trace": "tr(M)  last two dims D×D",
    "determinant": "det(M)", "eigenvalues": "λ = eigvalsh(M)", "inverse": "M⁻¹",
    "deviatoric": "M − (tr M / D)·I", "variance": "Var(f)", "entropy": "H(f)  treated as probability distribution",
    "norm_l2": "‖f‖₂",
}
_W = 72

def _show_operators() -> None:
    registered = {k for k in stenpy.OP_REGISTRY if not k.startswith("_")}
    print()
    print("  ╔" + "═" * (_W - 4) + "╗")
    print("  ║" + f"{'  AVAILABLE OPERATORS  ':^{_W - 4}}" + "║")
    print("  ╠" + "═" * (_W - 4) + "╣")
    for group, members in _OP_GROUPS.items():
        members = [m for m in members if m in registered or m in _COMPILER_EXPANDED]
        if not members: continue
        print("  ║  {:<{}}║".format(f"▸ {group}", _W - 6))
        for op in members:
            hint   = _OP_HINTS.get(op, "")
            label  = f"    {op}"
            max_h  = _W - 6 - len(label) - 4
            suffix = f"  — {hint[:max_h]}" if hint else ""
            print("  ║  {:<{}}║".format(label + suffix, _W - 6))
    print("  ║" + " " * (_W - 4) + "║")
    shown  = {op for grp in _OP_GROUPS.values() for op in grp}
    extras = sorted(registered - shown)
    if extras:
        print("  ║  {:<{}}║".format("▸ Other", _W - 6))
        for op in extras: print("  ║    {:<{}}║".format(op, _W - 8))
    print("  ║" + " " * (_W - 4) + "║")
    print("  ╠" + "═" * (_W - 4) + "╣")
    print("  ║  {:<{}}║".format("Multi-field:  f+g  |  gradient(f)*exp(g)  |  laplacian(f)-curl(g)", _W - 6))
    print("  ║  {:<{}}║".format("Auto-simplify:  divergence(gradient(f))  →  laplacian(f)", _W - 6))
    if _USER_VARS:
        uv_str = "User vars: " + ", ".join(f"{n}={uv.current:.4g}" for n, uv in _USER_VARS.items())
        print("  ║  {:<{}}║".format(uv_str, _W - 6))
    print("  ╚" + "═" * (_W - 4) + "╝")

def _show_fields(pipeline: "Pipeline") -> None:
    if not pipeline.fields: print("  (no fields loaded yet)"); return
    print(f"\n{'Name':<8}  {'Shape':<28}  {'GB':>7}  {'Mode':<12}  {'dx':>8}  Path")
    print(f"  {'─'*8}  {'─'*28}  {'─'*7}  {'─'*12}  {'─'*8}  {'─'*24}")
    for name, tensor in pipeline.fields.items():
        shape  = tuple(tensor.shape) if hasattr(tensor, "shape") else ("?",)
        mode   = "lazy/stream" if isinstance(tensor, stenpy.LazyField) else "eager/VRAM"
        path   = pipeline.field_paths.get(name, "<runtime>")
        sp_tup = pipeline.spacings.get(name, (float("nan"),))
        dx_val = float(sp_tup[0]) if sp_tup else float("nan")
        gb     = _tensor_gb(shape)
        print(f"  {name:<8}  {str(shape):<28}  {gb:>7.3f}  {mode:<12}  {dx_val:>8.5g}  {path}")
    if _NERD and pipeline.device.type == "cuda": print(f"\nVRAM  {_vram_bar(pipeline.device)}")

def _hpc_status(pipeline: "Pipeline") -> None:
    parts = [f"device={pipeline.device}", f"dx={pipeline.dx:.4g}", f"bc={pipeline.boundary}"]
    print("  HPC  " + "  |  ".join(parts))
    if pipeline.device.type == "cuda" and torch.cuda.is_available():
        print(f"  {_vram_bar(pipeline.device)}")
        bw = _gpu_peak_bw_gbs(pipeline.device)
        if not math.isnan(bw): print(f"  GPU peak mem BW : {bw:.0f} GB/s")

def _smart_define_field(pipeline: "Pipeline") -> None:
    print()
    print("  Enter one of:")
    print("    field_name          (e.g.  g  or  pressure)")
    print("    /path/to/file.h5    (name auto-assigned)")
    print("    name=/path/file.h5  (name and path together)")
    print()
    try: raw = input("  ❯ ").strip()
    except (EOFError, KeyboardInterrupt): print(); return
    if not raw: return
    name = path = None
    m = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)$', raw)
    if m: name, path = m.group(1).strip(), m.group(2).strip()
    elif os.sep in raw or raw.startswith(".") or raw.endswith((".h5", ".hdf5")):
        path = raw
        used = set(pipeline.fields.keys())
        name = next((c for c in "fghuvwxyzabcdeijklmnopqrst" if c not in used), f"field{len(pipeline.fields)}")
        print(f"  Auto-assigned name: '{name}'")
    elif re.match(r'^[A-Za-z_]\w*$', raw):
        name = raw
        try: path = input(f"  Path to HDF5 file for '{name}': ").strip()
        except (EOFError, KeyboardInterrupt): print(); return
    else: print(f"  ✗ Cannot parse '{raw}'."); return
    if not path: print("  ✗ No path given."); return
    if not Path(path).exists(): print(f"  ✗ File not found: {path!r}"); return
    try:
        pipeline.load_field(name, path)
        tensor = pipeline.fields[name]
        shape  = tuple(tensor.shape) if hasattr(tensor, "shape") else ("?",)
        mode   = "lazy/streamed" if isinstance(tensor, stenpy.LazyField) else "eager/in-VRAM"
        sp_tup = pipeline.spacings.get(name, (float("nan"),))
        gb     = _tensor_gb(shape)
        print(f"\n✓ Field '{name}' ready")
        print(f"  {'─'*40}")
        print(f"    shape     : {shape}")
        print(f"    size      : {gb:.3f} GB  (float64)")
        print(f"    mode      : {mode}")
        print(f"    spacing   : {sp_tup}")
        print(f"    origin    : {pipeline.origins.get(name)}")
        if pipeline.device.type == "cuda": print(f"    {_vram_bar(pipeline.device)}")
    except Exception as exc: print(f"  ✗ Could not load: {exc}")

def _operate(pipeline: "Pipeline") -> None:
    if not pipeline.fields: print("\n✗ No fields loaded — use [1] to define a field first."); return
    print()
    _show_fields(pipeline)
    print()
    _hpc_status(pipeline)
    if _USER_VARS: print(); _show_user_vars()
    print()
    print("  Compose freely using field names shown above.")
    if _USER_VARS: print(f"  User-defined variables available: {', '.join(_USER_VARS)}")
    print("  Separate multiple expressions with  ;  to run as a batch.")
    print("  Examples:  f+g   gradient(f)   laplacian(f)*2   curl(f)   f**2+g**2")
    print()
    try: raw = input("  Expression ❯ ").strip()
    except (EOFError, KeyboardInterrupt): print(); return
    if not raw: return
    raw_lower = raw.strip().lower()
    if re.match(r'^jacobian\s*\(\s*f\s*\)$', raw_lower):
        try:
            result = pipeline.run_jacobian(spatial_ndim=3)
            print(f"\n✓ Saved → {result.out_path}")
        except Exception as exc: print(f"  ✗ Error: {exc}")
        _prompt_vram_flush(pipeline.device)
        return
    if re.match(r'^hessian\s*\(\s*f\s*\)$', raw_lower):
        try:
            result = pipeline.run_hessian(spatial_ndim=3)
            print(f"\n✓ Saved → {result.out_path}")
        except Exception as exc: print(f"  ✗ Error: {exc}")
        _prompt_vram_flush(pipeline.device)
        return
    exprs = [e.strip() for e in raw.split(";") if e.strip()]
    if len(exprs) > 1: pipeline.run_batch(exprs)
    else:
        try:
            result = pipeline.run_expr(exprs[0])
            if result.out_path is not None:
                print(f"\n✓ Saved → {result.out_path}")
                print(f"  Throughput : {result.throughput_gbs:.3f} GB/s  │  {result.elapsed_ms:.0f} ms")
                if result.perf is not None and pipeline.verbose:
                    _print_perf_report([result.perf], pipeline.device, result.elapsed_ms / 1e3, verbose=True)
        except Exception as exc: print(f"  ✗ Error: {exc}")
        _prompt_vram_flush(pipeline.device)

def _settings(pipeline: "Pipeline") -> None:
    print(f"\n┌─── Settings ─────────────────────────────────────────────")
    print(f"  │  dx        : {pipeline.dx:.6g}")
    print(f"  │  boundary  : {pipeline.boundary}")
    print(f"  │  device    : {pipeline.device}")
    print(f"  │  row-stream: threshold {_ROW_STREAM_THRESHOLD/1024**2:.0f} MB/row  (OPS_ROW_STREAM_THRESHOLD)")
    if pipeline.device.type == "cuda":
        print(f"  │  {_vram_bar(pipeline.device)}")
        peak = torch.cuda.max_memory_allocated(pipeline.device) / 1024**3
        print(f"  │  peak VRAM : {peak:.3f} GB  (this session)")
    print(f"  └─────────────────────────────────────────────────────────────")
    print()
    print("  [1]  Change dx")
    print("  [2]  Change boundary condition")
    print("  [3]  Flush VRAM")
    print("  [4]  Back")
    try: c = input("  ❯ ").strip()
    except (EOFError, KeyboardInterrupt): return
    if c == "1":
        try:
            raw = input(f"  New dx [{pipeline.dx:.6g}]: ").strip()
            if raw: pipeline.dx = float(raw)
            print(f"  ✓ dx = {pipeline.dx:.6g}")
        except ValueError: print("  ✗ Invalid number.")
    elif c == "2":
        pipeline.boundary = _select_boundary()
        print(f"  ✓ boundary = {pipeline.boundary}")
    elif c == "3":
        _safe_mm_clear(pipeline.mm)
        try: pipeline.rt.flush_vram()
        except Exception as exc:
            if _NERD: print(f"  ⚠  rt.flush_vram() raised: {exc}")
        _flush_vram(pipeline.device, verbose=True)
        if pipeline.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pipeline.device)
            print("  ○  Peak VRAM counter reset.")

def _repl(pipeline: "Pipeline") -> None:
    print()
    print("═" * _W)
    print("  ops.py  │  HPC Field Pipeline  │  Interactive Mode")
    if _NERD: print("  ○  Nerd mode active  (OPS_NERD=1)  —  extra diagnostics enabled")
    print("═" * _W)
    _hpc_status(pipeline)
    _show_operators()
    while True:
        print()
        print("  ┌────────────────────────────────────────┐")
        print("  │                 MENU                   │")
        print("  ├────────────────────────────────────────┤")
        print("  │  [1]  Define / load a field            │")
        print("  │  [2]  Operate  (run an expression)     │")
        print("  │  [3]  List loaded fields               │")
        print("  │  [4]  HPC status & settings            │")
        print("  │  [5]  Show operator catalogue          │")
        print("  │  [6]  Define symbolic variable (t, …)  │")
        print("  │  [7]  Show symbolic variables          │")
        print("  │  [q]  Quit                             │")
        print("  └────────────────────────────────────────┘")
        if pipeline.fields:
            summary = "  ".join(f"{n}:{tuple(t.shape) if hasattr(t, 'shape') else '?'}" for n, t in pipeline.fields.items())
            print(f"\nFields ▸  {summary}")
        else: print("\nNo fields loaded yet.")
        if _USER_VARS:
            uv_summary = "  ".join(f"{n}=[{uv.start:.3g},{uv.end:.3g}]×{uv.steps}" for n, uv in _USER_VARS.items())
            print(f"  Vars   ▸  {uv_summary}")
        if pipeline.device.type == "cuda":
            used_gb = torch.cuda.memory_allocated(pipeline.device) / 1024**3
            if used_gb > 0.5:
                props_d = torch.cuda.get_device_properties(pipeline.device)
                pct = used_gb / max(props_d.total_memory / 1024**3, 1e-9) * 100
                sev = _sev_label(pct).strip()
                print(f"  VRAM   ▸  {used_gb:.2f} GB  [{sev}]")
        print()
        try: choice = input("  ❯ ").strip().lower()
        except (EOFError, KeyboardInterrupt): print("\nBye!"); break
        if not choice: continue
        elif choice == "1": _smart_define_field(pipeline)
        elif choice == "2": _operate(pipeline)
        elif choice == "3": _show_fields(pipeline)
        elif choice == "4": _settings(pipeline)
        elif choice == "5": _show_operators()
        elif choice == "6": _define_user_var(pipeline)
        elif choice == "7": _show_user_vars()
        elif choice in ("q", "quit", "exit"): print("  Bye!"); break
        else: print(f"  ✗ Unknown option '{choice}' — enter 1–7 or q.")

# ----------------------------------------------------------------------
# Demo field generator
# ----------------------------------------------------------------------
def _create_demo_field(path: str = "demo_field.h5", shape: Tuple[int, ...] = (64, 64, 64), dx: float = 0.1, seed: Optional[int] = None) -> str:
    _rank0_print(f"  Creating demo field  shape={shape}  dx={dx}")
    if seed is not None: torch.manual_seed(seed)
    coords = [torch.linspace(0, (n - 1) * dx, n, dtype=torch.float64) for n in shape]
    grids  = torch.meshgrid(*coords, indexing="ij")
    if len(shape) == 3: f = (torch.sin(2 * math.pi * grids[0]) * torch.cos(2 * math.pi * grids[1]) * torch.exp(-grids[2] ** 2))
    elif len(shape) == 2: f = torch.sin(2 * math.pi * grids[0]) * torch.cos(2 * math.pi * grids[1])
    else: f = torch.sin(2 * math.pi * grids[0])
    import h5py as _h5
    with _h5.File(path, "w") as hf:
        chunk_shape = (min(32, shape[0]),) + shape[1:]
        hf.create_dataset("data", data=f.numpy(), chunks=chunk_shape)
        hf.attrs["spacing"] = [dx] * len(shape)
        hf.attrs["origin"]  = [0.0] * len(shape)
    _rank0_print(f"  Demo saved → {path}  (chunks={chunk_shape}, no compression)")
    return path

# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Dual-mode HPC field pipeline", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=textwrap.dedent("""\
    Interactive (workstation):
    python main.py --input field.h5 --device cuda
    HPC batch (SLURM):
    srun -n 4 python main.py --field f=field.h5 \\
    --expr "gradient_nd(f)" --device cuda
    Row-stream threshold (default 512 MB/row):
    OPS_ROW_STREAM_THRESHOLD=268435456 python main.py ...
    Nerd mode:
    OPS_NERD=1 python main.py --input field.h5
    """))
    fg = p.add_argument_group("Field inputs")
    fg.add_argument("--field", "-F", metavar="NAME=PATH", action="append", dest="fields", default=[], help="--field f=file.h5  (repeatable)")
    fg.add_argument("--input", "-i", help="Shorthand: --field f=PATH")
    p.add_argument("--expr",      "-e", help="Expression(s); semicolons = batch")
    p.add_argument("--expr-file", "-f", help="File with one expression per line")
    p.add_argument("--project",   "-p", help="Project/output folder name")
    p.add_argument("--device",    "-d", default=None, help="cpu | cuda | cuda:N")
    p.add_argument("--dx",              type=float, default=1.0)
    p.add_argument("--boundary",        default="neumann", choices=["neumann", "dirichlet", "periodic", "reflect"])
    p.add_argument("--normalize",       action="store_true")
    p.add_argument("--repl",            action="store_true")
    p.add_argument("--demo",            action="store_true")
    p.add_argument("--demo-shape",      default="64,64,64")
    p.add_argument("--quiet", "-q",     action="store_true")
    return p

def _parse_field_args(args) -> Dict[str, str]:
    field_paths: Dict[str, str] = {}
    if args.input: field_paths["f"] = args.input
    for raw in (args.fields or []):
        if "=" not in raw: raise ValueError(f"--field must be NAME=PATH, got: {raw!r}")
        name, path = raw.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name.isidentifier(): raise ValueError(f"Field name must be a valid identifier: {name!r}")
        field_paths[name] = path
    return field_paths

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    if _IS_HPC:
        try:
            if stenpy.dist_init(): _rank0_print(f"  ✓ dist_init()  rank={_HPC_RANK}/{_HPC_WORLD}")
        except Exception as exc: _rank0_print(f"  ⚠  dist_init() failed: {exc}")
    _use_adv = os.environ.get("OPS_USE_ADV_MM", "1").lower() in ("1", "true", "yes")
    if _use_adv:
        try:
            stenpy.use_advanced_mm()
            _rank0_print("  ✓ Advanced MemoryManager active")
        except Exception as exc: _rank0_print(f"  ⚠  use_advanced_mm() failed: {exc}")
    elif os.environ.get("OPS_USE_ADV_MM", "0").lower() in ("1", "true", "yes"):
        try:
            stenpy.use_advanced_mm()
            print("  ✓ Advanced MemoryManager active")
        except Exception as exc: print(f"  ⚠  use_advanced_mm() failed: {exc}")

    if args.device:
        device_str = args.device
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            _rank0_print("  ⚠  CUDA not available — falling back to CPU")
            device_str = "cpu"
    elif _IS_HPC:
        device_str = str(_select_device_hpc())
        _rank0_print(f"  ✓ HPC device assignment: rank {_HPC_RANK} → {device_str}")
    else:
        device_str = str(_select_device())

    field_paths: Dict[str, str] = {}
    if args.demo:
        demo_shape = tuple(int(s) for s in args.demo_shape.split(","))
        demo_dx    = args.dx if args.dx != 1.0 else 0.1
        demo_path  = "demo_field.h5"
        if not _IS_HPC or _HPC_RANK == 0: field_paths["f"] = _create_demo_field(demo_path, shape=demo_shape, dx=demo_dx)
        if _IS_HPC:
            try:
                import torch.distributed as _dist
                if _dist.is_initialized(): _dist.barrier()
            except Exception: time.sleep(2.0)
            field_paths["f"] = demo_path
    else:
        try: field_paths = _parse_field_args(args)
        except ValueError as exc: parser.error(str(exc))
        if not field_paths and not _IS_HPC:
            print()
            print("  No fields specified.")
            print("  Options:")
            print("    Enter a file path           → bound to 'f'")
            print("    Type  multi  (or  m  or  b) → enter several NAME=PATH pairs")
            print()
            raw = input("  ❯ ").strip()
            if raw.lower() in ("multi", "m", "b"):
                print()
                print("  Enter  name=/path/file.h5  or just  /path/file.h5  (name auto-assigned)")
                print("  Empty line to finish.")
                while True:
                    line = input("  ❯ ").strip()
                    if not line: break
                    eq = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)$', line)
                    if eq: n, pth = eq.group(1).strip(), eq.group(2).strip()
                    elif os.sep in line or line.endswith(".h5"):
                        used = set(field_paths.keys())
                        n    = next((c for c in "fghuvwxyz" if c not in used), f"f{len(field_paths)}")
                        pth  = line
                        print(f"  Auto-name: '{n}'")
                    else:
                        print("  ✗ Use  name=/path  or a bare /path."); continue
                    if not Path(pth).exists(): print(f"  ✗ Not found: {pth!r}"); continue
                    field_paths[n] = pth
                    print(f"  ✓ '{n}' → {pth}")
            else:
                pth = raw
                while not Path(pth).exists():
                    print(f"  ✗ Not found: {pth!r}")
                    pth = input("  Path ❯ ").strip()
                field_paths["f"] = pth
        elif not field_paths and _IS_HPC:
            sys.exit(f"[rank {_HPC_RANK}] ERROR: HPC mode requires --field or --input.")

    expressions: List[str] = []
    if args.expr: expressions = [p.strip() for p in args.expr.split(";") if p.strip()]
    if args.expr_file:
        with open(args.expr_file) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#"): expressions.append(line)
    if _IS_HPC and not expressions: sys.exit(f"[rank {_HPC_RANK}] ERROR: HPC mode requires --expr or --expr-file.")

    pipeline = Pipeline(
        field_paths = field_paths, project = args.project, device = device_str,
        dx = args.dx, boundary = args.boundary, normalize = args.normalize, verbose = not args.quiet,
    )
    if expressions: pipeline.run_batch(expressions)
    if not _IS_HPC and (args.repl or not expressions):
        if sys.stdin.isatty(): _repl(pipeline)
        else: _rank0_print("  ℹ  stdin is not a terminal — skipping REPL.")
    pipeline.cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)
    except Exception as exc:
        try:
            import traceback as _tb
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            crash_dir = Path("outputs") / f"crash_{ts}"
            crash_dir.mkdir(parents=True, exist_ok=True)
            with open(crash_dir / "traceback.txt", "w") as fh:
                _tb.print_exc(file=fh)
            print(f"Unhandled exception; crash artifacts written to {crash_dir}")
        except Exception:
            pass
        raise
