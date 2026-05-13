# main.py

from __future__ import annotations
import argparse
import datetime
import json
import math
import os
import re
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch

# tqdm
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# SymPy required
try:
    import sympy as sp
    from sympy import Function, Symbol, Add, Mul, Pow
    from sympy.core.function import AppliedUndef, UndefinedFunction
except ImportError:
    sys.exit("SymPy is required: pip install sympy")

# stenpy
try:
    import stenpy
except ImportError:
    sys.exit("sten.py not found — place it in the same directory as main.py")

import h5py as _h5py

# ----------------------------------------------------------------------
# HPC environment detection
# ----------------------------------------------------------------------

_HPC_RANK_VARS = (
    "SLURM_PROCID",
    "PMI_RANK",
    "OMPI_COMM_WORLD_RANK",
    "MPI_RANK",
    "MV2_COMM_WORLD_RANK",
    "JSM_NAMESPACE_RANK",
)

_IS_HPC: bool = any(k in os.environ for k in _HPC_RANK_VARS)

def _env_rank() -> int:
    for k in _HPC_RANK_VARS:
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0

def _env_world() -> int:
    for k in ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE",
              "MPI_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return max(1, int(v))
            except ValueError:
                pass
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
        self.total = total or 0
        self.n     = 0
        self.desc  = desc
        self._t0   = time.perf_counter()
        _rank0_print(f"  {desc} …")

    def update(self, n=1):
        self.n += n
        pct     = 100 * self.n / self.total if self.total else 0
        elapsed = time.perf_counter() - self._t0
        if not _IS_HPC or _HPC_RANK == 0:
            print(f"\r  {self.desc}  {pct:5.1f}%  [{elapsed:.1f}s]",
                  end="", flush=True)

    def set_postfix_str(self, s, **kw): pass
    def set_postfix(self, **kw):        pass
    def close(self):
        if not _IS_HPC or _HPC_RANK == 0:
            print()
    def __enter__(self):  return self
    def __exit__(self, *a): self.close()

def _make_bar(total, desc="", unit="chunk", colour=None):
    if HAS_TQDM and (not _IS_HPC or _HPC_RANK == 0):
        return _tqdm(
            total        = total,
            desc         = f"  {desc}",
            unit         = unit,
            dynamic_ncols= True,
            colour       = colour or "cyan",
            bar_format   = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )
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
    used  = torch.cuda.memory_allocated(device)
    peak  = torch.cuda.max_memory_allocated(device)
    frac  = used  / total
    pfrac = peak  / total
    fi    = int(frac  * width)
    pi    = int(pfrac * width)
    bar   = "█" * fi + "▒" * max(0, pi - fi) + "░" * (width - max(fi, pi))
    pct   = frac * 100
    return (f"[{bar}] {used/1024**3:.2f}/{total/1024**3:.2f} GB  "
            f"{pct:.0f}%{_sev_label(pct)}")

def _hdf5_dissect(path: str) -> None:
    if not _NERD:
        return
    try:
        disk_mb = os.path.getsize(path) / 1024**2
    except OSError:
        disk_mb = 0.0
    _rank0_print(f"\n  ┌─── HDF5 dissection ─────────────────────────────────────────")
    _rank0_print(f"  │  file      : {Path(path).name}")
    _rank0_print(f"  │  disk size : {disk_mb:.2f} MB")
    try:
        with _h5py.File(path, "r") as f:
            if dict(f.attrs):
                _rank0_print(f"  │  file attrs:")
                for k, v in f.attrs.items():
                    _rank0_print(f"  │    {k} = {v}")
            def _visit(name: str, obj: Any) -> None:
                if not isinstance(obj, _h5py.Dataset):
                    return
                nbytes   = obj.dtype.itemsize * int(np.prod(obj.shape))
                stored   = obj.id.get_storage_size()
                ratio    = stored / max(nbytes, 1)
                comp     = obj.compression or "none"
                comp_lvl = f"/{obj.compression_opts}" if obj.compression_opts is not None else ""
                ram_gb   = nbytes / 1024**3
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

def _budget_explain(
    full_shape:  Tuple[int, ...],
    device:      torch.device,
    n_fields:    int,
    chunk_rows:  int,
    n_chunks:    int,
) -> None:
    if not _NERD:
        return
    row_bytes = math.prod(full_shape[1:]) * 8
    dim0      = full_shape[0]
    cap       = max(1, dim0 // 4)
    _rank0_print(f"\n  ┌─── Chunk budget ─────────────────────────────────────────────")
    if device.type == "cuda" and torch.cuda.is_available():
        props     = torch.cuda.get_device_properties(device)
        total     = props.total_memory
        used      = torch.cuda.memory_allocated(device)
        free      = total - used
        budget_pf = int(free * _CHUNK_VRAM_FRACTION) // max(n_fields, 1)
        raw_rows  = max(1, budget_pf // max(row_bytes, 1))
        _rank0_print(f"  │  free VRAM          : {free/1024**3:.3f} GB")
        _rank0_print(f"  │  VRAM fraction      : {_CHUNK_VRAM_FRACTION:.0%}")
        _rank0_print(f"  │  budget / field     : {budget_pf/1024**2:.1f} MB")
        _rank0_print(f"  │  row size           : {row_bytes/1024**2:.3f} MB  "
              f"({math.prod(full_shape[1:])} elements × 8 B)")
        _rank0_print(f"  │  raw rows           : {raw_rows}")
        _rank0_print(f"  │  dim-0 cap (÷4)     : {cap}")
    else:
        _rank0_print(f"  │  device             : CPU")
        _rank0_print(f"  │  row size           : {row_bytes/1024**2:.3f} MB")
    _rank0_print(f"  │  ──────────────────────────────────────────────────────────")
    _rank0_print(f"  │  chunk_rows         : {chunk_rows}  "
          f"({chunk_rows * row_bytes * n_fields / 1024**3:.3f} GB / iter)")
    _rank0_print(f"  │  n_chunks           : {n_chunks}  (dim-0 {dim0} rows)")
    _rank0_print(f"  └─────────────────────────────────────────────────────────────")

def _nerd(msg: str) -> None:
    if _NERD and (not _IS_HPC or _HPC_RANK == 0):
        print(f"  ○  {msg}")

# ----------------------------------------------------------------------
# Performance instrumentation
# ----------------------------------------------------------------------
_FLOP_PER_ELEMENT: Dict[str, float] = {
    "add":                1.0,
    "sub":                1.0,
    "mul":                1.0,
    "div":                4.0,
    "neg":                1.0,
    "clamp":              2.0,
    "exp":               20.0,
    "log":               20.0,
    "sqrt":               4.0,
    "sin":               15.0,
    "tanh":              25.0,
    "sum":                1.0,
    "mean":               2.0,
    "norm_l2":            3.0,
    "variance":           4.0,
    "entropy":            6.0,
    "integrate":          3.0,
    "cumulative_integral":3.0,
    "gradient":           6.0,
    "gradient_nd":        6.0,
    "divergence":         6.0,
    "laplacian":         14.0,
    "curl":              18.0,
    "hessian":           12.0,
    "mean_curvature":    30.0,
    "surface_normals":   18.0,
    "material_derivative":22.0,
    "spectral_gradient":  0.0,
    "spectral_laplacian": 0.0,
    "fft":                0.0,
    "ifft":               0.0,
    "trace":              2.0,
    "determinant":        6.0,
    "eigenvalues":       12.0,
    "inverse":           10.0,
    "deviatoric":         5.0,
}

def _estimate_flops(expr_str: str,
                    output_shape: Optional[Tuple],
                    input_shape:  Optional[Tuple] = None) -> float:
    if output_shape is None:
        return 0.0
    n_elem = max(1, math.prod(int(s) for s in output_shape))
    n_in   = max(1, math.prod(int(s) for s in input_shape)) if input_shape else n_elem

    ops_found: List[str] = []
    for op in _FLOP_PER_ELEMENT:
        if re.search(rf'\b{re.escape(op)}\b', expr_str):
            ops_found.append(op)

    if not ops_found:
        return float(n_elem)

    total = 0.0
    for op in ops_found:
        fpe = _FLOP_PER_ELEMENT[op]
        if fpe == 0.0:
            total += 5.0 * n_in * max(1.0, math.log2(n_in))
        else:
            total += fpe * n_elem
    return total

def _gpu_peak_bw_gbs(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return float("nan")
    props = torch.cuda.get_device_properties(device)
    if hasattr(props, 'memory_clock_rate'):
        mem_clock_khz = props.memory_clock_rate
    elif hasattr(props, 'clock_rate'):
        mem_clock_khz = props.clock_rate
    else:
        return float("nan")
    if hasattr(props, 'memory_bus_width'):
        bus_width_bits = props.memory_bus_width
    else:
        return float("nan")
    return (2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8)) / 1e9

def _gpu_name(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "CPU"
    p = torch.cuda.get_device_properties(device)
    return f"{p.name}  ({p.total_memory/1024**3:.0f} GB VRAM)"

@dataclass
class PerfStats:
    """Performance telemetry for a single expression run."""
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
    n_chunks:        int = 0
    chunk_rows:      int = 0
    peak_vram_gb:    float = 0.0
    vram_total_gb:   float = 0.0
    est_flops:       float = 0.0

    def read_bw_gbs(self)    -> float:
        return self.input_gb  / self.t_read    if self.t_read    > 1e-9 else 0.0
    def write_bw_gbs(self)   -> float:
        return self.output_gb / self.t_write   if self.t_write   > 1e-9 else 0.0
    def h2d_bw_gbs(self)     -> float:
        return self.input_gb  / self.t_h2d     if self.t_h2d     > 1e-9 else 0.0
    def d2h_bw_gbs(self)     -> float:
        return self.output_gb / self.t_d2h     if self.t_d2h     > 1e-9 else 0.0
    def compute_bw_gbs(self) -> float:
        return self.input_gb  / self.t_compute if self.t_compute > 1e-9 else 0.0
    def tflops(self)         -> float:
        return (self.est_flops / 1e12) / self.t_compute if self.t_compute > 1e-9 else 0.0
    def overall_bw_gbs(self) -> float:
        return (self.input_gb + self.output_gb) / self.t_total if self.t_total > 1e-9 else 0.0

def _arr_stats(vals: List[float]) -> Tuple[float, float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    a   = np.array(vals, dtype=float)
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
    if v == 0.0: return "  —  "
    if v >= 1.0: return f"{v:.3f} TFLOP/s"
    if v >= 1e-3: return f"{v*1000:.2f} GFLOP/s"
    return f"{v*1e6:.1f} MFLOP/s"

def _pct_bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)

def _print_perf_report(
    stats_list:  List[PerfStats],
    device:      torch.device,
    total_wall_s: float,
    verbose:     bool = True,
) -> None:
    if not verbose or (_IS_HPC and _HPC_RANK != 0) or not stats_list:
        return

    W = 76
    def _hdr(title: str) -> None:
        print(f"  ┌─── {title} {'─' * max(0, W - 6 - len(title))}┐")
    def _ftr() -> None:
        print(f"  └{'─' * W}┘")
    def _row(label: str, value: str, note: str = "") -> None:
        note_s = f"  {note}" if note else ""
        inner  = f"  {label:<30}{value:<20}{note_s}"
        pad    = W - len(inner)
        print(f"  │{inner}{' ' * max(0, pad)}│")
    def _divider() -> None:
        print(f"  │{'─' * W}│")
    def _blank() -> None:
        print(f"  │{' ' * W}│")

    agg_input_gb   = sum(s.input_gb   for s in stats_list)
    agg_output_gb  = sum(s.output_gb  for s in stats_list)
    agg_flops      = sum(s.est_flops  for s in stats_list)
    agg_t_read     = sum(s.t_read     for s in stats_list)
    agg_t_h2d      = sum(s.t_h2d      for s in stats_list)
    agg_t_compile  = sum(s.t_compile  for s in stats_list)
    agg_t_compute  = sum(s.t_compute  for s in stats_list)
    agg_t_d2h      = sum(s.t_d2h      for s in stats_list)
    agg_t_write    = sum(s.t_write    for s in stats_list)
    agg_chunks     = sum(s.n_chunks   for s in stats_list)

    peak_vram_gb   = max((s.peak_vram_gb  for s in stats_list), default=0.0)
    vram_total_gb  = max((s.vram_total_gb for s in stats_list), default=0.0)

    gpu_peak_bw = _gpu_peak_bw_gbs(device)
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
    _row("Wall time (total):",        _fmt_s(total_wall_s))
    _row("Data read from disk:",      _fmt_gb(agg_input_gb),
         f"→ {_fmt_bw(agg_read_bw)} read")
    _row("Data written to disk:",     _fmt_gb(agg_output_gb),
         f"→ {_fmt_bw(agg_write_bw)} write")
    _row("Aggregate data moved:",     _fmt_gb(agg_input_gb + agg_output_gb))

    if agg_flops > 0:
        if agg_flops >= 1e12:
            flop_str = f"{agg_flops/1e12:.3f} TFLOP (est.)"
        elif agg_flops >= 1e9:
            flop_str = f"{agg_flops/1e9:.2f} GFLOP (est.)"
        else:
            flop_str = f"{agg_flops/1e6:.1f} MFLOP (est.)"
        _row("Estimated FLOPs:", flop_str)
        _row("Achieved compute rate:", _fmt_tf(agg_tflops_s))

    if not math.isnan(gpu_peak_bw):
        _row("GPU peak mem bandwidth:", f"{gpu_peak_bw:.0f} GB/s  (theoretical)")
        if not math.isnan(mem_util_pct):
            bar = _pct_bar(mem_util_pct / 100, width=16)
            _row("GPU BW utilisation:",
                 f"{mem_util_pct:.1f}%  [{bar}]")
    _ftr()

    _hdr("STAGE BREAKDOWN  (cumulative over all runs)")
    inner = f"  {'Stage':<30}{'Time':>10}   {'Share':>5}   {'Bandwidth / Rate'}"
    print(f"  │{inner}{' ' * max(0, W - len(inner))}│")
    _divider()

    def _stage_row(label: str, t: float, bw_label: str = "") -> None:
        if t < 1e-9:
            return
        frac  = t / total_stages if total_stages > 1e-9 else 0.0
        bar   = _pct_bar(frac, width=10)
        inner = (f"  {label:<30}{_fmt_s(t):>10}   {_pct(t):>5}"
                 f"   [{bar}]  {bw_label}")
        pad   = W - len(inner)
        print(f"  │{inner}{' ' * max(0, pad)}│")

    _stage_row("HDF5 read  (disk → CPU)",   agg_t_read, _fmt_bw(agg_read_bw))
    _stage_row("H→D transfer  (CPU → GPU)", agg_t_h2d, _fmt_bw(agg_h2d_bw) + " (PCIe/NVLink)")
    _stage_row("Compile / graph trace",     agg_t_compile)
    _stage_row("GPU compute",               agg_t_compute, _fmt_tf(agg_tflops_s))
    _stage_row("D→H transfer  (GPU → CPU)", agg_t_d2h, _fmt_bw(agg_d2h_bw) + " (PCIe/NVLink)")
    _stage_row("HDF5 write  (CPU → disk)",  agg_t_write, _fmt_bw(agg_write_bw))

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
            cv_note = ("uniform" if cv < 0.05
                       else "moderate variance" if cv < 0.15
                       else "HIGH variance — check input regularity")
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
        expr_short = s.expr_str[:col_e]
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
        _row("Peak VRAM used:",
             f"{peak_vram_gb:.3f} / {vram_total_gb:.1f} GB",
             f"({vram_pct:.1f}%)  [{bar}]")
        if has_chunks and stats_list:
            s0 = stats_list[0]
            if s0.n_chunks and s0.input_gb:
                chunk_gb = s0.input_gb / s0.n_chunks
                _row("Est. chunk working-set:",
                     f"{_fmt_gb(chunk_gb)} × {s0.n_chunks} chunks")
        _ftr()

    print(f"  {'═' * W}")
    print()

# ----------------------------------------------------------------------
# Symbolic operator vocabulary
# ----------------------------------------------------------------------
_ALIAS: Dict[str, str] = {
    "gradient":            "gradient",
    "grad":                "gradient",
    "gradient_nd":         "gradient_nd",
    "grad_nd":             "gradient_nd",
    "divergence":          "divergence",
    "div":                 "divergence",
    "laplacian":           "laplacian",
    "laplace":             "laplacian",
    "lap":                 "laplacian",
    "curl":                "curl",
    "spectral_gradient":   "spectral_gradient",
    "spectral_laplacian":  "spectral_laplacian",
    "fft":                 "fft",
    "ifft":                "ifft",
    "add":                 "add",
    "sub":                 "sub",
    "mul":                 "mul",
    "div_op":              "div",
    "neg":                 "neg",
    "clamp":               "clamp",
    "exp":                 "exp",
    "log":                 "log",
    "sqrt":                "sqrt",
    "sin":                 "sin",
    "tanh":                "tanh",
    "sum":                 "sum",
    "mean":                "mean",
    "norm_l2":             "norm_l2",
    "variance":            "variance",
    "entropy":             "entropy",
    "hessian":             "hessian",
    "material_derivative": "material_derivative",
    "mean_curvature":      "mean_curvature",
    "surface_normals":     "surface_normals",
    "integrate":           "integrate",
    "cumulative_integral": "cumulative_integral",
    "trace":               "trace",
    "determinant":         "determinant",
    "eigenvalues":         "eigenvalues",
    "inverse":             "inverse",
    "deviatoric":          "deviatoric",
}

_NO_STENCIL_OPS: Set[str] = {
    "add", "sub", "mul", "div", "neg", "clamp",
    "exp", "log", "sqrt", "sin", "tanh",
    "sum", "mean", "norm_l2", "min_max", "variance", "entropy",
    "trace", "determinant", "eigenvalues", "inverse", "deviatoric",
    "fft", "ifft", "covariance", "correlation",
}

_SHORT_OP: Dict[str, str] = {
    "spectral_gradient":   "spec_grad",
    "spectral_laplacian":  "spec_lap",
    "cumulative_integral": "cumintg",
    "material_derivative": "matd",
    "mean_curvature":      "curv",
    "surface_normals":     "normals",
    "determinant":         "det",
    "eigenvalues":         "eig",
    "gradient_nd":         "grad_nd",
    "divergence":          "div",
    "laplacian":           "lap",
    "gradient":            "grad",
    "variance":            "var",
    "entropy":             "ent",
    "integrate":           "intg",
    "inverse":             "inv",
    "deviatoric":          "dev",
    "hessian":             "hess",
    "norm_l2":             "norm",
    "trace":               "tr",
    "clamp":               "clamp",
    "curl":                "curl",
    "tanh":                "tanh",
    "sqrt":                "sqrt",
    "mean":                "mean",
    "neg":                 "neg",
    "exp":                 "exp",
    "log":                 "log",
    "sin":                 "sin",
    "sum":                 "sum",
    "fft":                 "fft",
    "add":                 "add",
    "sub":                 "sub",
    "mul":                 "mul",
    "ifft":                "ifft",
}

_SIMPLIFY_RULES: Dict[Tuple[str, str], str] = {
    ("divergence", "gradient_nd"): "laplacian",
    ("divergence", "gradient"):    "laplacian",
}

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
    def values(self) -> np.ndarray:
        return np.linspace(self.start, self.end, self.steps)

    @property
    def step_size(self) -> float:
        return 0.0 if self.steps < 2 else (self.end - self.start) / (self.steps - 1)

_USER_VARS: Dict[str, UserVar] = {}

def _define_user_var(pipeline: Optional["Pipeline"] = None) -> Optional[UserVar]:
    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │         DEFINE A SYMBOLIC VARIABLE               │")
    print("  │  Examples:  t (time)   omega (frequency)  k      │")
    print("  └──────────────────────────────────────────────────┘")
    try:
        name = input("  Variable name  ❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); return None
    if not name or not re.match(r'^[A-Za-z_]\w*$', name):
        print("  ✗ Invalid name."); return None

    if pipeline is not None and name in pipeline.fields:
        print(f"  ✗ '{name}' is already a loaded field — choose a different name.")
        return None

    if name in _USER_VARS:
        print(f"  ℹ  '{name}' already defined — overwriting.")
    try:
        start = float(input(f"  Start value for {name}  ❯ ").strip())
        end   = float(input(f"  End   value for {name}  ❯ ").strip())
        raw_s = input(f"  Number of sample points [1 = single value]  ❯ ").strip()
        steps = max(1, int(raw_s) if raw_s else 1)
    except (ValueError, EOFError, KeyboardInterrupt):
        print("  ✗ Aborted."); return None

    uv = UserVar(name=name, start=start, end=end, steps=steps, current=start)
    _USER_VARS[name] = uv
    print(f"\n  ✓ '{name}'  range [{start}, {end}]  steps={steps}")
    if steps > 1:
        print(f"     step size = {uv.step_size:.6g}")
    return uv

def _show_user_vars() -> None:
    if not _USER_VARS:
        print("  (no user-defined variables yet — use [6] to add one)")
        return
    print(f"\n  {'Name':<12}  {'Start':>10}  {'End':>10}  {'Steps':>6}  {'Current':>12}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*12}")
    for uv in _USER_VARS.values():
        print(f"  {uv.name:<12}  {uv.start:>10.4g}  {uv.end:>10.4g}"
              f"  {uv.steps:>6}  {uv.current:>12.6g}")

# ----------------------------------------------------------------------
# Expression parser 
# ----------------------------------------------------------------------

def _build_sympy_namespace(field_names: Optional[Set[str]] = None) -> Dict[str, Any]:
    ns: Dict[str, Any] = {}
    for sym in ("f", "g", "h", "u", "v", "x", "y", "z", "t"):
        ns[sym] = sp.Symbol(sym)
    if field_names:
        for name in field_names:
            if name not in ns:
                ns[name] = sp.Symbol(name)
    for name in _USER_VARS:
        if name not in ns:
            ns[name] = sp.Symbol(name)
    seen: Set[str] = set()
    for alias, canonical in _ALIAS.items():
        if alias not in seen:
            ns[alias] = sp.Function(alias)
            seen.add(alias)
        if canonical not in seen:
            ns[canonical] = sp.Function(canonical)
            seen.add(canonical)
    ns["pi"] = sp.pi
    ns["E"]  = sp.E
    return ns

def parse_expression(expr_str: str,
                     field_names: Optional[Set[str]] = None) -> sp.Expr:
    def _encode_kwargs(s: str) -> str:
        pattern = re.compile(r'(\w+)\s*\(([^()]*?)\)', re.DOTALL)
        def _repl(m: re.Match) -> str:
            fname    = m.group(1)
            raw_args = m.group(2)
            pos_args, kw_parts = [], []
            for part in raw_args.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    kw_parts.append(f"_kw_{k.strip()}_{v.strip()}")
                else:
                    pos_args.append(part)
            return f"{fname}({', '.join(pos_args + kw_parts)})"
        prev = None
        while prev != s:
            prev = s; s = pattern.sub(_repl, s)
        return s

    cleaned = _encode_kwargs(expr_str.strip())
    ns = _build_sympy_namespace(field_names=field_names)
    for kf in re.findall(r'_kw_\w+', cleaned):
        ns[kf] = sp.Symbol(kf)
    try:
        return sp.sympify(cleaned, locals=ns)
    except Exception as exc:
        raise ValueError(
            f"Could not parse '{expr_str}': {exc}\n"
            f"Processed form: '{cleaned}'"
        ) from exc

# ----------------------------------------------------------------------
# AST -> ops.Graph compiler
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

def _decode_kwparams(func_name: str,
                     args: Tuple) -> Tuple[str, Dict[str, Any], List]:
    real_args, kwargs = [], []
    for a in args:
        s = str(a)
        if s.startswith("_kw_"):
            rest = s[4:]; idx = rest.index("_")
            key  = rest[:idx]; val = rest[idx + 1:]
            try:    val = int(val)
            except ValueError:
                try: val = float(val)
                except ValueError: pass
            kwargs[key] = val
        else:
            real_args.append(a)
    canonical = _ALIAS.get(func_name.lower(), func_name.lower())
    return canonical, kwargs, real_args

def _op_params(canonical: str, dx: float, boundary: str) -> Dict[str, Any]:
    if canonical in _NO_STENCIL_OPS:
        return {}
    return {"dx": dx, "boundary": boundary}

def _compile_node(expr: sp.Expr, ctx: _CompileCtx) -> str:
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name in ctx.field_map:
            return ctx.field_map[name]
        if name in ("f", "g", "h", "u", "v"):
            return ctx.src_id
        ctx.warnings.append(f"Unknown symbol '{name}' — using fallback field")
        return ctx.src_id

    if isinstance(expr, (sp.Number, sp.Integer, sp.Float, sp.Rational)):
        val = float(expr)
        key = ("_scalar", val)
        if key in ctx.cse_cache:
            return ctx.cse_cache[key]
        nid = ctx.graph.add("_constant", (),
                            {"value": torch.tensor(val, dtype=torch.float64)})
        ctx.cse_cache[key] = nid
        return nid

    if isinstance(expr, sp.core.function.AppliedUndef):
        func_name = type(expr).__name__
        canonical, extra_kw, real_args = _decode_kwparams(func_name, expr.args)
        child_ids = tuple(_compile_node(a, ctx) for a in real_args)
        params: Dict[str, Any] = _op_params(canonical, ctx.dx, ctx.boundary)
        params.update(extra_kw)

        if len(child_ids) == 1:
            child_node = ctx.graph._nodes.get(child_ids[0])
            if child_node:
                rule_key = (canonical, child_node.op_name)
                if rule_key in _SIMPLIFY_RULES:
                    replacement = _SIMPLIFY_RULES[rule_key]
                    ctx.warnings.append(
                        f"Simplified {canonical}({child_node.op_name}(f))"
                        f" → {replacement}(f)"
                    )
                    canonical = replacement
                    child_ids = child_node.input_ids

        frozen = tuple(sorted(
            (k, v) for k, v in params.items()
            if isinstance(v, (int, float, str, bool))
        ))
        cse_key = (canonical, child_ids, frozen)
        if cse_key in ctx.cse_cache:
            return ctx.cse_cache[cse_key]

        if canonical not in stenpy.OP_REGISTRY:
            raise ValueError(
                f"Unknown operator '{canonical}' (from '{func_name}').\n"
                f"Available: {sorted(stenpy.OP_REGISTRY.keys())}"
            )
        nid = ctx.graph.add(canonical, child_ids, params)
        ctx.cse_cache[cse_key] = nid
        return nid

    if isinstance(expr, sp.Add):
        ids = [_compile_node(o, ctx) for o in expr.args]
        result_id = ids[0]
        for nxt in ids[1:]:
            cse_key = ("add", (result_id, nxt), ())
            if cse_key in ctx.cse_cache:
                result_id = ctx.cse_cache[cse_key]
            else:
                result_id = ctx.graph.add("add", (result_id, nxt), {})
                ctx.cse_cache[cse_key] = result_id
        return result_id

    if isinstance(expr, sp.Mul):
        operands = list(expr.args)
        if sp.Integer(-1) in operands and len(operands) == 2:
            other    = [o for o in operands if o != sp.Integer(-1)][0]
            child_id = _compile_node(other, ctx)
            cse_key  = ("neg", (child_id,), ())
            if cse_key in ctx.cse_cache:
                return ctx.cse_cache[cse_key]
            nid = ctx.graph.add("neg", (child_id,), {})
            ctx.cse_cache[cse_key] = nid
            return nid
        ids = [_compile_node(o, ctx) for o in operands]
        result_id = ids[0]
        for nxt in ids[1:]:
            cse_key = ("mul", (result_id, nxt), ())
            if cse_key in ctx.cse_cache:
                result_id = ctx.cse_cache[cse_key]
            else:
                result_id = ctx.graph.add("mul", (result_id, nxt), {})
                ctx.cse_cache[cse_key] = result_id
        return result_id

    if isinstance(expr, sp.Pow):
        base_id = _compile_node(expr.args[0], ctx)
        exp_val = float(expr.args[1])
        if abs(exp_val - 0.5) < 1e-9:
            cse_key = ("sqrt", (base_id,), ())
            if cse_key in ctx.cse_cache:
                return ctx.cse_cache[cse_key]
            nid = ctx.graph.add("sqrt", (base_id,), {})
            ctx.cse_cache[cse_key] = nid
            return nid
        log_id    = ctx.graph.add("log",  (base_id,), {})
        const_nid = _compile_node(sp.Float(exp_val), ctx)
        mul_id    = ctx.graph.add("mul",  (log_id, const_nid), {})
        return     ctx.graph.add("exp",  (mul_id,), {})

    raise ValueError(
        f"Unsupported SymPy node {type(expr).__name__}: {expr}\n"
        "Use recognised operators and arithmetic (+, *, **)."
    )

def compile_expression(
    expr_str:  str,
    dx:        float = 1.0,
    boundary:  str   = "neumann",
    field_map: Optional[Dict[str, Any]] = None,
) -> Tuple[stenpy.Graph, str, List[str], Dict[str, str]]:
    if field_map is None:
        field_map = {"f": torch.zeros(1, dtype=torch.float64)}

    field_names = set(field_map.keys())
    sympy_expr  = parse_expression(expr_str, field_names=field_names)

    g   = stenpy.Graph()
    ctx = _CompileCtx(graph=g, dx=dx, boundary=boundary)

    for name, tensor in field_map.items():
        nid = g.add("_constant", (), {"value": tensor})
        ctx.field_map[name] = nid

    ctx.src_id = ctx.field_map.get("f", next(iter(ctx.field_map.values())))
    sink_id    = _compile_node(sympy_expr, ctx)

    return g, sink_id, ctx.warnings, dict(ctx.field_map)

# ----------------------------------------------------------------------
# HPC helpers
# ----------------------------------------------------------------------
def _tensor_gb(shape: Tuple) -> float:
    try:
        return math.prod(int(s) for s in shape) * 8 / 1024**3
    except Exception:
        return 0.0

def _gpu_free_gb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return float("inf")
    props = torch.cuda.get_device_properties(device)
    return (props.total_memory - torch.cuda.memory_allocated(device)) / 1024**3

def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        print("  Device: CPU  (no CUDA GPU found)")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    print(f"\n  {'─'*62}")
    print("  SELECT COMPUTE DEVICE")
    print(f"  {'─'*62}")
    print("    [0]  CPU")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"    [{i+1}]  CUDA:{i}  {p.name}  "
              f"{p.total_memory/1024**3:.1f} GB VRAM  "
              f"{p.multi_processor_count} SMs")
    print(f"  {'─'*62}")
    while True:
        try:
            raw = input("  Choice [0 = CPU]: ").strip() or "0"
        except (EOFError, KeyboardInterrupt):
            return torch.device("cpu")
        try:
            idx = int(raw)
        except ValueError:
            print(f"  ✗ Enter 0–{n}"); continue
        if idx == 0:
            return torch.device("cpu")
        if 1 <= idx <= n:
            return torch.device(f"cuda:{idx - 1}")
        print(f"  ✗ Enter 0–{n}")

def _select_device_hpc() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = 0
    for k in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
              "MPI_LOCALRANKID", "PMI_LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try: local_rank = int(v); break
            except ValueError: pass
    else:
        local_rank = _HPC_RANK

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        return torch.device("cpu")
    device_idx = local_rank % n_gpu
    return torch.device(f"cuda:{device_idx}")

def _select_boundary() -> str:
    opts = {"1": "neumann", "2": "dirichlet", "3": "periodic", "4": "reflect"}
    print("\n  Boundary condition:")
    print("    [1]  Neumann    — replicate edge  ← default")
    print("    [2]  Dirichlet  — zero pad")
    print("    [3]  Periodic   — wrap around")
    print("    [4]  Reflect")
    try:
        raw = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        return "neumann"
    return opts.get(raw, "neumann")

# ----------------------------------------------------------------------
# VRAM management
# ----------------------------------------------------------------------
def _flush_vram(device: torch.device, verbose: bool = True) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
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
        pct   = after_alloc / props.total_memory * 100
        print(f"\n  ♻  VRAM flush")
        print(f"     allocated  : freed {freed_alloc:.3f} GB  │  now {after_alloc/1024**3:.3f} GB")
        print(f"     reserved   : freed {freed_res:.3f} GB  │  now {after_res/1024**3:.3f} GB")
        print(f"     gauge      : {_vram_bar(device)}")
        if _NERD:
            print(f"     before     : alloc {before_alloc/1024**3:.3f} GB  "
                  f"res {before_res/1024**3:.3f} GB")
            print(f"     headroom   : {(total - after_alloc/1024**3):.3f} GB free")

def _prompt_vram_flush(device: torch.device) -> None:
    if _IS_HPC or device.type != "cuda":
        return
    used_gb = torch.cuda.memory_allocated(device) / 1024**3
    if used_gb < 0.1:
        return
    pct = used_gb / (torch.cuda.get_device_properties(device).total_memory / 1024**3) * 100
    sev = _sev_label(pct).strip()
    print(f"\n  ♻  VRAM holds {used_gb:.2f} GB  [{sev}]")
    try:
        ans = input("  Flush VRAM now? [Y/n] ❯ ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if ans in ("", "y", "yes"):
        _flush_vram(device, verbose=True)
    else:
        print("  VRAM kept — remember to flush before large operations.")

# ----------------------------------------------------------------------
# Multi-field domain merging
# ----------------------------------------------------------------------
def _merge_domain(
    spacings: Dict[str, Tuple[float, ...]],
    origins:  Dict[str, Tuple[float, ...]],
    shapes:   Dict[str, Tuple[int, ...]],
) -> Tuple[float, Tuple[float, ...], Tuple[float, ...]]:
    if not spacings:
        return 1.0, (0.0,), (1.0,)
    all_sp = [s for sp_tuple in spacings.values() for s in sp_tuple]
    dx     = float(max(all_sp)) if all_sp else 1.0
    ndims  = min(len(sp) for sp in spacings.values())
    merged_origin = tuple(
        max(origins[n][d] for n in origins if d < len(origins[n]))
        for d in range(ndims)
    )
    ends: List[List[float]] = []
    for name, sp in spacings.items():
        sh   = shapes.get(name, ())
        n_sp = min(len(sp), len(sh), ndims)
        ends.append([origins[name][d] + (sh[d] - 1) * sp[d] for d in range(n_sp)])
    merged_end = tuple(min(e[d] for e in ends) for d in range(ndims))
    return dx, merged_origin, merged_end

# ----------------------------------------------------------------------
# GPU-saturating direct-HDF5 chunked executor
# ----------------------------------------------------------------------
_CHUNK_VRAM_FRACTION = 0.12
_PIPELINE_DEPTH      = 2

def _compute_chunk_rows(
    full_shape: Tuple[int, ...],
    device:     torch.device,
    n_fields:   int = 1,
) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        row_bytes = math.prod(full_shape[1:]) * 8
        return max(1, (256 * 1024 * 1024) // max(row_bytes, 1))
    try:
        props     = torch.cuda.get_device_properties(device)
        free_vram = props.total_memory - torch.cuda.memory_allocated(device)
        budget    = int(free_vram * _CHUNK_VRAM_FRACTION) // max(n_fields, 1)
        row_bytes = math.prod(full_shape[1:]) * 8
        rows      = max(1, budget // max(row_bytes, 1))
        rows      = min(rows, max(1, full_shape[0] // 4))
        return rows
    except Exception:
        return max(1, full_shape[0] // 16)

def _open_hdf5_field(path: str) -> Tuple["_h5py.Dataset", Tuple[int, ...]]:
    f   = _h5py.File(path, "r")
    key = next(
        (k for k in ("data", "field") if k in f and isinstance(f[k], _h5py.Dataset)),
        next(k for k in f if isinstance(f[k], _h5py.Dataset)),
    )
    ds = f[key]
    return ds, tuple(ds.shape)

def _hdf5_direct_chunked_run(
    expr_str:    str,
    field_paths: Dict[str, str],
    scalar_map:  Dict[str, torch.Tensor],
    dx:          float,
    boundary:    str,
    out_path:    str,
    spacing:     Tuple,
    origin:      Tuple,
    device:      torch.device,
    verbose:     bool = True,
) -> Dict[str, Any]:
    handles:    Dict[str, "_h5py.Dataset"] = {}
    hdf5_files: List["_h5py.File"]        = []
    shapes:     Dict[str, Tuple[int, ...]] = {}

    for name, path in field_paths.items():
        _hdf5_dissect(path)
        f   = _h5py.File(path, "r")
        key = next(
            (k for k in ("data", "field") if k in f and isinstance(f[k], _h5py.Dataset)),
            next(k for k in f if isinstance(f[k], _h5py.Dataset)),
        )
        ds = f[key]
        handles[name]  = ds
        shapes[name]   = tuple(ds.shape)
        hdf5_files.append(f)

    primary_name  = next(iter(field_paths))
    primary_shape = shapes[primary_name]
    total_rows    = primary_shape[0]
    n_fields      = len(field_paths)

    chunk_rows = _compute_chunk_rows(primary_shape, device, n_fields)
    n_chunks   = math.ceil(total_rows / chunk_rows)

    _budget_explain(primary_shape, device, n_fields, chunk_rows, n_chunks)

    if verbose and (not _IS_HPC or _HPC_RANK == 0):
        chunk_gb = _tensor_gb((chunk_rows,) + primary_shape[1:]) * n_fields
        print(f"  Direct-HDF5 stream: {total_rows} rows → "
              f"{n_chunks} chunks × {chunk_rows} rows  "
              f"({chunk_gb:.3f} GB/chunk × {n_fields} field(s))")
        if device.type == "cuda":
            print(f"  {_vram_bar(device)}")

    # Probe: compile graph + discover output shape
    probe_rows = min(8, total_rows)
    probe_fmap: Dict[str, torch.Tensor] = dict(scalar_map)
    for name, ds in handles.items():
        arr = ds[:probe_rows]
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        probe_fmap[name] = torch.from_numpy(arr.astype(np.float64, copy=False)).to(device)

    _nerd(f"probe: {probe_rows} rows, compiling expression …")

    t_compile_start = time.perf_counter()
    pre_graph, pre_sink, warns, field_node_ids = compile_expression(
        expr_str, dx=dx, boundary=boundary, field_map=probe_fmap
    )
    t_compile_end = time.perf_counter()
    t_compile     = t_compile_end - t_compile_start

    with torch.no_grad():
        probe_rt      = stenpy.Runtime(stenpy.MemoryManager(), device=str(device))
        probe_out_map = probe_rt.run(pre_graph)

    probe_out    = probe_out_map.get(pre_sink)
    out_trailing = tuple(probe_out.shape[1:]) if isinstance(probe_out, torch.Tensor) else ()

    _nerd(f"probe output trailing dims: {out_trailing}  field_node_ids: {field_node_ids}")

    del probe_fmap, probe_out_map, probe_out
    del probe_rt
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _nerd(f"post-probe VRAM: {torch.cuda.memory_allocated(device)/1024**3:.3f} GB")

    # Pre-allocate output HDF5
    out_full_shape = (total_rows,) + out_trailing
    out_f  = _h5py.File(out_path, "w")
    out_ds = out_f.create_dataset(
        "data",
        shape            = out_full_shape,
        dtype            = np.float64,
        chunks           = (min(chunk_rows, total_rows),) + out_trailing if out_trailing else None,
        compression      = "gzip",
        compression_opts = 1,
    )
    out_f.attrs["spacing"] = list(spacing) if spacing else [1.0]
    out_f.attrs["origin"]  = list(origin)  if origin  else [0.0]

    t0        = time.perf_counter()
    out_min   = float("inf")
    out_max   = float("-inf")
    out_sum   = 0.0
    out_count = 0
    bytes_in  = 0

    chunk_read_times:    List[float] = []
    chunk_h2d_times:     List[float] = []
    chunk_compute_times: List[float] = []
    chunk_dh_times:      List[float] = []
    chunk_write_times:   List[float] = []

    read_q: Queue = Queue(maxsize=_PIPELINE_DEPTH)

    def _reader() -> None:
        try:
            for ci in range(n_chunks):
                row_start = ci * chunk_rows
                row_end   = min(row_start + chunk_rows, total_rows)
                t_r0 = time.perf_counter()
                chunk_np: Dict[str, np.ndarray] = {}
                for name, ds in handles.items():
                    arr = ds[row_start:row_end]
                    chunk_np[name] = arr.astype(np.float64, copy=False)
                chunk_read_times.append(time.perf_counter() - t_r0)
                read_q.put((ci, row_start, row_end, chunk_np))
            read_q.put(None)
        except Exception as exc:
            read_q.put(exc)

    write_q: Queue = Queue(maxsize=_PIPELINE_DEPTH)

    def _writer() -> None:
        nonlocal out_min, out_max, out_sum, out_count
        while True:
            item = write_q.get()
            if item is None:
                write_q.task_done(); break
            row_start, row_end, out_cpu = item
            t_w0 = time.perf_counter()
            arr = out_cpu.numpy() if isinstance(out_cpu, torch.Tensor) else out_cpu
            out_ds[row_start:row_end] = arr
            chunk_write_times.append(time.perf_counter() - t_w0)
            flat       = arr.ravel()
            out_min    = min(out_min,   float(flat.min()))
            out_max    = max(out_max,   float(flat.max()))
            out_sum   += float(flat.sum())
            out_count += flat.size
            del arr, out_cpu
            write_q.task_done()

    reader_t = threading.Thread(target=_reader, daemon=True, name="hdf5-reader")
    writer_t = threading.Thread(target=_writer, daemon=True, name="hdf5-writer")
    reader_t.start()
    writer_t.start()

    with _make_bar(n_chunks, desc="Streaming", unit="chunk", colour="green") as bar:
        while True:
            item = read_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            ci, row_start, row_end, chunk_np = item

            # H2D transfer
            t_h2d_start = time.perf_counter()
            chunk_tensors: Dict[str, torch.Tensor] = {}
            for name, arr in chunk_np.items():
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)
                chunk_tensors[name] = torch.from_numpy(arr).to(device, non_blocking=True)
                bytes_in += arr.nbytes
            del chunk_np
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            chunk_h2d_times.append(time.perf_counter() - t_h2d_start)

            # Inject chunk tensors into graph
            chunk_graph = pre_graph
            for name, tensor in chunk_tensors.items():
                if name in field_node_ids:
                    chunk_graph = chunk_graph.clone_with_replacement(
                        field_node_ids[name], tensor
                    )

            # GPU compute
            t_compute_start = time.perf_counter()
            with torch.no_grad():
                chunk_rt      = stenpy.Runtime(stenpy.MemoryManager(), device=str(device))
                chunk_results = chunk_rt.run(chunk_graph)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            chunk_compute_times.append(time.perf_counter() - t_compute_start)

            out_gpu = chunk_results.get(pre_sink)

            # D2H transfer
            t_dh_start = time.perf_counter()
            out_cpu = (out_gpu.detach().cpu()
                       if isinstance(out_gpu, torch.Tensor)
                       else torch.zeros((row_end - row_start,) + out_trailing,
                                        dtype=torch.float64))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            chunk_dh_times.append(time.perf_counter() - t_dh_start)

            vram_before_free = (torch.cuda.memory_allocated(device)
                                if device.type == "cuda" else 0)
            n_live = len(chunk_rt.mm._live) if hasattr(chunk_rt.mm, "_live") else -1

            del chunk_tensors
            del chunk_results
            del out_gpu
            chunk_graph = None
            del chunk_rt

            if device.type == "cuda":
                torch.cuda.empty_cache()
                vram_after = torch.cuda.memory_allocated(device)
                freed_gb   = (vram_before_free - vram_after) / 1024**3
                vram_pct   = vram_after / torch.cuda.get_device_properties(device).total_memory * 100

                if _NERD:
                    sev = _sev_label(vram_pct)
                    _rank0_print(
                        f"\n  │  chunk {ci+1:>3}/{n_chunks}"
                        f"  freed {freed_gb:.3f} GB"
                        f"  live_refs {n_live}→0"
                        f"  VRAM {vram_after/1024**3:.2f} GB"
                        f"  {vram_pct:.0f}%{sev}"
                    )

                if vram_pct >= _T_CRIT:
                    bar.set_postfix_str(
                        f"  ██ CRIT {vram_pct:.0f}% VRAM — risk of OOM", refresh=True)

            write_q.put((row_start, row_end, out_cpu))

            elapsed = time.perf_counter() - t0
            tput    = (bytes_in / 1024**3) / elapsed if elapsed > 0 else 0.0
            bar.update(1)
            if device.type == "cuda" and not _NERD:
                vram_gb = torch.cuda.memory_allocated(device) / 1024**3
                bar.set_postfix_str(f"{tput:.2f} GB/s  VRAM {vram_gb:.1f}GB", refresh=False)
            else:
                bar.set_postfix_str(f"{tput:.2f} GB/s", refresh=False)

    write_q.put(None)
    reader_t.join()
    writer_t.join()
    write_q.join()
    out_f.close()

    for f in hdf5_files:
        try: f.close()
        except Exception: pass

    elapsed  = time.perf_counter() - t0
    out_gb   = _tensor_gb(out_full_shape)
    out_mean = out_sum / out_count if out_count else 0.0

    peak_vram_gb  = 0.0
    vram_total_gb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_vram_gb  = torch.cuda.max_memory_allocated(device) / 1024**3
        vram_total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return {
        "shape_out":       list(out_full_shape),
        "min":             out_min if out_min != float("inf")  else 0.0,
        "max":             out_max if out_max != float("-inf") else 0.0,
        "mean":            out_mean,
        "elapsed_s":       elapsed,
        "throughput_gbs":  out_gb / elapsed if elapsed > 0 else 0.0,
        "t_compile":       t_compile,
        "t_read":          sum(chunk_read_times),
        "t_h2d":           sum(chunk_h2d_times),
        "t_compute":       sum(chunk_compute_times),
        "t_d2h":           sum(chunk_dh_times),
        "t_write":         sum(chunk_write_times),
        "chunk_read_s":    chunk_read_times,
        "chunk_h2d_s":     chunk_h2d_times,
        "chunk_compute_s": chunk_compute_times,
        "chunk_dh_s":      chunk_dh_times,
        "chunk_write_s":   chunk_write_times,
        "n_chunks":        n_chunks,
        "chunk_rows":      chunk_rows,
        "peak_vram_gb":    peak_vram_gb,
        "vram_total_gb":   vram_total_gb,
        "input_gb":        bytes_in / 1024**3,
        "output_gb":       out_gb,
    }

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
    out_path:        Optional[str] = None
    throughput_gbs:  float         = 0.0
    perf:            Optional[PerfStats] = None

def _print_banner(title: str, width: int = 72) -> None:
    _rank0_print(f"\n{'═'*width}\n  {title}\n{'═'*width}")

def _print_section(title: str, width: int = 72) -> None:
    _rank0_print(f"\n{'─'*width}\n  {title}\n{'─'*width}")

def _load_field(path: str, device: torch.device,
                normalize: bool) -> Tuple[Any, Tuple, Tuple]:
    with _h5py.File(path, "r") as f:
        key    = next((k for k in ("data", "field") if k in f), list(f.keys())[0])
        nbytes = f[key].size * 8
    mode   = "eager" if nbytes < 500 * 1024**2 else "lazy"
    loaded = stenpy.load_tensor(path, device=device, normalize=normalize,
                             return_mode=mode, max_eager_gb=0.5)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        return loaded
    return loaded, (1.0,) * loaded.ndim, (0.0,) * loaded.ndim

class Pipeline:
    def __init__(
        self,
        field_paths: Dict[str, str],
        project:     Optional[str] = None,
        device:      str           = "cpu",
        dx:          float         = 1.0,
        boundary:    str           = "neumann",
        normalize:   bool          = False,
        verbose:     bool          = True,
    ) -> None:
        if not field_paths:
            raise ValueError("At least one field path must be supplied.")

        self.field_paths = field_paths
        self.device      = torch.device(device)
        self.dx          = dx
        self.boundary    = boundary
        self.normalize   = normalize
        self.verbose     = verbose and (not _IS_HPC or _HPC_RANK == 0)

        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        proj_name = project or f"run_{ts}"

        if _IS_HPC:
            self.out_dir = Path(_hpc_scratch()) / "ops_outputs" / proj_name
        else:
            self.out_dir = Path("outputs") / proj_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.fields:   Dict[str, Any]               = {}
        self.spacings: Dict[str, Tuple[float, ...]] = {}
        self.origins:  Dict[str, Tuple[float, ...]] = {}
        self.shapes:   Dict[str, Tuple[int, ...]]   = {}
        self.spacing:  Optional[Tuple[float, ...]]  = None
        self.origin:   Optional[Tuple[float, ...]]  = None
        self.src_stem  = "field"

        for name, path in field_paths.items():
            if self.verbose:
                print(f"  Loading field '{name}' from : {path}")
            _hdf5_dissect(path)
            tensor, spacing, origin = _load_field(path, self.device, normalize)
            shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()
            self.fields[name]   = tensor
            self.spacings[name] = spacing
            self.origins[name]  = origin
            self.shapes[name]   = shape
            if self.spacing is None:
                self.spacing  = spacing
                self.origin   = origin
                self.src_stem = Path(path).stem

        self.tensor = next(iter(self.fields.values()))

        if len(field_paths) > 1:
            merged_dx, merged_origin, merged_end = _merge_domain(
                self.spacings, self.origins, self.shapes
            )
            if self.verbose:
                print(f"\n  ── Multi-field domain merge ──────────────────────")
                print(f"  dx      : {merged_dx:.6g}  (largest across all fields)")
                print(f"  origin  : {merged_origin}")
                print(f"  end     : {merged_end}")
            if self.dx == 1.0:
                self.dx = merged_dx
            self.spacing = tuple(merged_dx for _ in merged_origin)
            self.origin  = merged_origin
        elif self.dx == 1.0 and self.spacing is not None:
            self.dx = float(min(self.spacing))

        if self.verbose:
            pshape = tuple(self.tensor.shape) if hasattr(self.tensor, "shape") else ("?",)
            print(f"\n  Fields loaded       : {list(self.fields.keys())}")
            print(f"  Primary shape       : {pshape}")
            print(f"  Field size          : {_tensor_gb(pshape):.3f} GB  (float64)")
            print(f"  Spacing             : {self.spacing}")
            print(f"  dx  (from HDF5)     : {self.dx:.6g}")
            print(f"  Device              : {self.device}")
            if self.device.type == "cuda":
                print(f"  {_vram_bar(self.device)}")
                p = torch.cuda.get_device_properties(self.device)
                bw = _gpu_peak_bw_gbs(self.device)
                if math.isnan(bw):
                    print(f"  GPU peak mem BW     : N/A (ROCm attributes missing)")
                else:
                    mem_clock = getattr(p, 'memory_clock_rate', None) or getattr(p, 'clock_rate', None)
                    mem_bus   = getattr(p, 'memory_bus_width', None)
                    if mem_clock is not None and mem_bus is not None:
                        print(f"  GPU peak mem BW     : {bw:.0f} GB/s  "
                            f"({mem_clock/1e6:.2f} GHz × {mem_bus}-bit DDR)")
                    else:
                        print(f"  GPU peak mem BW     : {bw:.0f} GB/s  (details missing)")
            print(f"  Output folder       : {self.out_dir}/")
            if _IS_HPC:
                print(f"  HPC mode            : rank {_HPC_RANK}/{_HPC_WORLD}")
            if _NERD:
                print(f"  Nerd mode           : ON  (OPS_NERD=1)")

        self.mm = stenpy.MemoryManager()
        self.rt = stenpy.Runtime(self.mm, device=device)
        self._run_index = 0

    def load_field(self, name: str, path: str) -> None:
        if self.verbose:
            print(f"  Loading field '{name}' from : {path}")
        _hdf5_dissect(path)
        tensor, spacing, origin = _load_field(path, self.device, self.normalize)
        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()
        self.fields[name]      = tensor
        self.spacings[name]    = spacing
        self.origins[name]     = origin
        self.shapes[name]      = shape
        self.field_paths[name] = path

        merged_dx, merged_origin, merged_end = _merge_domain(
            self.spacings, self.origins, self.shapes
        )
        self.dx      = merged_dx
        self.spacing = tuple(merged_dx for _ in merged_origin)
        self.origin  = merged_origin

        if self.verbose:
            print(f"  '{name}' loaded — shape {shape}")
            print(f"  Domain re-merged  dx={merged_dx:.6g}  "
                  f"origin={merged_origin}  end={merged_end}")

    def _resolve_boundary_and_vars(self, expr_str: str) -> Tuple[str, Dict[str, Any]]:
        boundary    = self.boundary
        spatial_ops = {
            "gradient", "gradient_nd", "divergence", "laplacian",
            "curl", "hessian", "mean_curvature", "surface_normals",
            "material_derivative", "spectral_gradient", "spectral_laplacian",
        }
        if (not _IS_HPC and
                any(op in expr_str for op in spatial_ops) and
                boundary == "neumann"):
            print(f"\n  ℹ  Expression uses a spatial operator.")
            print(f"  Current boundary condition: {boundary}")
            try:
                ans = input("  Keep it or change? [Enter = keep, c = change] ❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans.lower() in ("c", "change"):
                boundary = _select_boundary()
                self.boundary = boundary

        field_map = dict(self.fields)
        for name, uv in _USER_VARS.items():
            if not re.search(rf'\b{re.escape(name)}\b', expr_str):
                continue
            if uv.steps <= 1:
                uv.current = uv.start
                field_map[name] = torch.tensor(uv.current, dtype=torch.float64)
                if self.verbose:
                    print(f"  ℹ  '{name}' = {uv.current:.6g}")
            else:
                if _IS_HPC:
                    uv.current = uv.start
                    field_map.pop(name, None)
                    field_map[f"__sweep_{name}"] = uv
                else:
                    print(f"\n  Variable '{name}'  range [{uv.start}, {uv.end}]  {uv.steps} steps")
                    print("    [1]  Use start value only")
                    print("    [2]  Enter a specific value")
                    print("    [3]  Sweep all values (runs once per step)")
                    try:
                        ch = input("  Choice [2] ❯ ").strip() or "2"
                    except (EOFError, KeyboardInterrupt):
                        ch = "1"
                    if ch == "1":
                        uv.current = uv.start
                        field_map[name] = torch.tensor(uv.current, dtype=torch.float64)
                    elif ch == "3":
                        uv.current = uv.start
                        field_map.pop(name, None)
                        field_map[f"__sweep_{name}"] = uv
                    else:
                        try:
                            uv.current = float(
                                input(f"  Value for {name} [{uv.current}] ❯ ").strip()
                                or str(uv.current)
                            )
                        except ValueError:
                            uv.current = uv.start
                        field_map[name] = torch.tensor(uv.current, dtype=torch.float64)
                        print(f"  ℹ  '{name}' = {uv.current:.6g}")

        return boundary, field_map

    def run_expr(self, expr_str: str) -> RunResult:
        self._run_index += 1
        if self.verbose:
            _print_section(f"Expression #{self._run_index}:  {expr_str}")

        boundary, field_map = self._resolve_boundary_and_vars(expr_str)

        sweep_keys = [k for k in field_map if k.startswith("__sweep_")]
        if sweep_keys:
            return self._run_sweep(expr_str, field_map, sweep_keys, boundary)

        short_name = _expr_to_filename(expr_str, self.src_stem)
        out_path   = str(self.out_dir / f"{short_name}.h5")
        primary    = self.tensor
        is_lazy    = isinstance(primary, stenpy.LazyField)

        # LazyField (streaming) path
        if is_lazy:
            lazy_field_names = {
                n for n, v in field_map.items() if isinstance(v, stenpy.LazyField)
            }
            lazy_path_map = {
                n: self.field_paths[n] for n in lazy_field_names
                if n in self.field_paths
            }
            scalar_map_for_chunk = {
                n: v for n, v in field_map.items()
                if n not in lazy_path_map
                and not n.startswith("__")
                and isinstance(v, torch.Tensor)
            }

            try:
                probe_fm = dict(scalar_map_for_chunk)
                for n, p in lazy_path_map.items():
                    ds, _ = _open_hdf5_field(p)
                    probe_fm[n] = torch.from_numpy(ds[:2].astype(np.float64))
                    ds.file.close()
                _, _, warns, _ = compile_expression(
                    expr_str, dx=self.dx, boundary=boundary, field_map=probe_fm
                )
                if self.verbose and warns:
                    for w in warns:
                        print(f"  ℹ  {w}")
            except Exception:
                warns = []

            in_shape = tuple(primary.shape)
            if self.verbose:
                print(f"\n  LazyField  {in_shape}  {_tensor_gb(in_shape):.3f} GB")
                print(f"  → Direct-HDF5 VRAM-safe chunked stream")
                if self.device.type == "cuda":
                    print(f"  {_vram_bar(self.device)}")

            t_total_start = time.perf_counter()
            stats = _hdf5_direct_chunked_run(
                expr_str    = expr_str,
                field_paths = lazy_path_map,
                scalar_map  = scalar_map_for_chunk,
                dx          = self.dx,
                boundary    = boundary,
                out_path    = out_path,
                spacing     = self.spacing,
                origin      = self.origin,
                device      = self.device,
                verbose     = self.verbose,
            )
            t_total = time.perf_counter() - t_total_start

            shape_out  = tuple(stats["shape_out"])
            elapsed_ms = stats["elapsed_s"] * 1e3
            tput       = stats["throughput_gbs"]

            if self.verbose:
                elapsed_s = elapsed_ms / 1e3
                print(f"\n  ✓ {elapsed_ms:.0f} ms ({elapsed_s:.1f} s)  │  {tput:.3f} GB/s")
                print(f"  Output shape  : {shape_out}  ({_tensor_gb(shape_out):.3f} GB)")
                print(f"  min / max     : {stats['min']:.6g}  /  {stats['max']:.6g}")
                print(f"  mean          : {stats['mean']:.6g}")
                if self.device.type == "cuda":
                    print(f"  {_vram_bar(self.device)}")
                print(f"  Saved → {out_path}")

            est_flops = _estimate_flops(expr_str, shape_out, in_shape)
            perf = PerfStats(
                expr_str        = expr_str,
                mode            = "lazy",
                input_gb        = stats.get("input_gb", _tensor_gb(in_shape)),
                output_gb       = stats.get("output_gb", _tensor_gb(shape_out)),
                t_compile       = stats.get("t_compile", 0.0),
                t_read          = stats.get("t_read",    0.0),
                t_h2d           = stats.get("t_h2d",     0.0),
                t_compute       = stats.get("t_compute", stats["elapsed_s"]),
                t_d2h           = stats.get("t_d2h",     0.0),
                t_write         = stats.get("t_write",   0.0),
                t_total         = t_total,
                chunk_read_s    = stats.get("chunk_read_s",    []),
                chunk_h2d_s     = stats.get("chunk_h2d_s",     []),
                chunk_compute_s = stats.get("chunk_compute_s", []),
                chunk_dh_s      = stats.get("chunk_dh_s",      []),
                chunk_write_s   = stats.get("chunk_write_s",   []),
                n_chunks        = stats.get("n_chunks",   0),
                chunk_rows      = stats.get("chunk_rows", 0),
                peak_vram_gb    = stats.get("peak_vram_gb",  0.0),
                vram_total_gb   = stats.get("vram_total_gb", 0.0),
                est_flops       = est_flops,
            )

            _flush_vram(self.device, verbose=False)
            return RunResult(
                op_name=expr_str, expr_str=expr_str, output=None,
                elapsed_ms=elapsed_ms, shape_in=in_shape, shape_out=shape_out,
                simplifications=warns, node_count=0,
                out_path=out_path, throughput_gbs=tput,
                perf=perf,
            )

        # Eager (in-VRAM) path
        t_compile_start = time.perf_counter()
        try:
            graph, sink_id, warns, field_node_ids = compile_expression(
                expr_str, dx=self.dx, boundary=boundary,
                field_map={k: v for k, v in field_map.items()
                           if not k.startswith("__")},
            )
        except ValueError as exc:
            print(f"  ✗ Parse/compile error: {exc}"); raise
        t_compile = time.perf_counter() - t_compile_start

        if self.verbose and warns:
            for w in warns:
                print(f"  ℹ  {w}")

        if self.verbose:
            topo = graph.topological_sort()
            print(f"  Graph  ({len(topo)} nodes):")
            for n in topo:
                meta     = stenpy.OP_METADATA.get(n.op_name, {})
                cost     = meta.get("cost", "")
                cost_str = f"  [{cost}]" if cost else ""
                deps     = f"← {n.input_ids}" if n.input_ids else "(source)"
                print(f"    {n.op_name:<22}  {deps}{cost_str}")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        try:
            t_compute_start = time.perf_counter()
            if self.verbose:
                with _make_bar(1, desc="Computing", unit="step") as bar:
                    with torch.no_grad():
                        results = self.rt.run(graph)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    bar.update(1)
            else:
                with torch.no_grad():
                    results = self.rt.run(graph)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

            t_compute = time.perf_counter() - t_compute_start
            elapsed_ms = (t_compile + t_compute) * 1e3

            output     = results.get(sink_id)
            shape_out  = tuple(output.shape) if isinstance(output, torch.Tensor) else None
            out_gb     = _tensor_gb(shape_out) if shape_out else 0.0
            tput       = out_gb / (t_compute) if t_compute > 1e-9 else 0.0

            t_dh_start = time.perf_counter()
            if isinstance(output, torch.Tensor) and output.ndim >= 1:
                out_cpu_save = output.detach().cpu()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            t_d2h = time.perf_counter() - t_dh_start

            peak_vram_gb  = 0.0
            vram_total_gb = 0.0
            if self.device.type == "cuda" and torch.cuda.is_available():
                peak_vram_gb  = torch.cuda.max_memory_allocated(self.device) / 1024**3
                vram_total_gb = (torch.cuda.get_device_properties(self.device)
                                 .total_memory / 1024**3)

            if self.verbose:
                print(f"  ✓ {elapsed_ms:.0f} ms  │  {tput:.3f} GB/s")
                if shape_out:
                    print(f"  Output shape  : {shape_out}  ({out_gb:.3f} GB)")
                if isinstance(output, torch.Tensor):
                    flat = output.flatten()
                    print(f"  min / max     : {flat.min().item():.6g}  /  "
                          f"{flat.max().item():.6g}")
                    print(f"  mean          : {flat.mean().item():.6g}")
                    if torch.isnan(output).any():
                        print("  ██ CRIT  NaN detected in output!")
                    if torch.isinf(output).any():
                        print("  ▲▲ WARN  Inf detected in output!")
                if self.device.type == "cuda":
                    print(f"  {_vram_bar(self.device)}")

            in_shape = tuple(self.tensor.shape) if hasattr(self.tensor, "shape") else ()
            in_gb    = _tensor_gb(in_shape)

            t_write_start = time.perf_counter()
            if isinstance(output, torch.Tensor) and output.ndim >= 1:
                stenpy.save_tensor(out_cpu_save, self.spacing, self.origin, out_path)
                if self.verbose:
                    print(f"  Saved → {out_path}")
            else:
                out_path   = out_path.replace(".h5", ".json")
                scalar_val = output.item() if isinstance(output, torch.Tensor) else float(output)
                with open(out_path, "w") as jf:
                    json.dump({"expression": expr_str, "result": scalar_val,
                               "elapsed_ms": elapsed_ms}, jf, indent=2)
                if self.verbose:
                    print(f"  Scalar: {scalar_val:.6g}  → {out_path}")
            t_write = time.perf_counter() - t_write_start

            est_flops = _estimate_flops(expr_str, shape_out, in_shape)
            perf = PerfStats(
                expr_str      = expr_str,
                mode          = "eager",
                input_gb      = in_gb,
                output_gb     = out_gb,
                t_compile     = t_compile,
                t_read        = 0.0,
                t_h2d         = 0.0,
                t_compute     = t_compute,
                t_d2h         = t_d2h,
                t_write       = t_write,
                t_total       = t_compile + t_compute + t_d2h + t_write,
                peak_vram_gb  = peak_vram_gb,
                vram_total_gb = vram_total_gb,
                est_flops     = est_flops,
            )

            return RunResult(
                op_name=expr_str, expr_str=expr_str, output=output,
                elapsed_ms=elapsed_ms, shape_in=in_shape, shape_out=shape_out,
                simplifications=warns, node_count=len(graph),
                out_path=out_path, throughput_gbs=tput,
                perf=perf,
            )
        finally:
            self.rt.flush_vram()

    def _run_sweep(
        self,
        expr_str:   str,
        field_map:  Dict[str, Any],
        sweep_keys: List[str],
        boundary:   str,
    ) -> RunResult:
        svars: List[UserVar] = [field_map.pop(k) for k in sweep_keys]
        clean_map = {k: v for k, v in field_map.items() if not k.startswith("__")}
        results_list: List[RunResult] = []
        uv   = svars[0]
        vals = uv.values

        has_lazy = any(isinstance(v, stenpy.LazyField) for v in clean_map.values())

        if self.verbose:
            print(f"\n  Sweeping '{uv.name}' over {len(vals)} values …")

        with _make_bar(len(vals), desc=f"Sweep {uv.name}", unit="step",
                       colour="green") as bar:
            for v in vals:
                uv.current = float(v)
                sweep_map  = {**clean_map,
                              uv.name: torch.tensor(uv.current, dtype=torch.float64)}

                fname   = _expr_to_filename(f"{expr_str}_{uv.name}{v:.4g}",
                                            self.src_stem)
                op_path = str(self.out_dir / f"{fname}.h5")

                try:
                    if has_lazy:
                        lazy_path_map = {
                            n: self.field_paths[n]
                            for n, val in clean_map.items()
                            if isinstance(val, stenpy.LazyField) and n in self.field_paths
                        }
                        scalar_map_for_chunk = {
                            n: val for n, val in sweep_map.items()
                            if n not in lazy_path_map
                            and not n.startswith("__")
                            and isinstance(val, torch.Tensor)
                        }
                        stats = _hdf5_direct_chunked_run(
                            expr_str    = expr_str,
                            field_paths = lazy_path_map,
                            scalar_map  = scalar_map_for_chunk,
                            dx          = self.dx,
                            boundary    = boundary,
                            out_path    = op_path,
                            spacing     = self.spacing,
                            origin      = self.origin,
                            device      = self.device,
                            verbose     = self.verbose,
                        )
                        shape_out = tuple(stats["shape_out"])
                        in_shape  = tuple(next(iter(self.shapes.values())))
                        est_flops = _estimate_flops(expr_str, shape_out, in_shape)
                        perf = PerfStats(
                            expr_str        = expr_str,
                            mode            = "lazy",
                            input_gb        = stats.get("input_gb", 0.0),
                            output_gb       = stats.get("output_gb", _tensor_gb(shape_out)),
                            t_compile       = stats.get("t_compile", 0.0),
                            t_read          = stats.get("t_read",    0.0),
                            t_h2d           = stats.get("t_h2d",     0.0),
                            t_compute       = stats.get("t_compute", stats["elapsed_s"]),
                            t_d2h           = stats.get("t_d2h",     0.0),
                            t_write         = stats.get("t_write",   0.0),
                            t_total         = stats["elapsed_s"],
                            chunk_read_s    = stats.get("chunk_read_s",    []),
                            chunk_h2d_s     = stats.get("chunk_h2d_s",     []),
                            chunk_compute_s = stats.get("chunk_compute_s", []),
                            chunk_dh_s      = stats.get("chunk_dh_s",      []),
                            chunk_write_s   = stats.get("chunk_write_s",   []),
                            n_chunks        = stats.get("n_chunks",    0),
                            chunk_rows      = stats.get("chunk_rows",  0),
                            peak_vram_gb    = stats.get("peak_vram_gb",  0.0),
                            vram_total_gb   = stats.get("vram_total_gb", 0.0),
                            est_flops       = est_flops,
                        )
                        results_list.append(RunResult(
                            op_name=expr_str, expr_str=expr_str, output=None,
                            elapsed_ms=stats["elapsed_s"] * 1e3,
                            shape_in=in_shape, shape_out=shape_out,
                            simplifications=[], node_count=0,
                            out_path=op_path,
                            throughput_gbs=stats["throughput_gbs"],
                            perf=perf,
                        ))
                        if self.verbose:
                            print(f"  {uv.name}={v:.4g}  →  {op_path}"
                                  f"  ({stats['elapsed_s']:.1f}s  "
                                  f"{stats['throughput_gbs']:.3f} GB/s)")
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()

                    else:
                        g, sid, warns, _ = compile_expression(
                            expr_str, dx=self.dx, boundary=boundary,
                            field_map=sweep_map,
                        )
                        with torch.no_grad():
                            res = self.rt.run(g)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        out = res.get(sid)
                        if isinstance(out, torch.Tensor) and out.ndim >= 1:
                            stenpy.save_tensor(out, self.spacing, self.origin, op_path)
                        results_list.append(RunResult(
                            op_name=expr_str, expr_str=expr_str, output=out,
                            elapsed_ms=0, shape_in=(),
                            shape_out=tuple(out.shape) if isinstance(out, torch.Tensor) else None,
                            simplifications=warns, node_count=len(g),
                            out_path=op_path,
                        ))

                except Exception as exc:
                    _rank0_print(f"\n  ✗ {uv.name}={v:.4g}  {exc}")
                finally:
                    if not has_lazy:
                        self.rt.flush_vram()

                bar.update(1)

        if results_list:
            if self.verbose:
                print(f"  Sweep complete — {len(results_list)} outputs in {self.out_dir}/")
            return results_list[-1]
        else:
            _rank0_print(f"  ✗ Sweep produced no outputs — all steps failed.")
            return RunResult(
                op_name=expr_str, expr_str=expr_str, output=None,
                elapsed_ms=0, shape_in=(), shape_out=None,
                simplifications=[], node_count=0,
            )

    def run_batch(self, expressions: List[str]) -> List[RunResult]:
        _print_banner(f"ops.py  Multi-Operator Pipeline  —  {len(expressions)} expr(s)")
        _rank0_print(f"  Fields : {list(self.fields.keys())}  "
                     f"dx={self.dx:.4g}  bc={self.boundary}  device={self.device}")
        if _IS_HPC:
            _rank0_print(f"  HPC    : rank {_HPC_RANK}/{_HPC_WORLD}  "
                         f"scratch={_hpc_scratch()}")
        if self.device.type == "cuda":
            _rank0_print(f"  {_vram_bar(self.device)}")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        results_list: List[RunResult] = []
        total_t0 = time.perf_counter()

        for expr_str in expressions:
            try:
                results_list.append(self.run_expr(expr_str))
            except Exception as exc:
                _rank0_print(f"\n  ✗ FAILED: {expr_str}  →  {exc}")
            if self.device.type == "cuda" and _gpu_free_gb(self.device) < 1.0:
                _rank0_print(f"  ▲▲ WARN  Low VRAM — flushing automatically.")
                _flush_vram(self.device, verbose=self.verbose)

        total_s  = time.perf_counter() - total_t0
        total_ms = total_s * 1e3

        _print_section("Summary")
        col_w = 36
        _rank0_print(f"  {'#':<3}  {'Expression':<{col_w}}  {'ms':>9}  {'GB/s':>6}  Shape")
        _rank0_print(f"  {'─'*3}  {'─'*col_w}  {'─'*9}  {'─'*6}  {'─'*20}")
        for i, r in enumerate(results_list, 1):
            shape = str(r.shape_out) if r.shape_out else "scalar"
            expr  = r.expr_str[:col_w]
            _rank0_print(f"  {i:<3}  {expr:<{col_w}}  "
                         f"{r.elapsed_ms:>9.0f}  {r.throughput_gbs:>6.3f}  {shape}")
        _rank0_print(f"\n  Total  : {total_ms:.0f} ms  ({total_s:.1f} s)")
        _rank0_print(f"  Output : {self.out_dir}/")

        perf_list = [r.perf for r in results_list if r.perf is not None]
        if perf_list and self.verbose:
            _print_perf_report(perf_list, self.device, total_s, verbose=True)

        manifest = {
            "fields":    self.field_paths,
            "dx":        self.dx,
            "boundary":  self.boundary,
            "device":    str(self.device),
            "total_ms":  total_ms,
            "hpc_rank":  _HPC_RANK  if _IS_HPC else None,
            "hpc_world": _HPC_WORLD if _IS_HPC else None,
            "runs": [
                {
                    "index":          i,
                    "expr":           r.expr_str,
                    "elapsed_ms":     r.elapsed_ms,
                    "throughput_gbs": r.throughput_gbs,
                    "shape_in":       list(r.shape_in),
                    "shape_out":      list(r.shape_out) if r.shape_out else None,
                    "nodes":          r.node_count,
                    "simplifications":r.simplifications,
                    "output_file":    r.out_path,
                    "perf": {
                        "mode":          r.perf.mode,
                        "input_gb":      r.perf.input_gb,
                        "output_gb":     r.perf.output_gb,
                        "t_compile_s":   r.perf.t_compile,
                        "t_read_s":      r.perf.t_read,
                        "t_h2d_s":       r.perf.t_h2d,
                        "t_compute_s":   r.perf.t_compute,
                        "t_d2h_s":       r.perf.t_d2h,
                        "t_write_s":     r.perf.t_write,
                        "t_total_s":     r.perf.t_total,
                        "est_flops":     r.perf.est_flops,
                        "tflops":        r.perf.tflops(),
                        "peak_vram_gb":  r.perf.peak_vram_gb,
                        "n_chunks":      r.perf.n_chunks,
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
        try:
            self.mm.clear_pool()
        except Exception:
            pass
        _flush_vram(self.device, verbose=self.verbose)

# ----------------------------------------------------------------------
# Interactive REPL                                                       
# ----------------------------------------------------------------------
  
_OP_GROUPS: Dict[str, List[str]] = {
    "Arithmetic":    ["add", "sub", "mul", "div", "neg", "clamp"],
    "Elementwise":   ["exp", "log", "sqrt", "sin", "tanh"],
    "Reductions":    ["sum", "mean", "norm_l2", "variance", "entropy",
                      "integrate", "cumulative_integral"],
    "Differential":  ["gradient", "gradient_nd", "divergence", "laplacian",
                      "curl", "hessian", "mean_curvature", "surface_normals",
                      "material_derivative"],
    "Spectral":      ["fft", "ifft", "spectral_gradient", "spectral_laplacian"],
    "Tensor/Matrix": ["trace", "determinant", "eigenvalues", "inverse", "deviatoric"],
}

_OP_HINTS: Dict[str, str] = {
    "gradient":           "∂f/∂x  (dim=N for other dims)",
    "gradient_nd":        "∇f  all dims → (*shape, ndim)",
    "divergence":         "∇·F  needs vector field (*shape, ndim)",
    "laplacian":          "∇²f",
    "curl":               "∇×F  3-D vector field (*shape, 3)",
    "spectral_gradient":  "∂f/∂x via FFT  (periodic BC)",
    "spectral_laplacian": "∇²f via FFT  (spectrally accurate)",
    "fft":                "N-D forward FFT  → real part",
    "ifft":               "N-D inverse FFT  → real part",
    "hessian":            "∇∇f  → (*shape, ndim, ndim)",
    "mean_curvature":     "H = ∇·(∇f/|∇f|)",
    "surface_normals":    "unit normals of level set f=0",
    "integrate":          "∫f dV  all dims, Simpson",
    "cumulative_integral":"running ∫ along dim 0",
    "trace":              "tr(M)  last two dims D×D",
    "determinant":        "det(M)",
    "eigenvalues":        "λ = eigvalsh(M)",
    "inverse":            "M⁻¹",
    "deviatoric":         "M − (tr M / D)·I",
    "variance":           "Var(f)",
    "entropy":            "H(f)  treated as probability distribution",
    "norm_l2":            "‖f‖₂",
}

_W = 72

def _show_operators() -> None:
    registered = {k for k in stenpy.OP_REGISTRY if not k.startswith("_")}
    print()
    print("  ╔" + "═" * (_W - 4) + "╗")
    print("  ║" + f"{'  AVAILABLE OPERATORS  ':^{_W - 4}}" + "║")
    print("  ╠" + "═" * (_W - 4) + "╣")
    for group, members in _OP_GROUPS.items():
        members = [m for m in members if m in registered]
        if not members:
            continue
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
        for op in extras:
            print("  ║    {:<{}}║".format(op, _W - 8))
        print("  ║" + " " * (_W - 4) + "║")
    print("  ╠" + "═" * (_W - 4) + "╣")
    print("  ║  {:<{}}║".format(
        "Multi-field:  f+g  |  gradient(f)*exp(g)  |  laplacian(f)-curl(g)", _W - 6))
    print("  ║  {:<{}}║".format(
        "Auto-simplify:  divergence(gradient(f))  →  laplacian(f)", _W - 6))
    if _USER_VARS:
        uv_str = "User vars: " + ", ".join(
            f"{n}={uv.current:.4g}" for n, uv in _USER_VARS.items())
        print("  ║  {:<{}}║".format(uv_str, _W - 6))
    print("  ╚" + "═" * (_W - 4) + "╝")

def _show_fields(pipeline: Pipeline) -> None:
    if not pipeline.fields:
        print("  (no fields loaded yet)"); return
    print(f"\n  {'Name':<8}  {'Shape':<28}  {'GB':>7}  {'Mode':<12}  {'dx':>8}  Path")
    print(f"  {'─'*8}  {'─'*28}  {'─'*7}  {'─'*12}  {'─'*8}  {'─'*24}")
    for name, tensor in pipeline.fields.items():
        shape  = tuple(tensor.shape) if hasattr(tensor, "shape") else ("?",)
        mode   = "lazy/stream" if isinstance(tensor, stenpy.LazyField) else "eager/VRAM"
        path   = pipeline.field_paths.get(name, "<runtime>")
        sp     = pipeline.spacings.get(name, (float("nan"),))
        dx_val = float(sp[0]) if sp else float("nan")
        gb     = _tensor_gb(shape)
        print(f"  {name:<8}  {str(shape):<28}  {gb:>7.3f}  "
              f"{mode:<12}  {dx_val:>8.5g}  {path}")
    if _NERD and pipeline.device.type == "cuda":
        print(f"\n  VRAM  {_vram_bar(pipeline.device)}")

def _hpc_status(pipeline: Pipeline) -> None:
    parts = [f"device={pipeline.device}", f"dx={pipeline.dx:.4g}",
             f"bc={pipeline.boundary}"]
    print("  HPC  " + "  |  ".join(parts))
    if pipeline.device.type == "cuda" and torch.cuda.is_available():
        print(f"  {_vram_bar(pipeline.device)}")
        bw = _gpu_peak_bw_gbs(pipeline.device)
        if not math.isnan(bw):
            print(f"  GPU peak mem BW : {bw:.0f} GB/s")

def _smart_define_field(pipeline: Pipeline) -> None:
    print()
    print("  Enter one of:")
    print("    field_name          (e.g.  g  or  pressure)")
    print("    /path/to/file.h5    (name auto-assigned)")
    print("    name=/path/file.h5  (name and path together)")
    print()
    try:
        raw = input("  ❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if not raw:
        return

    name = path = None
    m = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)$', raw)
    if m:
        name, path = m.group(1).strip(), m.group(2).strip()
    elif os.sep in raw or raw.startswith(".") or raw.endswith((".h5", ".hdf5")):
        path = raw
        used = set(pipeline.fields.keys())
        name = next((c for c in "fghuvwxyzabcdeijklmnopqrst" if c not in used),
                    f"field{len(pipeline.fields)}")
        print(f"  Auto-assigned name: '{name}'")
    elif re.match(r'^[A-Za-z_]\w*$', raw):
        name = raw
        try:
            path = input(f"  Path to HDF5 file for '{name}': ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
    else:
        print(f"  ✗ Cannot parse '{raw}'."); return

    if not path:
        print("  ✗ No path given."); return
    if not Path(path).exists():
        print(f"  ✗ File not found: {path!r}"); return

    try:
        pipeline.load_field(name, path)
        tensor = pipeline.fields[name]
        shape  = tuple(tensor.shape) if hasattr(tensor, "shape") else ("?",)
        mode   = "lazy/streamed" if isinstance(tensor, stenpy.LazyField) else "eager/in-VRAM"
        sp     = pipeline.spacings.get(name, (float("nan"),))
        gb     = _tensor_gb(shape)
        print(f"\n  ✓ Field '{name}' ready")
        print(f"  {'─'*40}")
        print(f"    shape     : {shape}")
        print(f"    size      : {gb:.3f} GB  (float64)")
        print(f"    mode      : {mode}")
        print(f"    spacing   : {sp}")
        print(f"    origin    : {pipeline.origins.get(name)}")
        if pipeline.device.type == "cuda":
            print(f"    {_vram_bar(pipeline.device)}")
        if _NERD and not isinstance(tensor, stenpy.LazyField):
            t = tensor if isinstance(tensor, torch.Tensor) else None
            if t is not None:
                s = t.reshape(-1)[:65536].float()
                print(f"    sampled stats  min={s.min():.4g}  max={s.max():.4g}"
                      f"  mean={s.mean():.4g}  std={s.std():.4g}")
    except Exception as exc:
        print(f"  ✗ Could not load: {exc}")

def _operate(pipeline: Pipeline) -> None:
    if not pipeline.fields:
        print("\n  ✗ No fields loaded — use [1] to define a field first."); return
    print()
    _show_fields(pipeline)
    print()
    _hpc_status(pipeline)
    if _USER_VARS:
        print(); _show_user_vars()
    print()
    print("  Compose freely using field names shown above.")
    if _USER_VARS:
        print(f"  User-defined variables available: {', '.join(_USER_VARS)}")
    print("  Separate multiple expressions with  ;  to run as a batch.")
    print("  Examples:  f+g   gradient(f)   laplacian(f)*2   curl(f)   f**2+g**2")
    print()
    try:
        raw = input("  Expression ❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if not raw:
        return
    exprs = [e.strip() for e in raw.split(";") if e.strip()]
    if len(exprs) > 1:
        pipeline.run_batch(exprs)
    else:
        try:
            result = pipeline.run_expr(exprs[0])
            if result.out_path is not None:
                print(f"\n  ✓ Saved → {result.out_path}")
                print(f"  Throughput : {result.throughput_gbs:.3f} GB/s  │  "
                      f"{result.elapsed_ms:.0f} ms")
            if result.perf is not None and pipeline.verbose:
                _print_perf_report([result.perf], pipeline.device,
                                   result.elapsed_ms / 1e3, verbose=True)
        except Exception as exc:
            print(f"  ✗ Error: {exc}")
    _prompt_vram_flush(pipeline.device)

def _settings(pipeline: Pipeline) -> None:
    print(f"\n  ┌─── HPC Settings ────────────────────────────────────────────")
    print(f"  │  dx        : {pipeline.dx:.6g}")
    print(f"  │  boundary  : {pipeline.boundary}")
    print(f"  │  device    : {pipeline.device}")
    if pipeline.device.type == "cuda":
        print(f"  │  {_vram_bar(pipeline.device)}")
        peak = torch.cuda.max_memory_allocated(pipeline.device) / 1024**3
        print(f"  │  peak VRAM : {peak:.3f} GB  (this session)")
        bw = _gpu_peak_bw_gbs(pipeline.device)
        print(f"  │  peak mem BW: {bw:.0f} GB/s  (theoretical)")
    if _NERD:
        print(f"  │  nerd mode : ON  (OPS_NERD=1)")
        print(f"  │  chunk frac: {_CHUNK_VRAM_FRACTION:.0%}")
    print(f"  └─────────────────────────────────────────────────────────────")
    print()
    print("  [1]  Change dx")
    print("  [2]  Change boundary condition")
    print("  [3]  Flush VRAM  (clear GPU cache + allocated tensors)")
    print("  [4]  Back")
    try:
        c = input("  ❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if c == "1":
        try:
            raw = input(f"  New dx [{pipeline.dx:.6g}]: ").strip()
            if raw:
                pipeline.dx = float(raw)
            print(f"  ✓ dx = {pipeline.dx:.6g}")
        except ValueError:
            print("  ✗ Invalid number.")
    elif c == "2":
        pipeline.boundary = _select_boundary()
        print(f"  ✓ boundary = {pipeline.boundary}")
    elif c == "3":
        _flushed_ops = False
        try:
            pipeline.rt.flush_vram()
            _flushed_ops = True
        except AttributeError:
            pass
        except Exception as exc:
            print(f"  ⚠  rt.flush_vram() raised: {exc}")
        try:
            pipeline.mm.clear_pool()
            _flushed_ops = True
        except AttributeError:
            pass
        except Exception as exc:
            print(f"  ⚠  mm.clear_pool() raised: {exc}")
        if _NERD and _flushed_ops:
            print("  ○  ops Runtime/MemoryManager pools cleared.")
        _flush_vram(pipeline.device, verbose=True)
        if pipeline.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pipeline.device)
            print("  ○  Peak VRAM counter reset.")

def _repl(pipeline: Pipeline) -> None:
    print()
    print("═" * _W)
    print("  ops.py  │  HPC Field Pipeline  │  Interactive Mode")
    if _NERD:
        print("  ○  Nerd mode active  (OPS_NERD=1)  —  extra diagnostics enabled")
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
            summary = "  ".join(
                f"{n}:{tuple(t.shape) if hasattr(t, 'shape') else '?'}"
                for n, t in pipeline.fields.items()
            )
            print(f"\n  Fields ▸  {summary}")
        else:
            print("\n  No fields loaded yet.")

        if _USER_VARS:
            uv_summary = "  ".join(
                f"{n}=[{uv.start:.3g},{uv.end:.3g}]×{uv.steps}"
                for n, uv in _USER_VARS.items()
            )
            print(f"  Vars   ▸  {uv_summary}")

        if pipeline.device.type == "cuda":
            used_gb = torch.cuda.memory_allocated(pipeline.device) / 1024**3
            if used_gb > 0.5:
                pct = used_gb / (torch.cuda.get_device_properties(
                    pipeline.device).total_memory / 1024**3) * 100
                sev = _sev_label(pct).strip()
                print(f"  VRAM   ▸  {used_gb:.2f} GB  [{sev}]")

        print()
        try:
            choice = input("  ❯ ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!"); break

        if not choice:
            continue
        elif choice == "1":
            _smart_define_field(pipeline)
        elif choice == "2":
            _operate(pipeline)
        elif choice == "3":
            _show_fields(pipeline)
        elif choice == "4":
            _settings(pipeline)
        elif choice == "5":
            _show_operators()
        elif choice == "6":
            _define_user_var(pipeline)
        elif choice == "7":
            _show_user_vars()
        elif choice in ("q", "quit", "exit"):
            print("  Bye!"); break
        else:
            print(f"  ✗ Unknown option '{choice}' — enter 1–7 or q.")

# ----------------------------------------------------------------------
# Demo field generator
# ----------------------------------------------------------------------

def _create_demo_field(path: str = "demo_field.h5",
                       shape: Tuple[int, ...] = (64, 64, 64),
                       dx: float = 0.1,
                       seed: Optional[int] = None) -> str:
    _rank0_print(f"  Creating demo field  shape={shape}  dx={dx}")
    if seed is not None:
        torch.manual_seed(seed)
    coords = [torch.linspace(0, (n - 1) * dx, n, dtype=torch.float64) for n in shape]
    grids  = torch.meshgrid(*coords, indexing="ij")
    if len(shape) == 3:
        f = (torch.sin(2 * math.pi * grids[0]) *
             torch.cos(2 * math.pi * grids[1]) *
             torch.exp(-grids[2] ** 2))
    elif len(shape) == 2:
        f = torch.sin(2 * math.pi * grids[0]) * torch.cos(2 * math.pi * grids[1])
    else:
        f = torch.sin(2 * math.pi * grids[0])
    stenpy.save_tensor(f, (dx,) * len(shape), (0.0,) * len(shape), path)
    _rank0_print(f"  Demo saved → {path}")
    return path

# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
                         
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Dual-mode HPC field pipeline — interactive workbench or "
                    "non-interactive batch job (SLURM/PBS/MPI auto-detected)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Interactive (workstation):
              python main.py --input field.h5 --device cuda
              python main.py --field f=f1.h5 --field g=f2.h5 \\
                  --expr "gradient(f) + laplacian(g)"

            HPC batch (SLURM):
              srun -n 4 python main.py --field f=field.h5 \\
                  --expr "curl(f)" --device cuda --boundary periodic

            Nerd mode (verbose diagnostics):
              OPS_NERD=1 python main.py --input field.h5

            Advanced memory manager (spill-to-disk, NUMA):
              OPS_USE_ADV_MM=1 python main.py --input field.h5
        """),
    )
    fg = p.add_argument_group("Field inputs")
    fg.add_argument("--field", "-F", metavar="NAME=PATH",
                    action="append", dest="fields", default=[],
                    help="--field f=file.h5  (repeatable)")
    fg.add_argument("--input", "-i", help="Shorthand: --field f=PATH")
    p.add_argument("--expr",      "-e", help="Expression(s); semicolons = batch")
    p.add_argument("--expr-file", "-f", help="File with one expression per line")
    p.add_argument("--project",   "-p", help="Project/output folder name")
    p.add_argument("--device",    "-d", default=None,
                   help="cpu | cuda | cuda:N  (HPC: auto-assigned per local rank)")
    p.add_argument("--dx",              type=float, default=1.0,
                   help="Grid spacing override (default: read from HDF5)")
    p.add_argument("--boundary",        default="neumann",
                   choices=["neumann", "dirichlet", "periodic", "reflect"])
    p.add_argument("--normalize",       action="store_true")
    p.add_argument("--repl",            action="store_true",
                   help="Force interactive REPL even if --expr supplied")
    p.add_argument("--demo",            action="store_true")
    p.add_argument("--demo-shape",      default="64,64,64")
    p.add_argument("--quiet", "-q",     action="store_true")
    return p

def _parse_field_args(args) -> Dict[str, str]:
    field_paths: Dict[str, str] = {}
    if args.input:
        field_paths["f"] = args.input
    for raw in (args.fields or []):
        if "=" not in raw:
            raise ValueError(f"--field must be NAME=PATH, got: {raw!r}")
        name, path = raw.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name.isidentifier():
            raise ValueError(f"Field name must be a valid identifier: {name!r}")
        field_paths[name] = path
    return field_paths

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if _IS_HPC:
        try:
            if stenpy.dist_init():
                _rank0_print(f"  ✓ dist_init()  rank={_HPC_RANK}/{_HPC_WORLD}")
            else:
                _rank0_print("  ⚠  dist_init() returned False — running single-rank")
        except Exception as exc:
            _rank0_print(f"  ⚠  dist_init() failed: {exc} — continuing single-rank")

        _use_adv = os.environ.get("OPS_USE_ADV_MM", "1").lower() in ("1", "true", "yes")
        if _use_adv:
            try:
                stenpy.use_advanced_mm()
                _rank0_print("  ✓ Advanced MemoryManager active (spill + NUMA)")
            except Exception as exc:
                _rank0_print(f"  ⚠  use_advanced_mm() failed: {exc} — using built-in MM")

    elif os.environ.get("OPS_USE_ADV_MM", "0").lower() in ("1", "true", "yes"):
        try:
            stenpy.use_advanced_mm()
            print("  ✓ Advanced MemoryManager active")
        except Exception as exc:
            print(f"  ⚠  use_advanced_mm() failed: {exc}")

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
        if not _IS_HPC or _HPC_RANK == 0:
            field_paths["f"] = _create_demo_field(demo_path,
                                                   shape=demo_shape, dx=demo_dx)
        if _IS_HPC:
            try:
                import torch.distributed as _dist
                if _dist.is_initialized():
                    _dist.barrier()
            except Exception:
                time.sleep(2.0)
            field_paths["f"] = demo_path
    else:
        try:
            field_paths = _parse_field_args(args)
        except ValueError as exc:
            parser.error(str(exc))

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
                if not line:
                    break
                eq = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)$', line)
                if eq:
                    n, pth = eq.group(1).strip(), eq.group(2).strip()
                elif os.sep in line or line.endswith(".h5"):
                    used = set(field_paths.keys())
                    n    = next((c for c in "fghuvwxyz" if c not in used),
                                f"f{len(field_paths)}")
                    pth  = line
                    print(f"  Auto-name: '{n}'")
                else:
                    print("  ✗ Use  name=/path  or a bare /path."); continue
                if not Path(pth).exists():
                    print(f"  ✗ Not found: {pth!r}"); continue
                field_paths[n] = pth
                print(f"  ✓ '{n}' → {pth}")
        else:
            pth = raw
            while not Path(pth).exists():
                print(f"  ✗ Not found: {pth!r}")
                pth = input("  Path ❯ ").strip()
            field_paths["f"] = pth

    elif not field_paths and _IS_HPC:
        sys.exit(
            f"[rank {_HPC_RANK}] ERROR: HPC mode requires --field or --input. "
            "No interactive prompt available on compute nodes."
        )

    expressions: List[str] = []
    if args.expr:
        expressions = [p.strip() for p in args.expr.split(";") if p.strip()]
    if args.expr_file:
        with open(args.expr_file) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#"):
                    expressions.append(line)

    if _IS_HPC and not expressions:
        sys.exit(
            f"[rank {_HPC_RANK}] ERROR: HPC mode requires --expr or --expr-file. "
            "The interactive REPL cannot run on a compute node."
        )

    pipeline = Pipeline(
        field_paths = field_paths,
        project     = args.project,
        device      = device_str,
        dx          = args.dx,
        boundary    = args.boundary,
        normalize   = args.normalize,
        verbose     = not args.quiet,
    )

    if expressions:
        pipeline.run_batch(expressions)

    if not _IS_HPC and (args.repl or not expressions):
        if sys.stdin.isatty():
            _repl(pipeline)
        else:
            _rank0_print("  ℹ  stdin is not a terminal — skipping REPL.")

    pipeline.cleanup()

if __name__ == "__main__":
    main()
