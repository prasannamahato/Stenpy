# Stenpy: GPU-Accelerated Symbolic Math for Volumetric Data


## License & Commercial Use
This project is for **personal and non-commercial use only** under the [PolyForm Noncommercial License 1.0.0](./LICENSE).

**Commercial entities and for-profit companies:** You must obtain a separate commercial license before use. Please contact me at **prasanna@prasannamahato.com.np** to arrange licensing.

## StenPy – Stencil Computing for the Rest of Us
Write math. Run on terabyte datasets. Get performance reports.

StenPy is a GPU‑accelerated field pipeline that turns symbolic expressions into production‑ready HPC workflows. If you can write gradient(f) + laplacian(g), StenPy can execute it on multi‑gigabyte volumetric data — in‑memory or streaming from disk.

## Why StenPy?
Feature	What it does for you
Symbolic expressions	Write f**2 + tanh(f)*exp(f) – no low‑level kernel coding.
Smart streaming	Fields larger than GPU memory? StenPy chunks, streams, and overlaps I/O with compute.
Automatic stencil optimization	divergence(gradient(f)) → laplacian(f). Compiles with CSE.
HPC ready	SLURM, MPI, OPS_USE_ADV_MM=1 (spill‑to‑disk, NUMA).
Performance telemetry	Stage‑by‑stage breakdown (disk, PCIe, compute), chunk statistics, estimated GFLOP/s.
No boilerplate	One command: stenpy --field f=data.h5 --expr "curl(f)" --device cuda


## How it works 
Parse your expression with SymPy → AST.

Compile to an optimized ops.Graph (CSE, simplification).

Execute in one of two modes:

Eager – everything fits in VRAM → pure PyTorch speed.

Lazy – dataset > VRAM → chunked HDF5 streaming with threaded I/O.

Report – bandwidth, compute rate, bottleneck identification.

## Example performance 
On a 1 GB field with f**2 + tanh(f)*exp(f):

GPU compute time: 212 ms (28 GFLOP/s)

Disk read + write: 5.9 s (bottleneck)

Result: A full performance report tells you exactly what to optimize (RAM disk, faster SSD, eager mode).

Benchmarks performed on:  
GPU: RX 9070 XT 16GB (PCIe 4.0 x16)   
CPU: Ryzen 7 7700X  
RAM: 16GB DDR5 6400MHz


## Operators included
Differential – gradient, divergence, laplacian, curl, hessian, mean curvature, surface normals, material derivative.

Spectral – FFT‑based gradient/laplacian, forward/inverse FFT.

Tensor – trace, determinant, eigenvalues, inverse, deviatoric.

Reductions – sum, mean, L2 norm, variance, entropy, integrate, cumulative integral.

Arithmetic – add, sub, mul, div, neg, clamp, exp, log, sqrt, sin, tanh.

## Perfect for

Computational physics / PDEs on structured grids

Large‑scale simulation post‑processing
