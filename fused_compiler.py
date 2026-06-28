#fused_compiler.py

from __future__ import annotations

import hashlib
import math
import threading
import tempfile
import importlib.util
import sys
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from stenpy import OP_METADATA, OP_REGISTRY

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from stenpy import Graph, Runtime


# ---------------------------------------------------------------------------
# 1. Cost model helpers
# ---------------------------------------------------------------------------

def compute_flops(group: "FusionGroup") -> int:
    return group.flops

def compute_physical_bytes(group: "FusionGroup") -> int:
    return group.bytes_read + group.bytes_written

def arithmetic_intensity(group: "FusionGroup") -> float:
    flops = compute_flops(group)
    bytes_ = compute_physical_bytes(group)
    return flops / bytes_ if bytes_ > 0 else float('inf')

def estimate_registers(group: "FusionGroup") -> int:
    alpha, beta, gamma = 1.0, 0.5, 5
    loads = len(group.inputs)          
    ops = len(group.nodes)
    return int(alpha * loads + beta * ops + gamma)

def estimate_occupancy(regs: int, max_regs_per_thread: int = 64) -> float:
    if regs <= max_regs_per_thread:
        return 1.0
    return max(0.0, min(1.0, max_regs_per_thread / regs))

def classify_memory_bound(group: "FusionGroup", ridge_point: float = 11.3) -> bool:
    return arithmetic_intensity(group) < ridge_point


# ---------------------------------------------------------------------------
# 2. IR (Intermediate Representation)
# ---------------------------------------------------------------------------

class VirtualRegister:
    __slots__ = ("id", "dtype", "producer")
    def __init__(self, id: int, dtype: torch.dtype = torch.float64):
        self.id = id
        self.dtype = dtype
        self.producer: Optional["Instruction"] = None

class Instruction:
    __slots__ = ("inputs", "outputs")
    def __init__(self, inputs: List[VirtualRegister], outputs: List[VirtualRegister]):
        self.inputs = inputs
        self.outputs = outputs
    def emit(self, codegen: "TritonCodegen") -> str:
        raise NotImplementedError

class Load(Instruction):
    __slots__ = ("tensor_name", "shape", "strides", "idx_expr")
    def __init__(self, tensor_name: str, shape: Tuple[int, ...],
                 strides: Tuple[int, ...], var: VirtualRegister):
        super().__init__([], [var])
        self.tensor_name = tensor_name
        self.shape = shape
        self.strides = strides
        self.idx_expr: Optional[str] = None   

class Store(Instruction):
    __slots__ = ("tensor_name", "shape", "strides", "idx_expr")
    def __init__(self, tensor_name: str, shape: Tuple[int, ...],
                 strides: Tuple[int, ...], value: VirtualRegister):
        super().__init__([value], [])
        self.tensor_name = tensor_name
        self.shape = shape
        self.strides = strides
        self.idx_expr: Optional[str] = None

class BinaryOp(Instruction):
    __slots__ = ("op",)
    def __init__(self, op: str, a: VirtualRegister, b: VirtualRegister, out: VirtualRegister):
        super().__init__([a, b], [out])
        self.op = op

class UnaryOp(Instruction):
    __slots__ = ("op",)
    def __init__(self, op: str, a: VirtualRegister, out: VirtualRegister):
        super().__init__([a], [out])
        self.op = op

class Stencil7pt(Instruction):
    __slots__ = ("tensor_name", "offsets", "coeffs", "shape", "strides",
                 "center_expr", "neighbour_exprs", "dx_sq", "dy_sq", "dz_sq")
    def __init__(self, tensor_name: str, offsets: List[Tuple[int,int,int]],
                 coeffs: List[float], shape: Tuple[int,...], strides: Tuple[int,...],
                 out: VirtualRegister, dx_sq: str, dy_sq: str, dz_sq: str):
        super().__init__([], [out])
        self.tensor_name = tensor_name
        self.offsets = offsets
        self.coeffs = coeffs
        self.shape = shape
        self.strides = strides
        self.center_expr: Optional[str] = None
        self.neighbour_exprs: List[str] = []
        self.dx_sq = dx_sq
        self.dy_sq = dy_sq
        self.dz_sq = dz_sq

class IRProgram:
    def __init__(self):
        self.instructions: List[Instruction] = []
        self.registers: List[VirtualRegister] = []
        self.output_map: Dict[str, VirtualRegister] = {}
        self.constants: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 3. Fusion group and planner
# ---------------------------------------------------------------------------

class FusionGroup:
    def __init__(self):
        self.nodes: List[str] = []
        self.inputs: Dict[str, dict] = {}       
        self.outputs: Dict[str, dict] = {}
        self.common_shape: Tuple[int, ...] = ()
        self.ir_program: Optional[IRProgram] = None
        self.flops: int = 0
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self.estimated_registers: int = 0
        self.estimated_occupancy: float = 1.0
        self.ai: float = 0.0
        self.boundary_mode: str = "periodic"
        self.op_names: Dict[str, str] = {}     

FUSIBLE_OPS = {"add","sub","mul","div","exp","log","sqrt","sin","cos","tanh","neg","laplacian"}

def _broadcast_shapes(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    if s1 == s2:
        return s1
    r1, r2 = list(reversed(s1)), list(reversed(s2))
    common = []
    for d1, d2 in zip(r1, r2):
        if d1 == d2:
            common.append(d1)
        elif d1 == 1:
            common.append(d2)
        elif d2 == 1:
            common.append(d1)
        else:
            return None
    extra = r1[len(r2):] if len(r1) > len(r2) else r2[len(r1):]
    common.extend(extra)
    return tuple(reversed(common))

def can_fuse(pred_id: str, succ_id: str, graph: "Graph",
             groups: Dict[str, FusionGroup]) -> bool:
    pred_shape = graph.get_node_shape(pred_id)
    succ_shape = graph.get_node_shape(succ_id)
    if pred_shape is None or succ_shape is None:
        return False
    if _broadcast_shapes(pred_shape, succ_shape) is None:
        return False

    consumers = graph.build_consumer_map()
    pred_consumers = consumers.get(pred_id, [])
    succ_group = groups.get(succ_id)
    if succ_group is None:
        return False
    for c in pred_consumers:
        if c != succ_id and groups.get(c) != succ_group:
            return False

    if estimate_registers(succ_group) + 1 > 64:
        return False

    for nid in succ_group.nodes + [pred_id]:
        op_name = graph._nodes[nid].op_name
        if not OP_METADATA.get(op_name, {}).get("supports_fp64_fusion", True):
            return False

    stencil_ops = [nid for nid in [pred_id] + succ_group.nodes
                   if graph._nodes[nid].op_name == "laplacian"]
    if stencil_ops:
        modes = {graph._nodes[nid].params.get("boundary", "periodic")
                 for nid in stencil_ops}
        if len(modes) > 1:
            return False

    return True

def merge_groups(g1: FusionGroup, g2: FusionGroup) -> FusionGroup:
    merged = FusionGroup()
    merged.nodes = g1.nodes + g2.nodes
    merged.inputs = {**g1.inputs, **g2.inputs}
    merged.outputs = {**g1.outputs, **g2.outputs}
    for out_key in list(merged.outputs.keys()):
        if out_key in merged.inputs:
            del merged.inputs[out_key]
    if merged.outputs:
        merged.common_shape = next(iter(merged.outputs.values()))["shape"]
    merged.flops = g1.flops + g2.flops
    merged.bytes_read = g1.bytes_read + g2.bytes_read
    merged.bytes_written = g1.bytes_written + g2.bytes_written
    merged.estimated_registers = estimate_registers(merged)
    merged.estimated_occupancy = estimate_occupancy(merged.estimated_registers)
    merged.ai = arithmetic_intensity(merged)
    for g in (g1, g2):
        if g.boundary_mode:
            merged.boundary_mode = g.boundary_mode
    merged.op_names = {**g1.op_names, **g2.op_names}
    return merged

def plan_fusion(graph: "Graph") -> List[FusionGroup]:
    topo = [n.id for n in graph.topological_sort()]
    consumers = graph.build_consumer_map()

    groups: Dict[str, FusionGroup] = {}
    for nid in topo:
        node = graph._nodes[nid]
        if node.op_name not in FUSIBLE_OPS:
            continue
        if node.op_name in ("mul", "add", "sub") and "scalar" in node.params:
            continue
        g = FusionGroup()
        g.nodes = [nid]
        g.op_names[nid] = node.op_name
        for inp in node.input_ids:
            shape = graph.get_node_shape(inp)
            if shape is not None:
                g.inputs[inp] = {"shape": shape, "dtype": torch.float64,
                                 "broadcast": False}
        out_shape = graph.get_node_shape(nid)
        if out_shape is not None:
            g.outputs[nid] = {"shape": out_shape, "dtype": torch.float64}
        meta = OP_METADATA.get(node.op_name, {})
        mult = math.prod(out_shape) if out_shape else 1
        g.flops = meta.get("flops_per_element", 0) * mult
        g.bytes_read = meta.get("bytes_read_per_element", 0) * mult
        g.bytes_written = meta.get("bytes_written_per_element", 0) * mult
        g.estimated_registers = estimate_registers(g)
        g.estimated_occupancy = estimate_occupancy(g.estimated_registers)
        g.ai = arithmetic_intensity(g)
        if node.op_name == "laplacian":
            g.boundary_mode = node.params.get("boundary", "periodic")
        groups[nid] = g
    for nid in reversed(topo):
        node = graph._nodes[nid]
        for pred in node.input_ids:
            if pred not in groups or pred == nid:
                continue
            if can_fuse(pred, nid, graph, groups):
                pred_group = groups.pop(pred)
                nid_group = groups[nid]
                merged = merge_groups(pred_group, nid_group)
                groups[nid] = merged
                for node_id in merged.nodes:
                    groups[node_id] = merged

    # Return unique groups
    unique = list({id(g): g for g in groups.values()}.values())
    return unique


# ---------------------------------------------------------------------------
# 4. Lowering to IR
# ---------------------------------------------------------------------------

def _default_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    strides = [1]
    for s in reversed(shape[1:]):
        strides.append(strides[-1] * s)
    return tuple(reversed(strides))

def lower_graph(graph: "Graph", node_ids: List[str]) -> IRProgram:
    ir = IRProgram()
    reg_counter = 0
    node_to_reg: Dict[str, VirtualRegister] = {}

    def new_reg() -> VirtualRegister:
        nonlocal reg_counter
        r = VirtualRegister(reg_counter, torch.float64)
        reg_counter += 1
        ir.registers.append(r)
        return r

    for nid in node_ids:
        node = graph._nodes[nid]
        input_regs = []
        for inp in node.input_ids:
            if inp in node_to_reg:
                input_regs.append(node_to_reg[inp])
            else:
                shape = graph.get_node_shape(inp) or (1,)
                tensor_name = f"{inp}_ptr"
                strides = _default_strides(shape)
                reg = new_reg()
                load_instr = Load(tensor_name, shape, strides, reg)
                reg.producer = load_instr
                ir.instructions.append(load_instr)
                input_regs.append(reg)
                node_to_reg[inp] = reg

        out_reg = new_reg()
        op_name = node.op_name
        if op_name in ("mul","add","sub") and len(input_regs) < 2:
            continue
        if op_name in ("add","sub","mul","div"):
            ir.instructions.append(BinaryOp(op_name, input_regs[0], input_regs[1], out_reg))
        elif op_name in ("exp","log","sqrt","sin","cos","neg"):
            ir.instructions.append(UnaryOp(op_name, input_regs[0], out_reg))
        elif op_name == "tanh":
            ir.instructions.append(UnaryOp("tanh", input_regs[0], out_reg))
        elif op_name == "laplacian":
            dx_val = node.params.get("dx", 1.0)
            if isinstance(dx_val, (int, float)):
                dx = dy = dz = dx_val
            else:
                dx, dy, dz = dx_val[:3]
            cdx2 = f"dx2_{nid}"
            cdy2 = f"dy2_{nid}"
            cdz2 = f"dz2_{nid}"
            ir.constants[cdx2] = float(dx)**2
            ir.constants[cdy2] = float(dy)**2
            ir.constants[cdz2] = float(dz)**2

            offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
            coeffs = [1.0] * 6
            shape = graph.get_node_shape(nid) or (1,1,1)
            strides = _default_strides(shape)
            first_load = input_regs[0].producer
            tensor_name = first_load.tensor_name if isinstance(first_load, Load) else f"{nid}_input_ptr"
            stencil = Stencil7pt(tensor_name, offsets, coeffs, shape, strides,
                                 out_reg, cdx2, cdy2, cdz2)
            ir.instructions.append(stencil)
        else:
            raise ValueError(f"Unsupported op for fusion: {op_name}")

        out_reg.producer = ir.instructions[-1]
        node_to_reg[nid] = out_reg
        ir.output_map[nid] = out_reg

        out_shape = graph.get_node_shape(nid) or (1,)
        strides = _default_strides(out_shape)
        store = Store(f"{nid}_ptr", out_shape, strides, out_reg)
        ir.instructions.append(store)

    return ir


# ---------------------------------------------------------------------------
# 5. Triton code generator – broadcast‑aware, 1D grid, safe boundaries
# ---------------------------------------------------------------------------

class TritonCodegen:
    def __init__(self, block_size: int = 512, num_warps: int = 4):
        self.block_size = block_size
        self.num_warps = num_warps
        self._has_stencil = False

    def emit(self, group: FusionGroup, ir: IRProgram, boundary_mode: str) -> str:
        self._has_stencil = any(isinstance(instr, Stencil7pt) for instr in ir.instructions)
        shape = group.common_shape
        ndim = len(shape)
        lines = []
        lines.append("@triton.jit")
        lines.append("def fused_kernel(")
        for inp in sorted(group.inputs.keys()):
            lines.append(f"    {inp}_ptr,")
        for out in sorted(group.outputs.keys()):
            lines.append(f"    {out}_ptr,")
        for d in range(ndim):
            lines.append(f"    shape_{d},")
        lines.append("    total_elements,")
        for k in sorted(ir.constants.keys()):
            lines.append(f"    {k},")
        lines.append("    BLOCK_SIZE: tl.constexpr,")
        lines.append("):")

        lines.append("    pid = tl.program_id(0)")
        lines.append("    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        lines.append("    mask = offset < total_elements")
        if ndim == 3:
            lines.append("    nx = shape_0; ny = shape_1; nz = shape_2")
            lines.append("    i = offset // (ny * nz)")
            lines.append("    j = (offset // nz) % ny")
            lines.append("    k = offset % nz")
            lines.append("    idx_centre = i * (ny * nz) + j * nz + k")
        elif ndim == 2:
            lines.append("    nx = shape_0; ny = shape_1")
            lines.append("    i = offset // ny")
            lines.append("    j = offset % ny")
            lines.append("    idx_centre = i * ny + j")
        else:
            lines.append("    idx_centre = offset")

        for instr in ir.instructions:
            if isinstance(instr, Load):
                instr.idx_expr = self._broadcast_index(instr.shape, instr.strides, ndim)

        if self._has_stencil:
            stencil_instr = next(i for i in ir.instructions if isinstance(i, Stencil7pt))
            ptr = stencil_instr.tensor_name
            lines.append(f"    c = tl.load({ptr} + idx_centre, mask=mask, other=0.0)")
            lines.append(f"    r{stencil_instr.outputs[0].id} = 0.0")

            for idx, off in enumerate(stencil_instr.offsets):
                if ndim == 3:
                    if boundary_mode == "periodic":
                        lines.append(f"    n_i{idx} = (i + {off[0]} + nx) % nx")
                        lines.append(f"    n_j{idx} = (j + {off[1]} + ny) % ny")
                        lines.append(f"    n_k{idx} = (k + {off[2]} + nz) % nz")
                    else:
                        lines.append(f"    n_i{idx} = tl.minimum(tl.maximum(i + {off[0]}, 0), nx-1)")
                        lines.append(f"    n_j{idx} = tl.minimum(tl.maximum(j + {off[1]}, 0), ny-1)")
                        lines.append(f"    n_k{idx} = tl.minimum(tl.maximum(k + {off[2]}, 0), nz-1)")
                    lines.append(f"    n_idx{idx} = n_i{idx} * (ny * nz) + n_j{idx} * nz + n_k{idx}")
                elif ndim == 2:
                    if boundary_mode == "periodic":
                        lines.append(f"    n_i{idx} = (i + {off[0]} + nx) % nx")
                        lines.append(f"    n_j{idx} = (j + {off[1]} + ny) % ny")
                    else:
                        lines.append(f"    n_i{idx} = tl.minimum(tl.maximum(i + {off[0]}, 0), nx-1)")
                        lines.append(f"    n_j{idx} = tl.minimum(tl.maximum(j + {off[1]}, 0), ny-1)")
                    lines.append(f"    n_idx{idx} = n_i{idx} * ny + n_j{idx}")
                else:   
                    if boundary_mode == "periodic":
                        lines.append(f"    n_i{idx} = (i + {off[0]} + nx) % nx")
                    else:
                        lines.append(f"    n_i{idx} = tl.minimum(tl.maximum(i + {off[0]}, 0), nx-1)")
                    lines.append(f"    n_idx{idx} = n_i{idx}")

                lines.append(f"    n{idx} = tl.load({ptr} + n_idx{idx}, mask=mask, other=0.0)")

                if idx < 2:
                    scale = stencil_instr.dx_sq
                elif idx < 4:
                    scale = stencil_instr.dy_sq
                else:
                    scale = stencil_instr.dz_sq
                lines.append(f"    r{stencil_instr.outputs[0].id} += {stencil_instr.coeffs[idx]} * n{idx} / {scale}")

            lines.append(f"    r{stencil_instr.outputs[0].id} -= 2.0 * c / {stencil_instr.dx_sq}")
            lines.append(f"    r{stencil_instr.outputs[0].id} -= 2.0 * c / {stencil_instr.dy_sq}")
            lines.append(f"    r{stencil_instr.outputs[0].id} -= 2.0 * c / {stencil_instr.dz_sq}")

        for instr in ir.instructions:
            if isinstance(instr, Stencil7pt):
                continue   
            elif isinstance(instr, Load):
                lines.append(f"    r{instr.outputs[0].id} = tl.load({instr.tensor_name} + {instr.idx_expr}, mask=mask, other=0.0)")
            elif isinstance(instr, Store):
                lines.append(f"    tl.store({instr.tensor_name} + idx_centre, r{instr.inputs[0].id}, mask=mask)")
            elif isinstance(instr, BinaryOp):
                a, b, out = instr.inputs[0].id, instr.inputs[1].id, instr.outputs[0].id
                lines.append(f"    r{out} = r{a} {instr.op} r{b}")
            elif isinstance(instr, UnaryOp):
                a, out = instr.inputs[0].id, instr.outputs[0].id
                if instr.op == "tanh":
                    lines.append(f"    r{out} = (tl.exp(2.0 * r{a}) - 1.0) / (tl.exp(2.0 * r{a}) + 1.0)")
                else:
                    lines.append(f"    r{out} = tl.{instr.op}(r{a})")

        return "\n".join(lines)

    def _broadcast_index(self, shape: Tuple[int, ...], strides: Tuple[int, ...],
                         ndim: int) -> str:
        rank_diff = ndim - len(shape)
        padded_shape = (1,) * rank_diff + shape
        padded_strides = (0,) * rank_diff + strides
        terms = []
        coords = {3: ['i','j','k'], 2: ['i','j'], 1: ['i']}.get(ndim, [])
        for d in range(ndim):
            if padded_shape[d] == 1:
                term = "0"
            else:
                term = f"{coords[d]}*{padded_strides[d]}"
            terms.append(term)
        return " + ".join(terms)


# ---------------------------------------------------------------------------
# 6. Kernel cache
# ---------------------------------------------------------------------------

class KernelCache:
    def __init__(self, max_size: int = 256):
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size = max_size
        self._lock = threading.Lock()

    def structural_hash(self, group: FusionGroup, boundary_mode: str,
                        constants_keys: Tuple[str, ...]) -> str:
        op_names = sorted(set(group.op_names.values()))
        key_parts = op_names + [boundary_mode] + list(constants_keys)
        key_parts.append("float64")
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, kernel: Any):
        with self._lock:
            self.cache[key] = kernel
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)


# ---------------------------------------------------------------------------
# 7. Execution tasks and scheduler
# ---------------------------------------------------------------------------

class ExecutionTask:
    pass

class FusedKernelTask(ExecutionTask):
    def __init__(self, group: FusionGroup, kernel: Any,
                 constants: Dict[str, Any], outputs: List[str]):
        self.group = group
        self.kernel = kernel
        self.constants = constants
        self.outputs = outputs

class NormalNodeTask(ExecutionTask):
    def __init__(self, node_id: str, inputs: List[str]):
        self.node_id = node_id
        self.inputs = inputs


def _compile_kernel_from_source(src: str):
    full_src = "import triton\nimport triton.language as tl\n" + src
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_src)
        tmp_name = f.name
    try:
        spec = importlib.util.spec_from_file_location("_fused_kernel", tmp_name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_fused_kernel"] = mod
        spec.loader.exec_module(mod)
        return mod.fused_kernel
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass


def build_schedule(fusion_groups: List[FusionGroup],
                   graph: "Graph",
                   runtime: "Runtime") -> List[ExecutionTask]:
    cache = KernelCache()

    node_to_fused_task: Dict[str, FusedKernelTask] = {}
    added_fused: set = set()

    for group in fusion_groups:
        ir = lower_graph(graph, group.nodes)
        group.ir_program = ir
        codegen = TritonCodegen(block_size=512)
        src = codegen.emit(group, ir, group.boundary_mode)
        key = cache.structural_hash(group, group.boundary_mode, tuple(ir.constants.keys()))
        kernel = cache.get(key)
        if kernel is None:
            kernel = _compile_kernel_from_source(src)
            cache.put(key, kernel)
        task = FusedKernelTask(group, kernel, ir.constants, sorted(group.outputs.keys()))
        for nid in group.nodes:
            node_to_fused_task[nid] = task

    all_group_nodes: set = set(node_to_fused_task.keys())

    # Build tasks in topological order so inputs are always ready
    tasks: List[ExecutionTask] = []
    for nid in graph._order:
        if nid in all_group_nodes:
            task = node_to_fused_task[nid]
            if id(task) not in added_fused:
                tasks.append(task)
                added_fused.add(id(task))
        else:
            node = graph._nodes[nid]
            tasks.append(NormalNodeTask(nid, list(node.input_ids)))

    return tasks


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------

def compile_and_execute(graph: "Graph",
                        runtime: "Runtime",
                        seed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    fusion_groups = plan_fusion(graph)

    schedule = build_schedule(fusion_groups, graph, runtime)

    results = dict(seed or {})
    mm = runtime.mm
    step = 0

    for task in schedule:
        if isinstance(task, NormalNodeTask):
            if task.node_id in results:
                continue
            node = graph._nodes[task.node_id]
            inputs = []
            for inp in task.inputs:
                if inp in results:
                    inputs.append(results[inp])
                else:
                    inp_node = graph._nodes.get(inp)
                    if inp_node and inp_node.op_name == "_constant":
                        inputs.append(inp_node.params.get("value"))
                    else:
                        raise RuntimeError(
                            f"NormalNodeTask: input '{inp}' for '{task.node_id}' not in results"
                        )
            op_fn = OP_REGISTRY[node.op_name]
            kwargs = dict(node.params)
            import inspect as _inspect
            sig_params = _inspect.signature(op_fn).parameters
            if "alloc" in sig_params:
                kwargs["alloc"] = mm.allocate
            if "mm" in sig_params:
                kwargs["mm"] = mm
            output = op_fn(*inputs, **kwargs)
            results[task.node_id] = output

        elif isinstance(task, FusedKernelTask):
            for inp_name in sorted(task.group.inputs.keys()):
                inp_node = graph._nodes.get(inp_name)
                if inp_node is None:
                    continue
                meta = OP_METADATA.get(inp_node.op_name, {})
                radius = meta.get("stencil_radius", 0)
                if radius > 0 and inp_name in results:
                    dims = meta.get("exchange_dims", [0])
                    results[inp_name] = mm.halo_exchange(results[inp_name], radius, dims=dims)

            outputs = {}
            for out_id in task.outputs:        
                shape = graph.get_node_shape(out_id)
                if shape is None:
                    raise ValueError(f"Shape missing for output {out_id}")
                outputs[out_id] = mm.allocate(shape, runtime.device, key=out_id)

            args = []
            for inp_name in sorted(task.group.inputs.keys()):
                if inp_name not in results:
                    inp_node = graph._nodes.get(inp_name)
                    if inp_node and inp_node.op_name == "_constant":
                        results[inp_name] = inp_node.params.get("value")
                    else:
                        raise RuntimeError(
                            f"FusedKernelTask: input '{inp_name}' not in results"
                        )
                args.append(results[inp_name])
            for out_id in task.outputs:         
                args.append(outputs[out_id])
            shape = task.group.common_shape
            for d in range(len(shape)):
                args.append(shape[d])
            total_elems = math.prod(shape)
            args.append(total_elems)
            for k in sorted(task.constants.keys()):
                args.append(task.constants[k])

            BLOCK_SIZE = 512
            grid = (triton.cdiv(total_elems, BLOCK_SIZE),)
            task.kernel[grid](*args, BLOCK_SIZE=BLOCK_SIZE)

            for out_id, out_tensor in outputs.items():
                results[out_id] = out_tensor

        step += 1
        mm.advance_step()

    return results
