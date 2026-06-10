"""
TileLang codegen backend for torch_npu inductor (Ascend NPU).

Generates TileLang @T.prim_func kernels compiled via
``tilelang.compile(..., target='npuir')``.

Activate with::

    TORCHINDUCTOR_NPU_BACKEND=tilelang

Generated kernel structure:

    @T.prim_func
    def <name>_prim_fn(
        in_ptr0: T.Tensor((_xnumel,), 'float32'),
        out_ptr0: T.Tensor((_xnumel,), 'float32'),
    ):
        with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):
            _in_ptr0_local  = T.alloc_shared((_XBLOCK,), 'float32')  # L1/UB
            _out_ptr0_local = T.alloc_shared((_XBLOCK,), 'float32') # fragment
            T.copy(in_ptr0[cid * _XBLOCK], _in_ptr0_local)  # GM -> L1/UB
            T.vadd(_in_ptr0_local, _in_ptr1_local, _out_ptr0_local)  # vector op
            T.copy(_out_ptr0_local, out_ptr0[cid * _XBLOCK])  # fragment -> GM

Why vector ops instead of T.Parallel scalar loops:
- On Ascend NPU, scalar element-wise stores to L1 (shared.dyn / cbuf) are not
  supported by the BiShengHIR pipeline ('hivm.hir.store' only allows DMA copies).
- T.alloc_fragment on NPU also maps to shared.dyn (same L1 memory), so scalar
  stores to fragment buffers also fail.
- The correct approach is to use TileLang's NPU vector intrinsics (T.vadd, T.vexp,
  etc.) which lower to hardware vector instructions (hivm.hir.vadd, etc.).

Op graph tracking:
- TileLangCSE wraps inductor's CSE and records what op produced each temp var.
- TileLangOverrides.* sets _pending_op on the kernel before returning the expr.
- TileLangKernel.load() records which local buffer each CSE var reads from.
- TileLangKernel.store() records which CSE var holds each output's value.
- codegen_kernel() traverses this graph and emits T.v* calls.

Known limitations:
- Only 1-D contiguous pointwise kernels and simple contiguous reductions.
- No tail guard when xnumel % _XBLOCK != 0.
- Ops without a T.v* equivalent raise NotImplementedError (fallback to Triton).
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

import sympy
import torch
from torch.utils._ordered_set import OrderedSet

from torch._inductor import config, ir
from torch._inductor import scheduler as inductor_scheduler
from torch._inductor.codegen.common import (
    BackendFeature,
    CSE,
    CSEVariable,
    IndentedBuffer,
    OpOverrides,
    TensorArg,
)
from torch._inductor.codegen.simd import (
    SIMDKernel,
    SIMDScheduling,
    IterationRangesRoot,
    IterationRangesEntry,
    schedule_log,
)
from torch._inductor.codegen.triton import (
    get_fused_kernel_name,
    get_kernel_metadata,
)
from torch._inductor.utils import Placeholder, sympy_product
from torch._inductor.virtualized import ReductionType, StoreMode, V

from .scheduling import NPUTritonScheduling


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_TORCH_TO_TL_DTYPE: dict[torch.dtype, str] = {
    torch.float16:  "float16",
    torch.bfloat16: "bfloat16",
    torch.float32:  "float32",
    torch.float64:  "float64",
    torch.int8:     "int8",
    torch.int16:    "int16",
    torch.int32:    "int32",
    torch.int64:    "int64",
    torch.uint8:    "uint8",
    torch.bool:     "bool",
}


def tilelang_dtype(dtype: torch.dtype) -> str:
    return _TORCH_TO_TL_DTYPE.get(dtype, "float32")


# ---------------------------------------------------------------------------
# NPU vector op mappings  (op_name → (tilelang_fn, supported_dtypes))
#
# Dtype support sourced from tilelang-mlir-ascend/docs/Tilelang.language/
# Only dtypes reachable through inductor (no uint16 / uint32 / float64 paths).
# ---------------------------------------------------------------------------

_FP      = frozenset({torch.float16, torch.float32})
_FP_INT  = frozenset({torch.float16, torch.float32,
                      torch.int16, torch.int32, torch.int64})

# Binary vector ops: T.vXXX(A, B, C)  where B may be a scalar
_BINARY_VEC_OPS: dict[str, tuple[str, frozenset]] = {
    "add":         ("vadd", _FP),
    "sub":         ("vsub", _FP),
    "mul":         ("vmul", _FP_INT),
    "truediv":     ("vdiv", frozenset({torch.float16, torch.float32, torch.int64})),
    "maximum":     ("vmax", _FP),
    "minimum":     ("vmin", frozenset({torch.float16, torch.float32, torch.bfloat16,
                                       torch.int16, torch.int32, torch.int64})),
    # vpow: int32 only per hardware docs (fp16/fp32 cause MLIR verification failure).
    # All T.v* ops including vadd/vmul/vpow lower to AIV instructions, so there is
    # no AIC/AIV mixing issue; the dtype constraint is the only restriction.
    "pow":         ("vpow", frozenset({torch.int32})),
    "bitwise_and": ("vand", frozenset({torch.int8, torch.int64,
                                       torch.float16, torch.float32, torch.bool})),
    "bitwise_or":  ("vor",  frozenset()),   # uint16 only — not reachable via inductor
    "bitwise_xor": ("vxor", frozenset()),   # same
}

# Unary vector ops: T.vXXX(A, B)
_UNARY_VEC_OPS: dict[str, tuple[str, frozenset]] = {
    "exp":     ("vexp",     _FP),
    "log":     ("vln",      _FP),
    "exp2":    ("vexp2",    _FP),
    "log2":    ("vlog2",    _FP),
    "relu":    ("vrelu",    _FP),
    "sigmoid": ("vsigmoid", _FP),
    "sqrt":    ("vsqrt",    _FP),
    "rsqrt":   ("vrsqrt",   _FP),
    "abs":     ("vabs",     frozenset({torch.float16, torch.float32,
                                       torch.uint8, torch.int32, torch.int64})),
    "cos":     ("vcos",     _FP),
    "sin":     ("vsin",     _FP),
    "erf":     ("verf",     _FP),
    "tanh":    ("vtanh",    _FP),
}

# Union of all dtypes supported by at least one op — used as early gate in load().
_ANY_SUPPORTED_DTYPE: frozenset[torch.dtype] = frozenset().union(
    *[s for _, s in _BINARY_VEC_OPS.values()],
    *[s for _, s in _UNARY_VEC_OPS.values()],
    {torch.float16, torch.float32},   # always include for neg (vmul fallback)
)


# ---------------------------------------------------------------------------
# Expression -> op parser
# ---------------------------------------------------------------------------

def _parse_expr_op(expr: str) -> Optional[tuple[str, list[str]]]:
    """
    Parse a scalar expression string emitted by TileLangOverrides and return
    (op_name, [operand_str, ...]) or None if unrecognised.

    Since inductor's CSE assigns a fresh tmp var to each compound expression,
    both operands of a binary op are always simple identifiers or number
    literals at the time this is called.
    """
    s = expr.strip()

    # --- unary: (-x) ---
    m = re.match(r'^\(-(\w+)\)$', s)
    if m:
        return ("neg", [m.group(1)])

    # --- unary: abs(x) ---
    m = re.match(r'^abs\((\w+)\)$', s)
    if m:
        return ("abs", [m.group(1)])

    # --- unary: T.exp(x), T.sigmoid(x) ---
    m = re.match(r'^T\.(\w+)\((\w+)\)$', s)
    if m:
        fn = m.group(1)
        arg = m.group(2)
        if fn in ("exp", "sigmoid"):
            return (fn, [arg])

    # --- unary: _math.xxx(x) ---
    _math_unary = {
        "log": "log", "log2": "log2", "log1p": "log1p",
        "sqrt": "sqrt", "sin": "sin", "cos": "cos",
        "tan": "tan", "tanh": "tanh",
        "asin": "asin", "acos": "acos", "atan": "atan",
        "erf": "erf", "erfc": "erfc",
        "floor": "floor", "ceil": "ceil", "trunc": "trunc",
    }
    m = re.match(r'^_math\.(\w+)\((\w+)\)$', s)
    if m and m.group(1) in _math_unary:
        return (_math_unary[m.group(1)], [m.group(2)])

    # --- relu: ((x) if (x) > 0.0 else 0.0) ---
    m = re.match(r'^\(\((\w+)\) if \(\1\) > 0\.0 else 0\.0\)$', s)
    if m:
        return ("relu", [m.group(1)])

    # --- binary: _math.pow(a, b) and _math.atan2(a, b) ---
    m = re.match(r'^_math\.pow\((\w+),\s*(\S+)\)$', s)
    if m:
        return ("pow", [m.group(1), m.group(2)])
    m = re.match(r'^_math\.atan2\((\w+),\s*(\w+)\)$', s)
    if m:
        return ("atan2", [m.group(1), m.group(2)])

    # --- binary: (a OP b) ---
    _bin_patterns: list[tuple[str, str]] = [
        (r'^\((\w+) \+ (\S+)\)$',  "add"),
        (r'^\((\w+) - (\S+)\)$',   "sub"),
        (r'^\((\w+) \* (\S+)\)$',  "mul"),
        (r'^\((\w+) / (\S+)\)$',   "truediv"),
        (r'^\((\w+) // (\S+)\)$',  "floordiv"),
        (r'^\((\w+) % (\S+)\)$',   "mod"),
        (r'^\((\w+) & (\S+)\)$',   "bitwise_and"),
        (r'^\((\w+) \| (\S+)\)$',  "bitwise_or"),
        (r'^\((\w+) \^ (\S+)\)$',  "bitwise_xor"),
        (r'^\((\w+) < (\S+)\)$',   "lt"),
        (r'^\((\w+) <= (\S+)\)$',  "le"),
        (r'^\((\w+) > (\S+)\)$',   "gt"),
        (r'^\((\w+) >= (\S+)\)$',  "ge"),
        (r'^\((\w+) == (\S+)\)$',  "eq"),
        (r'^\((\w+) != (\S+)\)$',  "ne"),
    ]
    for pattern, op in _bin_patterns:
        m = re.match(pattern, s)
        if m:
            return (op, [m.group(1), m.group(2)])

    return None


# ---------------------------------------------------------------------------
# CSE variable
# ---------------------------------------------------------------------------

class TileLangCSEVariable(CSEVariable):
    pass


# ---------------------------------------------------------------------------
# Op overrides
# ---------------------------------------------------------------------------

def _set_pending(op: str, operands: list, dtype: Optional[torch.dtype] = None) -> None:
    """Set _pending_op on the current TileLangKernel (called from overrides)."""
    try:
        k = V.kernel
        if isinstance(k, TileLangKernel):
            k._pending_op = (op, operands, dtype)
    except AttributeError:
        pass


class TileLangOverrides(OpOverrides):
    """
    Maps inductor element-wise ops to scalar Python expressions (used for
    the string CSE) and simultaneously records the op type on the kernel
    for NPU vector code emission.
    """

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype=None, use_compute_types=True):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"T.cast({x}, '{tilelang_dtype(dtype)}')"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        return (
            f"T.reinterpret_cast({x}, '{tilelang_dtype(src_dtype)}', "
            f"'{tilelang_dtype(dtype)}')"
        )

    @staticmethod
    def constant(value, dtype: torch.dtype):
        import torch._prims_common as prim
        literal = repr(prim.dtype_to_type(dtype)(value))
        _set_pending("const", [literal])
        return literal

    @staticmethod
    def abs(x):
        _set_pending("abs", [x])
        return f"abs({x})"

    @staticmethod
    def neg(x):
        _set_pending("neg", [x])
        return f"(-{x})"

    @staticmethod
    def exp(x):
        _set_pending("exp", [x])
        return f"T.exp({x})"

    @staticmethod
    def exp2(x):
        _set_pending("exp2", [x])
        return f"_math.pow(2.0, {x})"

    @staticmethod
    def expm1(x):
        # No single NPU intrinsic; will be caught as unsupported during emit
        return f"(T.exp({x}) - 1.0)"

    @staticmethod
    def log(x):
        _set_pending("log", [x])
        return f"_math.log({x})"

    @staticmethod
    def log2(x):
        _set_pending("log2", [x])
        return f"_math.log2({x})"

    @staticmethod
    def log1p(x):
        return f"_math.log1p({x})"

    @staticmethod
    def sqrt(x):
        _set_pending("sqrt", [x])
        return f"_math.sqrt({x})"

    @staticmethod
    def rsqrt(x):
        _set_pending("rsqrt", [x])
        return f"(1.0 / _math.sqrt({x}))"

    @staticmethod
    def sin(x):
        _set_pending("sin", [x])
        return f"_math.sin({x})"

    @staticmethod
    def cos(x):
        _set_pending("cos", [x])
        return f"_math.cos({x})"

    @staticmethod
    def tan(x):
        _set_pending("tan", [x])
        return f"_math.tan({x})"

    @staticmethod
    def tanh(x):
        _set_pending("tanh", [x])
        return f"_math.tanh({x})"

    @staticmethod
    def asin(x):
        _set_pending("asin", [x])
        return f"_math.asin({x})"

    @staticmethod
    def acos(x):
        _set_pending("acos", [x])
        return f"_math.acos({x})"

    @staticmethod
    def atan(x):
        _set_pending("atan", [x])
        return f"_math.atan({x})"

    @staticmethod
    def atan2(x, y):
        _set_pending("atan2", [x, y])
        return f"_math.atan2({x}, {y})"

    @staticmethod
    def sigmoid(x):
        _set_pending("sigmoid", [x])
        return f"T.sigmoid({x})"

    @staticmethod
    def relu(x):
        _set_pending("relu", [x])
        return f"(({x}) if ({x}) > 0.0 else 0.0)"

    @staticmethod
    def minimum(a, b):
        _set_pending("minimum", [a, b])
        return f"(({a}) if ({a}) < ({b}) else ({b}))"

    @staticmethod
    def maximum(a, b):
        _set_pending("maximum", [a, b])
        return f"(({a}) if ({a}) > ({b}) else ({b}))"

    @staticmethod
    def where(cond, a, b):
        return f"(({a}) if ({cond}) else ({b}))"

    @staticmethod
    def add(a, b):
        _set_pending("add", [a, b])
        return f"({a} + {b})"

    @staticmethod
    def sub(a, b):
        _set_pending("sub", [a, b])
        return f"({a} - {b})"

    @staticmethod
    def mul(a, b):
        _set_pending("mul", [a, b])
        return f"({a} * {b})"

    @staticmethod
    def truediv(a, b):
        _set_pending("truediv", [a, b])
        return f"({a} / {b})"

    @staticmethod
    def floordiv(a, b):
        _set_pending("floordiv", [a, b])
        return f"({a} // {b})"

    @staticmethod
    def mod(a, b):
        _set_pending("mod", [a, b])
        return f"({a} % {b})"

    @staticmethod
    def pow(a, b):
        _set_pending("pow", [a, b])
        return f"_math.pow({a}, {b})"

    @staticmethod
    def logical_not(a):    return f"(not ({a}))"
    @staticmethod
    def logical_and(a, b): return f"(({a}) and ({b}))"
    @staticmethod
    def logical_or(a, b):  return f"(({a}) or ({b}))"
    @staticmethod
    def logical_xor(a, b): return f"(bool({a}) != bool({b}))"

    @staticmethod
    def bitwise_and(a, b):
        _set_pending("bitwise_and", [a, b])
        return f"(({a}) & ({b}))"

    @staticmethod
    def bitwise_or(a, b):
        _set_pending("bitwise_or", [a, b])
        return f"(({a}) | ({b}))"

    @staticmethod
    def bitwise_xor(a, b):
        _set_pending("bitwise_xor", [a, b])
        return f"(({a}) ^ ({b}))"

    @staticmethod
    def bitwise_not(a):    return f"(~({a}))"

    @staticmethod
    def sign(x):  return f"(1 if ({x}) > 0 else (-1 if ({x}) < 0 else 0))"

    @staticmethod
    def floor(x):
        _set_pending("floor", [x])
        return f"_math.floor({x})"

    @staticmethod
    def ceil(x):
        _set_pending("ceil", [x])
        return f"_math.ceil({x})"

    @staticmethod
    def trunc(x):
        _set_pending("trunc", [x])
        return f"_math.trunc({x})"

    @staticmethod
    def erf(x):
        _set_pending("erf", [x])
        return f"_math.erf({x})"

    @staticmethod
    def erfc(x):
        _set_pending("erfc", [x])
        return f"_math.erfc({x})"

    @staticmethod
    def lt(a, b): return f"({a} < {b})"
    @staticmethod
    def le(a, b): return f"({a} <= {b})"
    @staticmethod
    def gt(a, b): return f"({a} > {b})"
    @staticmethod
    def ge(a, b): return f"({a} >= {b})"
    @staticmethod
    def eq(a, b): return f"({a} == {b})"
    @staticmethod
    def ne(a, b): return f"({a} != {b})"

    @staticmethod
    def rand(seed, offset):
        raise NotImplementedError("TileLang backend: rand() not supported")

    @staticmethod
    def randint64(seed, offset, low, high):
        raise NotImplementedError("TileLang backend: randint64() not supported")

    @staticmethod
    def load_seed(name, offset):
        raise NotImplementedError("TileLang backend: load_seed() not supported")

    @staticmethod
    def index_expr(expr, dtype):
        return str(V.kernel.rename_indexing(expr))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

_DEFAULT_XBLOCK = 128


_SUPPORTED_REDUCTIONS = frozenset({"sum", "max", "min"})
_MAX_REDUCTIONS_PER_KERNEL = 2


def _resolve_operand(operand) -> str:
    """Convert a CSEVariable or string operand to a string suitable for T.v* calls."""
    if isinstance(operand, (int, float)):
        return repr(operand)
    return str(operand)


def _is_scalar_literal_operand(operand: Any) -> bool:
    return isinstance(operand, (int, float)) and not isinstance(operand, bool)


def _build_vec_ops(
    var,
    target_buf: str,
    ops_list: list,
    var_bufs: dict,
    var_ops: dict,
    var_consts: Optional[dict] = None,
    _visited: Optional[set] = None,
) -> str:
    """
    Recursively traverse the op graph rooted at `var` and append
    (op_name, resolved_operands, out_buf) tuples to `ops_list` in
    topological (post) order.

    Returns the buffer name that will hold the result of `var`.
    For input-buffer vars this is the existing local buffer name.
    For computed vars this is `target_buf`.
    """
    if _visited is None:
        _visited = set()

    var_name = str(var)

    # Direct input buffer
    if var_name in var_bufs:
        return var_bufs[var_name]

    # Constant literal that went through CSE (e.g. tmp1 = 2.0)
    if var_consts and var_name in var_consts:
        try:
            return float(var_consts[var_name])
        except (ValueError, TypeError):
            return var_consts[var_name]

    # Computed var
    if var_name in var_ops:
        if var_name in _visited:
            return f"_{var_name}_frag"

        _visited.add(var_name)
        op_name, operands = var_ops[var_name]
        resolved = []
        for op in operands:
            op_str = str(op)
            if op_str in var_bufs:
                resolved.append(var_bufs[op_str])
            elif var_consts and op_str in var_consts:
                try:
                    resolved.append(float(var_consts[op_str]))
                except (ValueError, TypeError):
                    resolved.append(var_consts[op_str])
            elif op_str in var_ops:
                inter_buf = f"_{op_str}_frag"
                src = _build_vec_ops(op, inter_buf, ops_list, var_bufs, var_ops, var_consts, _visited)
                resolved.append(src)
            else:
                # Raw literal string (e.g. "2.0" passed directly without going through CSE)
                try:
                    resolved.append(float(op_str))
                except (ValueError, TypeError):
                    resolved.append(op_str)
        ops_list.append((op_name, resolved, target_buf))
        return target_buf

    # Fallback: treat as a literal / unknown symbol
    return var_name


def _has_reduction_index(index: sympy.Expr) -> bool:
    return any(
        getattr(symbol, "name", str(symbol)).startswith("r")
        for symbol in index.free_symbols
    )


def _has_pointwise_index(index: sympy.Expr) -> bool:
    return any(
        getattr(symbol, "name", str(symbol)).startswith("x")
        for symbol in index.free_symbols
    )


def _is_zero_index(index: sympy.Expr) -> bool:
    try:
        return bool(sympy.simplify(index) == 0)
    except Exception:
        return False


class TileLangKernel(SIMDKernel):
    """
    Generates a TileLang @T.prim_func body for a fused set of pointwise nodes.

    Uses NPU vector intrinsics (T.vadd, T.vexp, ...) instead of T.Parallel
    scalar loops - scalar element-wise stores to Ascend L1 (cbuf) are not
    supported by the BiShengHIR pipeline.
    """

    overrides = TileLangOverrides  # type: ignore[assignment]
    kexpr = SIMDKernel.sexpr       # type: ignore[assignment]

    def __init__(self, tiling: dict, **kwargs) -> None:
        super().__init__(tiling, **kwargs)
        self.cse: CSE = CSE(self.newvar_prefix, self.suffix)

        # Buffer registry
        self._tl_inputs:  dict[str, tuple[str, str, torch.dtype]] = {}
        self._tl_outputs: dict[str, tuple[str, str, torch.dtype]] = {}
        self._tl_reduction_input_locs: set[str] = set()
        self._tl_input_indices: dict[str, sympy.Expr] = {}
        self._pointwise_output_symbols: set[sympy.Symbol] = set()
        self._pointwise_output_index: Optional[sympy.Expr] = None

        # Op graph (built during load/store/overrides calls)
        self._pending_op: Optional[tuple] = None
        self._var_ops:    dict[str, tuple] = {}   # CSE var name -> (op, operands)
        self._var_bufs:   dict[str, str] = {}     # CSE var name -> local buffer name
        self._var_consts: dict[str, str] = {}     # CSE var name -> literal string
        self._var_dtypes: dict[str, torch.dtype] = {}
        self._output_vars: dict[str, tuple] = {}  # output local buffer -> (CSE var, dtype)
        self._output_indices: dict[str, sympy.Expr] = {}
        self._reduction_vars: dict[str, tuple[str, TileLangCSEVariable, torch.dtype]] = {}
        self._reduction_outputs: dict[str, tuple[TileLangCSEVariable, torch.dtype]] = {}

    # ------------------------------------------------------------------
    # SIMDKernel abstract interface
    # ------------------------------------------------------------------

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return tilelang_dtype(dtype)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        pass

    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str:
        return "cid"

    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str:
        return f"T.arange(0, {entry.prefix.upper()}BLOCK)"

    def iteration_ranges_scalar_code(self, entry: IterationRangesRoot, value: Any) -> str:
        return repr(value)

    def split_and_set_ranges(self, lengths: Sequence[Sequence[sympy.Expr]]):
        groups = {rt.prefix: rt.numel for rt in self.range_trees}
        if not self.inside_reduction:
            for prefix in groups:
                if prefix.startswith("r"):
                    groups[prefix] = sympy.S.One
        return self.map_kernel_groups_to_node_sizes(
            [*groups.values()], lengths, self.set_ranges
        )

    # ------------------------------------------------------------------
    # load / store / reduction
    # ------------------------------------------------------------------

    def load(self, name: str, index: sympy.Expr) -> TileLangCSEVariable:
        dtype = V.graph.get_dtype(name)
        if dtype not in _ANY_SUPPORTED_DTYPE:
            raise NotImplementedError(
                f"TileLang backend: no T.v* op supports dtype {dtype} "
                f"(buffer '{name}'); falling back to Triton."
            )
        var = self.args.input(name)
        local_name = f"_{var}_local"
        if name not in self._tl_inputs:
            self._tl_inputs[name] = (var, local_name, dtype)
        self._tl_input_indices[local_name] = index
        if self.inside_reduction:
            if _has_reduction_index(index):
                self._tl_reduction_input_locs.add(local_name)
        # Clear any stale _pending_op so load vars are never mis-attributed.
        self._pending_op = None
        cse_var = self.cse.generate(self.loads, f"{local_name}[_tl_i]", dtype=dtype)
        # Record: this CSE var reads from `local_name`
        self._var_bufs[cse_var.name] = local_name
        self._var_dtypes[cse_var.name] = dtype
        return cse_var

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: TileLangCSEVariable,
        mode: StoreMode = None,
    ) -> None:
        if mode == "atomic_add":
            raise NotImplementedError(
                "TileLang backend: atomic_add store not yet supported"
            )
        dtype = V.graph.get_dtype(name)
        self._pointwise_output_symbols.update(index.free_symbols)
        if self._pointwise_output_index is None:
            self._pointwise_output_index = index
        var = self.args.output(name)
        local_name = f"_{var}_local"
        if name not in self._tl_outputs:
            self._tl_outputs[name] = (var, local_name, dtype)
        # Check that every op in the compute graph supports this output dtype.
        # This is the right place to check: all loads/overrides have already run,
        # so _var_ops/_var_consts are fully populated. NotImplementedError here
        # propagates through _body() and triggers inductor's Triton fallback.
        self._check_op_graph_dtype(str(value), dtype)
        # Record: local_name should be filled with the result of `value`
        self._output_vars[local_name] = (value, dtype)
        self._output_indices[local_name] = index
        # Keep scalar emit for debugging / non-NPU paths
        self.stores.writeline(f"{local_name}[_tl_i] = {value}")

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: TileLangCSEVariable,
    ) -> TileLangCSEVariable:
        if not self.inside_reduction:
            raise RuntimeError("assert self.inside_reduction")
        if isinstance(value, tuple):
            raise NotImplementedError(
                "TileLang backend: tuple reductions are not yet supported; "
                "falling back to Triton."
            )
        if reduction_type not in _SUPPORTED_REDUCTIONS:
            raise NotImplementedError(
                f"TileLang backend: reduction '{reduction_type}' is not yet "
                "supported; falling back to Triton."
            )
        if len(self._reduction_outputs) >= _MAX_REDUCTIONS_PER_KERNEL:
            raise NotImplementedError(
                "TileLang backend: too many reductions in one kernel "
                f"(max {_MAX_REDUCTIONS_PER_KERNEL}); falling back to Triton."
            )
        existing_reduction_sequence = [
            reduce_info[0] for reduce_info in self._reduction_vars.values()
        ]
        next_reduction_sequence = [*existing_reduction_sequence, reduction_type]
        if len(next_reduction_sequence) > 1 and next_reduction_sequence not in (
            ["max", "sum"],
            ["sum", "sum"],
        ):
            raise NotImplementedError(
                "TileLang backend: multiple reductions currently support only "
                "max+sum and sum+sum patterns; falling back to Triton."
            )
        self._check_op_graph_dtype(str(value), dtype)

        result = self.cse.generate(
            self.compute,
            f"tl_reduce_{reduction_type}({value})",
            dtype=dtype,
        )
        self._reduction_vars[result.name] = (reduction_type, value, dtype)
        return result

    def store_reduction(
        self,
        name: str,
        index: sympy.Expr,
        value: TileLangCSEVariable,
    ) -> None:
        if not self.inside_reduction:
            raise RuntimeError("assert self.inside_reduction")
        dtype = V.graph.get_dtype(name)
        if str(value) not in self._reduction_vars:
            raise NotImplementedError(
                "TileLang backend: store_reduction expects a direct reduction "
                "result; falling back to Triton."
            )
        local_name = f"_{value}_reduce_local"
        self._var_bufs[str(value)] = local_name
        self._reduction_outputs[local_name] = (value, dtype)
        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            if name not in self._tl_outputs:
                self._tl_outputs[name] = (var, local_name, dtype)
        self.stores.writeline(f"{local_name}[0] = {value}")

    # ------------------------------------------------------------------
    # Source generation
    # ------------------------------------------------------------------

    def _pointwise_input_kind(self, loc: str) -> str:
        index = self._tl_input_indices.get(loc, sympy.S.Zero)
        if _is_zero_index(index):
            return "scalar"
        if self._pointwise_output_symbols:
            load_symbols = set(index.free_symbols)
            if not self._pointwise_output_symbols.issubset(load_symbols):
                return "broadcast"
        return "vector"

    def _pointwise_broadcast_factor(self, loc: str) -> Optional[sympy.Expr]:
        index = self._tl_input_indices.get(loc, sympy.S.Zero)
        if (
            self._pointwise_output_index is None
            or _is_zero_index(index)
            or not isinstance(index, sympy.Symbol)
        ):
            return None

        coeff = sympy.expand(self._pointwise_output_index).coeff(index)
        if coeff in (0, 1):
            return None
        return sympy.simplify(coeff)

    def _pointwise_broadcast_shape(self, loc: str) -> str:
        factor = self._pointwise_broadcast_factor(loc)
        if factor is None:
            return "(T.ceildiv(_xnumel, _XBLOCK),)"
        return f"(T.ceildiv(_xnumel, {factor}),)"

    def _pointwise_broadcast_load_index(self, loc: str) -> str:
        factor = self._pointwise_broadcast_factor(loc)
        if factor is None:
            return "cid"
        return f"((cid * _XBLOCK) // {factor})"

    def _reduction_input_kind(self, loc: str) -> str:
        index = self._tl_input_indices.get(loc, sympy.S.Zero)
        if _is_zero_index(index):
            return "scalar"
        has_x = _has_pointwise_index(index)
        has_r = _has_reduction_index(index)
        if has_x and has_r:
            return "matrix"
        if has_r:
            return "col_vector"
        return "row_broadcast"

    def _is_tensor_output_loc(
        self,
        loc: str,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> bool:
        for _, (var, out_loc, _) in self._tl_outputs.items():
            if out_loc != loc:
                continue
            return tensor_arg_names is None or var in tensor_arg_names
        return False

    def _valid_vector_epilogue_locs(
        self,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> set[str]:
        return {
            loc
            for loc in self._output_vars
            if loc not in self._reduction_outputs
            and self._is_tensor_output_loc(loc, tensor_arg_names)
            and _has_reduction_index(self._output_indices.get(loc, sympy.S.Zero))
        }

    def _has_vector_epilogue_output(
        self,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> bool:
        return bool(self._valid_vector_epilogue_locs(tensor_arg_names))

    def _reduction_output_extent(
        self,
        loc: str,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> str:
        if loc in self._reduction_outputs:
            return "scalar"
        if loc in self._valid_vector_epilogue_locs(tensor_arg_names):
            return "matrix"
        return "scalar"

    def _reduction_output_shape(
        self,
        loc: str,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> str:
        if self._reduction_output_extent(loc, tensor_arg_names) == "matrix":
            return "(1, _RBLOCK)"
        return "(1, 1)"

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        if self._reduction_outputs:
            return self._codegen_reduction_kernel(name)

        xblock       = _DEFAULT_XBLOCK
        prim_fn_name = f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn"

        argdefs, _, signature, _ = self.args.python_argdefs()
        input_shapes = {}
        for _, (var, loc, _) in self._tl_inputs.items():
            kind = self._pointwise_input_kind(loc)
            if kind == "scalar":
                input_shapes[var] = "(1,)"
            elif kind == "broadcast":
                input_shapes[var] = self._pointwise_broadcast_shape(loc)
            else:
                input_shapes[var] = "(_xnumel,)"

        prim_sig_parts: list[str] = []
        for argdef, sig in zip(argdefs, signature):
            if isinstance(sig, TensorArg):
                shape = input_shapes.get(argdef.name, "(_xnumel,)")
                prim_sig_parts.append(
                    f"{argdef.name}: T.Tensor({shape}, '{tilelang_dtype(sig.dtype)}')"
                )

        code = IndentedBuffer()
        code.writeline("import tilelang.language as T")
        code.writeline("import math as _math")
        code.writeline("")
        code.writeline(f"_XBLOCK = {xblock}")
        code.writeline("")
        code.writeline("@T.prim_func")
        code.writeline(f"def {prim_fn_name}(")
        with code.indent():
            for i, part in enumerate(prim_sig_parts):
                code.writeline(f"{part}{',' if i < len(prim_sig_parts) - 1 else ''}")
        code.writeline("):")

        with code.indent():
            code.writeline(
                "with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):"
            )
            with code.indent():
                # ---- allocate input buffers (L1/shared) ----
                for _, (var, loc, dtype) in self._tl_inputs.items():
                    kind = self._pointwise_input_kind(loc)
                    if kind in {"scalar", "broadcast"}:
                        code.writeline(
                            f"{loc}_brc_src = T.alloc_shared((1,), "
                            f"'{tilelang_dtype(dtype)}')"
                        )
                    code.writeline(
                        f"{loc} = T.alloc_shared((_XBLOCK,), '{tilelang_dtype(dtype)}')"
                    )

                # ---- allocate output buffers (fragment) ----
                input_locs = {loc for _, loc, _ in self._tl_inputs.values()}
                for _, (var, loc, dtype) in self._tl_outputs.items():
                    if loc not in input_locs:
                        code.writeline(
                            f"{loc} = T.alloc_shared((_XBLOCK,), '{tilelang_dtype(dtype)}')"
                        )
                code.writeline("")

                # ---- T.copy: GM -> L1 for every input ----
                for _, (var, loc, _) in self._tl_inputs.items():
                    kind = self._pointwise_input_kind(loc)
                    if kind == "scalar":
                        code.writeline(f"T.copy({var}[0], {loc}_brc_src)")
                        code.writeline(f"T.vbrc({loc}_brc_src, {loc})")
                    elif kind == "broadcast":
                        load_index = self._pointwise_broadcast_load_index(loc)
                        code.writeline(f"T.copy({var}[{load_index}], {loc}_brc_src)")
                        code.writeline(f"T.vbrc({loc}_brc_src, {loc})")
                    else:
                        code.writeline(f"T.copy({var}[cid * _XBLOCK], {loc})")
                code.writeline("")

                # ---- emit NPU vector ops ----
                already_allocated = (
                    {loc for _, loc, _ in self._tl_inputs.values()}
                    | {loc for _, loc, _ in self._tl_outputs.values()}
                )
                scalar_cache: dict[tuple[str, str], str] = {}
                for out_loc, (result_var, dtype) in self._output_vars.items():
                    ops_list: list[tuple] = []
                    _build_vec_ops(
                        result_var, out_loc, ops_list,
                        self._var_bufs, self._var_ops, self._var_consts,
                    )

                    if not ops_list:
                        # result_var is a direct input buffer reference (identity)
                        src = self._var_bufs.get(str(result_var), str(result_var))
                        if src != out_loc:
                            code.writeline(f"T.copy({src}, {out_loc})")
                        continue

                    # Fix the last op to write directly into out_loc
                    last_op, last_operands, _ = ops_list[-1]
                    ops_list[-1] = (last_op, last_operands, out_loc)

                    for op_name, operands, out_buf in ops_list:
                        # Allocate intermediate fragment buffers on first use
                        if out_buf not in already_allocated:
                            code.writeline(
                                f"{out_buf} = T.alloc_shared((_XBLOCK,), "
                                f"'{tilelang_dtype(dtype)}')"
                            )
                            already_allocated.add(out_buf)

                        operands = self._materialize_scalar_operands(
                            code, op_name, operands, dtype, scalar_cache
                        )
                        op_str = self._emit_vec_op(op_name, operands, out_buf)
                        code.writeline(op_str)

                code.writeline("")

                # ---- T.copy: fragment -> GM for every output ----
                for _, (var, loc, _) in self._tl_outputs.items():
                    code.writeline(f"T.copy({loc}, {var}[cid * _XBLOCK])")

        src = code.getvalue()
        print("====== TileLang prim_func ======")
        print(src)
        return src

    def _codegen_reduction_kernel(self, name: Optional[str] = None) -> str:
        prim_fn_name = f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn"

        argdefs, _, signature, _ = self.args.python_argdefs()
        tensor_arg_names = {
            argdef.name
            for argdef, sig in zip(argdefs, signature)
            if isinstance(sig, TensorArg)
        }
        input_vars = {var for var, _, _ in self._tl_inputs.values()}
        input_shapes = {}
        for _, (var, loc, _) in self._tl_inputs.items():
            kind = self._reduction_input_kind(loc)
            if kind == "scalar":
                input_shapes[var] = "(1,)"
            elif kind == "row_broadcast":
                input_shapes[var] = "(_xnumel, 1)"
            elif kind == "col_vector":
                input_shapes[var] = "(1, _rnumel)"
            else:
                input_shapes[var] = "(_xnumel, _rnumel)"
        output_vars = {
            var for var, _, _ in self._tl_outputs.values()
            if var in tensor_arg_names
        }
        vector_epilogue = self._has_vector_epilogue_output(tensor_arg_names)
        vector_epilogue_locs = self._valid_vector_epilogue_locs(tensor_arg_names)

        prim_sig_parts: list[str] = []
        for argdef, sig in zip(argdefs, signature):
            if not isinstance(sig, TensorArg):
                continue
            if argdef.name in input_vars:
                shape = input_shapes[argdef.name]
            elif argdef.name in output_vars:
                shape = "(_xnumel, _rnumel)" if vector_epilogue else "(_xnumel, 1)"
            else:
                shape = "(_xnumel, 1)"
            prim_sig_parts.append(
                f"{argdef.name}: T.Tensor({shape}, '{tilelang_dtype(sig.dtype)}')"
            )

        code = IndentedBuffer()
        code.writeline("import tilelang.language as T")
        code.writeline("import math as _math")
        code.writeline("")
        code.writeline("_RBLOCK = _rnumel")
        code.writeline("")
        code.writeline("@T.prim_func")
        code.writeline(f"def {prim_fn_name}(")
        with code.indent():
            for i, part in enumerate(prim_sig_parts):
                code.writeline(f"{part}{',' if i < len(prim_sig_parts) - 1 else ''}")
        code.writeline("):")

        with code.indent():
            code.writeline("with T.Kernel(_xnumel, is_npu=True) as (cid, _):")
            with code.indent():
                for _, (var, loc, dtype) in self._tl_inputs.items():
                    kind = self._reduction_input_kind(loc)
                    if kind in {"scalar", "row_broadcast"}:
                        code.writeline(
                            f"{loc}_brc_src = T.alloc_shared((1, 1), "
                            f"'{tilelang_dtype(dtype)}')"
                        )
                    code.writeline(
                        f"{loc} = T.alloc_shared((1, _RBLOCK), '{tilelang_dtype(dtype)}')"
                    )

                input_locs = {loc for _, loc, _ in self._tl_inputs.values()}
                already_allocated = set(input_locs)
                scalar_cache: dict[tuple[str, str], str] = {}
                for _, (var, loc, dtype) in self._tl_outputs.items():
                    if var not in tensor_arg_names:
                        continue
                    code.writeline(
                        f"{loc} = T.alloc_shared({self._reduction_output_shape(loc, tensor_arg_names)}, "
                        f"'{tilelang_dtype(dtype)}')"
                    )
                    already_allocated.add(loc)
                for loc, (_, dtype) in self._reduction_outputs.items():
                    if loc not in already_allocated:
                        code.writeline(
                            f"{loc} = T.alloc_shared((1, 1), '{tilelang_dtype(dtype)}')"
                        )
                        already_allocated.add(loc)
                code.writeline("")

                for _, (var, loc, _) in self._tl_inputs.items():
                    kind = self._reduction_input_kind(loc)
                    if kind == "scalar":
                        code.writeline(f"T.copy({var}[0], {loc}_brc_src)")
                        code.writeline(f"T.vbrc({loc}_brc_src, {loc})")
                    elif kind == "row_broadcast":
                        code.writeline(f"T.copy({var}[cid, 0], {loc}_brc_src)")
                        code.writeline(f"T.vbrc({loc}_brc_src, {loc})")
                    elif kind == "col_vector":
                        code.writeline(f"T.copy({var}[0, 0], {loc})")
                    else:
                        code.writeline(f"T.copy({var}[cid, 0], {loc})")
                code.writeline("")

                for out_loc, (result_var, dtype) in self._reduction_outputs.items():
                    reduction_type, value, _ = self._reduction_vars[str(result_var)]
                    reduce_input = f"_{result_var}_reduce_in"
                    ops_list: list[tuple] = []
                    scalar_locs = set(self._reduction_outputs)
                    broadcasted_scalars: dict[str, str] = {}

                    def vectorize_operand(operand):
                        if not isinstance(operand, str) or operand not in scalar_locs:
                            return operand
                        if operand not in broadcasted_scalars:
                            brc_buf = f"{operand}_brc"
                            if brc_buf not in already_allocated:
                                code.writeline(
                                    f"{brc_buf} = T.alloc_shared((1, _RBLOCK), "
                                    f"'{tilelang_dtype(dtype)}')"
                                )
                                already_allocated.add(brc_buf)
                            code.writeline(f"T.vbrc({operand}, {brc_buf})")
                            broadcasted_scalars[operand] = brc_buf
                        return broadcasted_scalars[operand]

                    src = _build_vec_ops(
                        value,
                        reduce_input,
                        ops_list,
                        self._var_bufs,
                        self._var_ops,
                        self._var_consts,
                    )

                    if ops_list:
                        if reduce_input not in already_allocated:
                            code.writeline(
                                f"{reduce_input} = T.alloc_shared((1, _RBLOCK), "
                                f"'{tilelang_dtype(dtype)}')"
                            )
                            already_allocated.add(reduce_input)
                        last_op, last_operands, _ = ops_list[-1]
                        ops_list[-1] = (last_op, last_operands, reduce_input)

                    for op_name, operands, op_out_buf in ops_list:
                        if op_out_buf not in already_allocated:
                            code.writeline(
                                f"{op_out_buf} = T.alloc_shared((1, _RBLOCK), "
                                f"'{tilelang_dtype(dtype)}')"
                            )
                            already_allocated.add(op_out_buf)
                        operands = [vectorize_operand(operand) for operand in operands]
                        operands = self._materialize_scalar_operands(
                            code, op_name, operands, dtype, scalar_cache
                        )
                        code.writeline(self._emit_vec_op(op_name, operands, op_out_buf))

                    reduce_src = reduce_input if ops_list else src
                    code.writeline(
                        f"T.reduce({reduce_src}, {out_loc}, dims=[1], "
                        f"reduce_mode='{reduction_type}')"
                    )

                for out_loc, (result_var, dtype) in self._output_vars.items():
                    if out_loc not in vector_epilogue_locs:
                        continue
                    out_kind = self._reduction_output_extent(out_loc, tensor_arg_names)
                    ops_list: list[tuple] = []
                    scalar_locs = set(self._reduction_outputs)
                    broadcasted_scalars: dict[str, str] = {}

                    def vectorize_operand(operand):
                        if (
                            out_kind != "matrix"
                            or not isinstance(operand, str)
                            or operand not in scalar_locs
                        ):
                            return operand
                        if operand not in broadcasted_scalars:
                            brc_buf = f"{operand}_brc"
                            if brc_buf not in already_allocated:
                                code.writeline(
                                    f"{brc_buf} = T.alloc_shared((1, _RBLOCK), "
                                    f"'{tilelang_dtype(dtype)}')"
                                )
                                already_allocated.add(brc_buf)
                            code.writeline(f"T.vbrc({operand}, {brc_buf})")
                            broadcasted_scalars[operand] = brc_buf
                        return broadcasted_scalars[operand]

                    src = _build_vec_ops(
                        result_var,
                        out_loc,
                        ops_list,
                        self._var_bufs,
                        self._var_ops,
                        self._var_consts,
                    )

                    if not ops_list:
                        if src != out_loc:
                            code.writeline(f"T.copy({src}, {out_loc})")
                        continue

                    last_op, last_operands, _ = ops_list[-1]
                    ops_list[-1] = (last_op, last_operands, out_loc)

                    for op_name, operands, op_out_buf in ops_list:
                        if op_out_buf not in already_allocated:
                            op_shape = "(1, _RBLOCK)" if out_kind == "matrix" else "(1, 1)"
                            code.writeline(
                                f"{op_out_buf} = T.alloc_shared({op_shape}, "
                                f"'{tilelang_dtype(dtype)}')"
                            )
                            already_allocated.add(op_out_buf)
                        operands = [vectorize_operand(operand) for operand in operands]
                        operands = self._materialize_scalar_operands(
                            code, op_name, operands, dtype, scalar_cache
                        )
                        code.writeline(self._emit_vec_op(op_name, operands, op_out_buf))

                code.writeline("")
                for _, (var, loc, _) in self._tl_outputs.items():
                    if var not in tensor_arg_names:
                        continue
                    if self._reduction_output_extent(loc, tensor_arg_names) == "matrix":
                        code.writeline(f"T.copy({loc}, {var}[cid, 0])")
                    else:
                        code.writeline(f"T.copy({loc}, {var}[cid, 0])")

        src = code.getvalue()
        print("====== TileLang reduction prim_func ======")
        print(src)
        return src

    def _check_op_graph_dtype(
        self,
        var_name: str,
        dtype: torch.dtype,
        _visited: Optional[set] = None,
    ) -> None:
        """
        Walk the op graph from var_name and raise NotImplementedError if any
        op does not support `dtype`.  Called from store() so it executes
        inside _body(*index_vars) and triggers inductor's Triton fallback.
        """
        if _visited is None:
            _visited = set()
        if var_name in _visited:
            return
        _visited.add(var_name)

        if var_name not in self._var_ops:
            return  # input buffer or constant — no op to check

        op_name, operands = self._var_ops[var_name]

        if op_name in _BINARY_VEC_OPS:
            _, supported = _BINARY_VEC_OPS[op_name]
            if dtype not in supported:
                raise NotImplementedError(
                    f"TileLang NPU: op '{op_name}' (T.{_BINARY_VEC_OPS[op_name][0]}) "
                    f"does not support dtype {dtype}; supported: {supported}. "
                    f"Falling back to Triton."
                )
        elif op_name in _UNARY_VEC_OPS:
            _, supported = _UNARY_VEC_OPS[op_name]
            if dtype not in supported:
                raise NotImplementedError(
                    f"TileLang NPU: op '{op_name}' (T.{_UNARY_VEC_OPS[op_name][0]}) "
                    f"does not support dtype {dtype}; supported: {supported}. "
                    f"Falling back to Triton."
                )
        elif op_name == "neg":
            _, supported = _BINARY_VEC_OPS["mul"]
            if dtype not in supported:
                raise NotImplementedError(
                    f"TileLang NPU: neg (→vmul×-1) does not support dtype {dtype}. "
                    f"Falling back to Triton."
                )

        for op in operands:
            self._check_op_graph_dtype(str(op), dtype, _visited)

    @staticmethod
    def _materialize_scalar_operands(
        code: IndentedBuffer,
        op_name: str,
        operands: list,
        dtype: torch.dtype,
        scalar_cache: dict[tuple[str, str], str],
    ) -> list:
        """Declare scalar literal operands before passing them to T.v* calls."""
        emit_operands = list(operands)
        if op_name == "neg":
            emit_operands = [emit_operands[0], -1.0]

        materialized = []
        tl_dtype = tilelang_dtype(dtype)
        for operand in emit_operands:
            if not _is_scalar_literal_operand(operand):
                materialized.append(operand)
                continue

            literal = repr(operand)
            key = (tl_dtype, literal)
            scalar_name = scalar_cache.get(key)
            if scalar_name is None:
                scalar_name = f"_tl_scalar_{len(scalar_cache)}"
                scalar_cache[key] = scalar_name
                code.writeline(f"{scalar_name} = T.alloc_var('{tl_dtype}')")
                code.writeline(
                    f"T.fill({scalar_name}, T.cast({literal}, '{tl_dtype}'))"
                )
            materialized.append(scalar_name)

        return materialized

    @staticmethod
    def _emit_vec_op(op_name: str, operands: list, out_buf: str) -> str:
        """Return the T.v* call string for one vector operation."""
        if op_name in _BINARY_VEC_OPS:
            vec_fn, _ = _BINARY_VEC_OPS[op_name]
            a = _resolve_operand(operands[0])
            b = _resolve_operand(operands[1])
            return f"T.{vec_fn}({a}, {b}, {out_buf})"

        if op_name in _UNARY_VEC_OPS:
            vec_fn, _ = _UNARY_VEC_OPS[op_name]
            a = _resolve_operand(operands[0])
            return f"T.{vec_fn}({a}, {out_buf})"

        # neg: implement as vmul(x, -1.0, out)
        if op_name == "neg":
            a = _resolve_operand(operands[0])
            b = _resolve_operand(operands[1])
            return f"T.vmul({a}, {b}, {out_buf})"

        raise NotImplementedError(
            f"TileLang NPU backend: op '{op_name}' has no T.v* equivalent. "
            f"This kernel will fall back to Triton."
        )

    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None) -> None:
        wrapper = V.graph.wrapper_code
        _, call_args, signature, _ = self.args.python_argdefs()
        tensor_args = [a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)]
        numel_args  = [str(tree.numel) for tree in self.active_range_trees()]
        wrapper.writeline(f"{name}({', '.join(tensor_args + numel_args)})")

    def create_cse_var(self, name, bounds=None, dtype=None) -> TileLangCSEVariable:
        var = TileLangCSEVariable(name, bounds)
        if dtype is not None:
            self._var_dtypes[name] = dtype
        # Consume any pending op set by TileLangOverrides.*
        # _pending_op is cleared in load() before load-expr generate calls,
        # so only compute-expression vars pick it up here.
        if self._pending_op is not None:
            op_name, operands = self._pending_op[0], self._pending_op[1]
            if op_name == "const":
                self._var_consts[name] = operands[0]  # e.g. "tmp1" -> "2.0"
            else:
                self._var_ops[name] = (op_name, operands)
            self._pending_op = None
        return var

    def should_use_persistent_reduction(self) -> bool:
        return False

    def should_use_cooperative_reduction(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class TileLangScheduling(NPUTritonScheduling):
    """
    Inductor scheduling backend that emits TileLang kernels for Ascend NPU.

    Registered via ``register_backend_for_device`` in __init__.py when
    ``TORCHINDUCTOR_NPU_BACKEND=tilelang``.
    """

    kernel_type: type[Any] = TileLangKernel

    backend_features: OrderedSet[BackendFeature] = OrderedSet()

    def __init__(self, input_scheduler) -> None:
        super().__init__(input_scheduler)
        self.kernel_type = TileLangKernel
        self._triton_scheduling = NPUTritonScheduling(input_scheduler)

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return cls.backend_features

    def codegen_comment(self, node_schedule) -> None:
        wrapper = V.graph.wrapper_code
        origins, _ = get_kernel_metadata(node_schedule, wrapper)
        if origins:
            wrapper.writeline(origins)

    def codegen_sync(self) -> None:
        V.graph.wrapper_code.writeline("torch.npu.synchronize()")

    def _fallback_to_triton(self, node, reason: str):
        schedule_log.debug("TileLang fallback to NPU Triton: %s", reason)
        return self._triton_scheduling.codegen_node(node)

    def _unsupported_tilelang_reason(
        self,
        nodes: list[inductor_scheduler.SchedulerNode],
        reduction_numel: sympy.Expr,
    ) -> Optional[str]:
        reduction_nodes = [node for node in nodes if node.is_reduction()]
        if len(reduction_nodes) > _MAX_REDUCTIONS_PER_KERNEL:
            names = ", ".join(node.get_name() for node in reduction_nodes)
            return (
                "multiple reductions in one TileLang kernel are not yet "
                f"supported ({len(reduction_nodes)} reductions: {names})"
            )

        for node in nodes:
            if node.is_template():
                return f"{node.get_name()} is a template node"
            if node.is_split_scan():
                return f"{node.get_name()} is a split-scan node"

        return None

    @staticmethod
    def _numel_product(value):
        if isinstance(value, (list, tuple)):
            return sympy_product(value)
        return value

    @classmethod
    def _same_numel(cls, lhs, rhs) -> bool:
        lhs = cls._numel_product(lhs)
        rhs = cls._numel_product(rhs)
        try:
            return bool(sympy.simplify(lhs - rhs) == 0)
        except Exception:
            return lhs == rhs

    @staticmethod
    def _reduction_types_for_node(node) -> list[str]:
        reduction_types: list[str] = []
        for scheduler_node in node.get_nodes():
            ir_node = getattr(scheduler_node, "node", None)
            data = getattr(ir_node, "data", None)
            reduction_type = getattr(data, "reduction_type", None)
            if reduction_type is not None:
                reduction_types.append(str(reduction_type))
        return reduction_types

    @classmethod
    def _is_supported_reduction_fusion(cls, node1, node2) -> bool:
        reduction_nodes = sorted(
            [*node1.get_nodes(), *node2.get_nodes()],
            key=lambda node: getattr(node, "min_order", 0),
        )
        reduction_types: list[str] = []
        for scheduler_node in reduction_nodes:
            ir_node = getattr(scheduler_node, "node", None)
            data = getattr(ir_node, "data", None)
            reduction_type = getattr(data, "reduction_type", None)
            if reduction_type is not None:
                reduction_types.append(str(reduction_type))
        return reduction_types in (["max", "sum"], ["sum", "sum"])

    def can_fuse(self, node1, node2):
        if node1.is_reduction() and node2.is_reduction():
            if not self._is_supported_reduction_fusion(node1, node2):
                return False
            return super().can_fuse(node1, node2)

        if node1.is_reduction() != node2.is_reduction():
            reduction_node = node1 if node1.is_reduction() else node2
            pointwise_node = node2 if node1.is_reduction() else node1
            reduction_types = self._reduction_types_for_node(reduction_node)
            if len(reduction_types) > 1 and reduction_types not in (
                ["max", "sum"],
                ["sum", "sum"],
            ):
                return False
            _, (pointwise_numel, pointwise_rnumel) = pointwise_node.group
            _, (reduction_numel, reduction_rnumel) = reduction_node.group
            is_vector_epilogue = self._same_numel(
                pointwise_numel,
                self._numel_product(reduction_numel)
                * self._numel_product(reduction_rnumel),
            )
            if is_vector_epilogue and reduction_types not in (
                ["max", "sum"],
                ["sum", "sum"],
                ["sum"],
            ):
                return False

            if self._same_numel(pointwise_rnumel, 1) and (
                self._same_numel(pointwise_numel, reduction_numel)
                or is_vector_epilogue
            ):
                try:
                    return all(
                        SIMDKernel.is_compatible(
                            (reduction_numel, reduction_rnumel), n.get_ranges()
                        )
                        for n in pointwise_node.get_nodes()
                    )
                except Exception:
                    return True

        return super().can_fuse(node1, node2)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def codegen_node(
        self,
        node: inductor_scheduler.FusedSchedulerNode | inductor_scheduler.SchedulerNode,
    ):
        nodes: list[inductor_scheduler.SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
        _, (_, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group

        reason = self._unsupported_tilelang_reason(nodes, rnumel)
        if reason is not None:
            return self._fallback_to_triton(node, reason)

        try:
            return SIMDScheduling.codegen_node(self, node)
        except NotImplementedError as exc:
            return self._fallback_to_triton(node, str(exc))

    def codegen_node_schedule(self, kernel_features, nodes=None):
        return SIMDScheduling.codegen_node_schedule(self, kernel_features)

    @classmethod
    def select_tiling(cls, nodes, numel, reduction_numel=1):
        if isinstance(numel, (list, tuple)):
            numel = sympy_product(numel)
        if isinstance(reduction_numel, (list, tuple)):
            reduction_numel = sympy_product(reduction_numel)
        if str(reduction_numel) == "1":
            return {"x": numel}
        return {"x": numel, "r0_": reduction_numel}

    def define_kernel(
        self,
        src_code: str,
        node_schedule,
        kernel: TileLangKernel,
        traced_graph_hash: Optional[str] = None,
    ) -> str:
        """
        Splice a shape-keyed caching wrapper into ``wrapper.header``.

        Pattern emitted at module level::

            import tilelang as _tilelang_<N>

            def _prim_factory_<name>(xnumel):
                # @T.prim_func definition (src_code)
                return <name>_prim_fn

            _<name>_cache = {}

            def <name>(in_ptr0, ..., xnumel):
                _key = (int(xnumel),)
                if _key not in _<name>_cache:
                    _<name>_cache[_key] = _tilelang_<N>.compile(
                        _prim_factory_<name>(_key[0]), target='npuir')
                _<name>_cache[_key](in_ptr0, ...)
        """
        wrapper = V.graph.wrapper_code

        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names else ""
        )
        suffix      = wrapper.next_kernel_suffix()
        kernel_name = "_".join(filter(None, ["tilelang", fused_name, suffix]))
        wrapper.src_to_kernel[src_code] = kernel_name

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, call_args, signature, _ = kernel.args.python_argdefs()
        tensor_call_args = [a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)]
        active_trees     = kernel.active_range_trees()
        numel_arg_names  = [f"{t.prefix}numel" for t in active_trees]
        outer_arg_list   = tensor_call_args + numel_arg_names

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        meta_comment = f"{origins}\n{detailed}".strip()

        tl_pkg_root: Optional[str] = None
        try:
            import tilelang as _tl
            import os as _os
            tl_pkg_root = _os.path.dirname(_os.path.dirname(_tl.__file__))
        except ImportError:
            pass

        import_alias = f"_tilelang_{suffix}"
        cache_var    = f"_{kernel_name}_cache"
        factory_fn   = f"_prim_factory_{kernel_name}"
        prim_fn_name = f"{kernel_name}_prim_fn"

        code = IndentedBuffer()
        code.writeline(f"\n# TileLang kernel - {meta_comment}")
        if tl_pkg_root:
            code.writeline("import sys as _sys")
            code.writeline(
                f"if {tl_pkg_root!r} not in _sys.path: "
                f"_sys.path.insert(0, {tl_pkg_root!r})"
            )
        code.writeline(f"import tilelang as {import_alias}")
        code.writeline("")

        factory_params = (
            ", ".join(numel_arg_names) if numel_arg_names else "_dummy=None"
        )
        code.writeline(f"def {factory_fn}({factory_params}):")
        with code.indent():
            if numel_arg_names:
                for arg_name in numel_arg_names:
                    code.writeline(f"_{arg_name} = {arg_name}")
                    if arg_name.startswith("r"):
                        code.writeline(f"_rnumel = {arg_name}")
            else:
                code.writeline("_xnumel = 1")
            code.splice(src_code)
            code.writeline(f"return {prim_fn_name}")

        code.writeline("")
        code.writeline(f"{cache_var} = {{}}")
        code.writeline("")

        code.writeline(f"def {kernel_name}({', '.join(outer_arg_list)}):")
        with code.indent():
            if numel_arg_names:
                code.writeline(
                    f"_key = ({', '.join(f'int({n})' for n in numel_arg_names)},)"
                )
            else:
                code.writeline("_key = ('static',)")
            code.writeline(f"if _key not in {cache_var}:")
            with code.indent():
                factory_call = (
                    ", ".join(f"_key[{i}]" for i in range(len(numel_arg_names)))
                    if numel_arg_names else ""
                )
                code.writeline(f"{cache_var}[_key] = {import_alias}.compile(")
                with code.indent():
                    code.writeline(f"{factory_fn}({factory_call}), target='npuir'")
                code.writeline(")")
            code.writeline(f"{cache_var}[_key]({', '.join(tensor_call_args)})")

        wrapper.header.splice(code.getvalue())
        return kernel_name
