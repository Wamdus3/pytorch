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
- No tail handling when xnumel % _XBLOCK != 0.
- Ops without a T.v* equivalent raise NotImplementedError (fallback to Triton).
"""
from __future__ import annotations

import dataclasses
import os
import re
from typing import Any, Optional, Sequence

import sympy
import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

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
from torch._inductor.ir import ChoiceCaller, TemplateBuffer, TensorBox
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
from .tile_generator import TileGenerator
from .triton import IterationRangesRootNPUIndex, NPUIndexTritonKernel
from .triton_utils import NPUKernelType, get_byte_per_numel


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
_DEFAULT_REDUCTION_XBLOCK = 8
_TILELANG_AUTOTUNE_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE"
_TILELANG_AUTOTUNE_CANDIDATES_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_CANDIDATES"
_TILELANG_AUTOTUNE_WARMUP_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_WARMUP"
_TILELANG_AUTOTUNE_REP_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_REP"
_TILELANG_AUTOTUNE_TIMEOUT_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_TIMEOUT"


def _tilelang_env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _tilelang_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _tilelang_autotune_enabled() -> bool:
    return _tilelang_env_flag(_TILELANG_AUTOTUNE_ENV, True)


def _tilelang_autotune_warmup() -> int:
    return max(0, _tilelang_env_int(_TILELANG_AUTOTUNE_WARMUP_ENV, 5))


def _tilelang_autotune_rep() -> int:
    return max(1, _tilelang_env_int(_TILELANG_AUTOTUNE_REP_ENV, 30))


def _tilelang_autotune_timeout() -> int:
    return max(1, _tilelang_env_int(_TILELANG_AUTOTUNE_TIMEOUT_ENV, 30))


def _tilelang_parse_xblock_candidate_filter() -> Optional[set[int]]:
    raw = os.getenv(_TILELANG_AUTOTUNE_CANDIDATES_ENV)
    if raw is None:
        return None
    candidates: set[int] = set()
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value > 0:
            candidates.add(value)
    return candidates or None


def _tilelang_xblock_configs(
    xnumel: int,
    rnumel: int,
    *,
    is_reduction: bool,
    dtype_bytes: int,
    buffer_count: int,
) -> list[dict[str, int]]:
    """Generate TileLang X0BLOCK configs using Triton-Ascend's TileGenerator."""
    xnumel = max(1, int(xnumel))
    rnumel = max(1, int(rnumel))
    buffer_count = min(max(1, int(buffer_count)), 3)

    dtype = {
        1: torch.int8,
        2: torch.float16,
        4: torch.float32,
        8: torch.int64,
    }.get(max(1, int(dtype_bytes)), torch.float32)
    try:
        tile_generator = TileGenerator(
            [xnumel],
            ["x0"],
            tiling_axis=[0] if not is_reduction else [],
            no_loop_axis=[],
            split_axis=[0],
            low_dims=[],
            persistent_reduction=False,
            dtype=dtype,
            npu_kernel_type=NPUKernelType.SIMD,
            input_ptr_num=buffer_count,
        )
        tile_configs = tile_generator.descend_split_tiling()
    except Exception:
        tile_configs = []

    allowed = _tilelang_parse_xblock_candidate_filter()
    configs: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for cfg in tile_configs:
        cfg_kwargs = getattr(cfg, "kwargs", {})
        x0block = cfg_kwargs.get("X0BLOCK")
        if x0block is None:
            continue
        x0block = int(x0block)
        x0block_sub = int(cfg_kwargs.get("X0BLOCK_SUB", x0block))
        if allowed is not None and x0block not in allowed:
            continue
        key = (x0block, x0block_sub)
        if key in seen:
            continue
        seen.add(key)
        configs.append({"X0BLOCK": x0block, "X0BLOCK_SUB": x0block_sub})

    fallback_hint = _DEFAULT_REDUCTION_XBLOCK if is_reduction else _DEFAULT_XBLOCK
    if allowed is None or fallback_hint in allowed:
        fallback_key = (fallback_hint, fallback_hint)
        if fallback_key not in seen:
            configs.append({"X0BLOCK": fallback_hint, "X0BLOCK_SUB": fallback_hint})
    if not configs:
        configs.append({"X0BLOCK": fallback_hint, "X0BLOCK_SUB": fallback_hint})
    return configs

_TILELANG_NPUIR_PASS_CONFIGS = {
    # Triton-Ascend maps metadata["multibuffer"]=True to
    # --enable-auto-multi-buffer=true. TileLang-Ascend exposes the same
    # BishengIR option through this pass config key.
    "npuir.enable_auto_multi_buffer": True,
}


_SUPPORTED_REDUCTIONS = frozenset({"sum", "max", "min"})


# ---------------------------------------------------------------------------
# Matmul (T.gemm) codegen
# ---------------------------------------------------------------------------

# Accumulation dtype for T.gemm: fp16 → fp32, int8 → int32
_GEMM_ACCUM_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.float32,
    torch.int8:    torch.int32,
}

# Default tile sizes tuned for Ascend NPU cube units
_MM_BLOCK_M = 128
_MM_BLOCK_N = 128
_MM_BLOCK_K = 64


@dataclasses.dataclass
class _TileLangGemmParams:
    """Captures the parameters needed to generate a T.gemm kernel."""
    M: Any
    N: Any
    K: Any
    dtype: torch.dtype
    accum_dtype: torch.dtype
    block_M: int = _MM_BLOCK_M
    block_N: int = _MM_BLOCK_N
    block_K: int = _MM_BLOCK_K


class _TileLangGemmRender:
    """
    Marker stored as ``TemplateBuffer.make_kernel_render``.

    ``TileLangScheduling.codegen_template()`` checks ``isinstance(render,
    _TileLangGemmRender)`` and routes to the TileLang T.gemm path.  The
    base Triton template path never actually calls this object.
    """

    def __init__(self, params: _TileLangGemmParams) -> None:
        self.params = params

    def __call__(self, template_buf: Any) -> None:
        raise NotImplementedError(
            "TileLang T.gemm render is handled by TileLangScheduling.codegen_template(); "
            "direct calls are not supported."
        )


class TileLangGemmCaller(ChoiceCaller):
    """
    ``ChoiceCaller`` that registers a TileLang T.gemm kernel as a candidate
    for ``aten.mm`` / ``aten.addmm`` autotuning.

    Register from ``mm.py``::

        from torch_npu._inductor.codegen.tilelang import add_tilelang_gemm_choices
        if use_tilelang_gemm(layout, m, n, k):
            add_tilelang_gemm_choices(choices, layout, [mat1, mat2])
    """

    def __init__(
        self,
        name: str,
        input_nodes: list,
        layout: Any,
        params: _TileLangGemmParams,
        description: str = "",
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.params = params

    # ------------------------------------------------------------------
    # ChoiceCaller interface
    # ------------------------------------------------------------------

    def hash_key(self) -> str:
        p = self.params
        return (
            f"tilelang_mm_{tilelang_dtype(p.dtype)}_{tilelang_dtype(p.accum_dtype)}"
            f"_bM{p.block_M}_bN{p.block_N}_bK{p.block_K}"
        )

    def output_node(self) -> TensorBox:
        buf = TemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=_TileLangGemmRender(self.params),
        )
        return TensorBox.create(buf)

    def to_callable(self) -> Any:
        raise NotImplementedError("TileLangGemmCaller has no standalone callable")

    def benchmark(self, *args, out=None) -> float:
        # Benchmarking TileLang T.gemm via the inductor autotuner is not yet
        # supported (requires a compiled NPU binary).  Return a large time so
        # that benchmarked ATEN / CATLASS choices win if they are also present.
        # In the common single-choice path this is never called.
        return float("inf")

    def call_name(self) -> str:
        return f"tilelang_mm.{self.name}"

    def get_make_kernel_render(self) -> _TileLangGemmRender:
        return _TileLangGemmRender(self.params)


def add_tilelang_gemm_choices(
    choices: list,
    layout: Any,
    input_nodes: list,
    *,
    block_M: int = _MM_BLOCK_M,
    block_N: int = _MM_BLOCK_N,
    block_K: int = _MM_BLOCK_K,
) -> None:
    """
    Add a ``TileLangGemmCaller`` to *choices* for ``aten.mm`` autotuning.

    Typical usage in ``mm.py``::

        from torch_npu._inductor.codegen.tilelang import add_tilelang_gemm_choices
        if use_tilelang_gemm(layout, m, n, k):
            add_tilelang_gemm_choices(choices, layout, [mat1, mat2])
    """
    mat1, mat2 = input_nodes[0], input_nodes[1]
    dtype = mat1.get_dtype()
    accum_dtype = _GEMM_ACCUM_DTYPE.get(dtype)
    if accum_dtype is None:
        return  # dtype not supported by T.gemm
    M = mat1.get_size()[0]
    K = mat1.get_size()[1]
    N = mat2.get_size()[1]
    params = _TileLangGemmParams(
        M=M, N=N, K=K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    # T.gemm accumulates in accum_dtype (fp32 for fp16 input) and writes fp32
    # to the output buffer.  Override the layout so inductor allocates the
    # correctly-typed buffer instead of the input dtype (fp16), which would
    # cause the fp32 bits to be reinterpreted as fp16 → garbage results.
    #
    # Must use FixedLayout (not FlexibleLayout): when the mm result is read by
    # a fused epilogue (relu/sigmoid/scale), the scheduler calls make_indexer()
    # on the output layout; FlexibleLayout.allow_indexing is False and asserts.
    if layout.dtype != accum_dtype:
        from torch._inductor.ir import FixedLayout
        size = list(layout.size)
        # Compute contiguous row-major strides (stride[-1]=1, stride[i]=stride[i+1]*size[i+1])
        stride = [sympy.Integer(1)] * len(size)
        for i in range(len(size) - 2, -1, -1):
            stride[i] = stride[i + 1] * size[i + 1]
        layout = FixedLayout(
            device=layout.device,
            dtype=accum_dtype,
            size=size,
            stride=stride,
        )

    desc = (
        f"TileLang T.gemm {tilelang_dtype(dtype)}->{tilelang_dtype(accum_dtype)} "
        f"bM{block_M}_bN{block_N}_bK{block_K}"
    )
    choices.append(
        TileLangGemmCaller(
            name="tilelang_mm",
            input_nodes=input_nodes,
            layout=layout,
            params=params,
            description=desc,
        )
    )


def codegen_tilelang_mm_src(
    name: str,
    *,
    block_M: int,
    block_N: int,
    block_K: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
) -> str:
    """
    Return a TileLang ``@T.prim_func`` source string for matrix multiplication.

    The generated kernel uses ``T.gemm`` to compute ``C = A @ B`` where:
    - ``A`` has shape ``(_M, _K)`` and dtype *dtype*
    - ``B`` has shape ``(_K, _N)`` and dtype *dtype*
    - ``C`` has shape ``(_M, _N)`` and dtype *accum_dtype* (fp32 for fp16 input)

    ``_M``, ``_N``, ``_K`` are injected by the factory function wrapper.
    """
    tl_dtype  = tilelang_dtype(dtype)
    tl_accum  = tilelang_dtype(accum_dtype)
    prim_name = f"{name}_prim_fn"

    code = IndentedBuffer()
    code.writeline("import tilelang.language as T")
    code.writeline("")
    code.writeline(f"_block_M = {block_M}")
    code.writeline(f"_block_N = {block_N}")
    code.writeline(f"_block_K = {block_K}")
    code.writeline("")
    code.writeline("@T.prim_func")
    code.writeline(f"def {prim_name}(")
    with code.indent():
        code.writeline(f"A: T.Tensor((_M, _K), '{tl_dtype}'),")
        code.writeline(f"B: T.Tensor((_K, _N), '{tl_dtype}'),")
        code.writeline(f"C: T.Tensor((_M, _N), '{tl_accum}'),")
    code.writeline("):")
    with code.indent():
        code.writeline(
            "with T.Kernel("
            "T.ceildiv(_N, _block_N) * T.ceildiv(_M, _block_M), "
            "is_npu=True) as (cid, _):"
        )
        with code.indent():
            code.writeline("by = cid // T.ceildiv(_N, _block_N)")
            code.writeline("bx = cid % T.ceildiv(_N, _block_N)")
            code.writeline("")
            code.writeline(
                f"A_shared = T.alloc_shared((_block_M, _block_K), '{tl_dtype}')"
            )
            code.writeline(
                f"B_shared = T.alloc_shared((_block_K, _block_N), '{tl_dtype}')"
            )
            code.writeline(
                f"C_local = T.alloc_fragment((_block_M, _block_N), '{tl_accum}')"
            )
            code.writeline("")
            code.writeline(
                "for k in T.Pipelined(T.ceildiv(_K, _block_K), num_stages=2):"
            )
            with code.indent():
                code.writeline(
                    "T.copy(A[by * _block_M, k * _block_K], A_shared)"
                )
                code.writeline(
                    "T.copy(B[k * _block_K, bx * _block_N], B_shared)"
                )
                code.writeline(
                    "T.gemm(A_shared, B_shared, C_local, initC=(k == 0))"
                )
            code.writeline("")
            code.writeline(
                "T.copy(C_local, C[by * _block_M, bx * _block_N])"
            )
    src = code.getvalue()
    print("====== TileLang T.gemm prim_func ======")
    print(src)
    return src


_MAX_REDUCTIONS_PER_KERNEL = 2


def _resolve_operand(operand) -> str:
    """Convert a CSEVariable or string operand to a string suitable for T.v* calls."""
    if isinstance(operand, (int, float)):
        return repr(operand)
    return str(operand)


def _is_scalar_literal_operand(operand: Any) -> bool:
    return isinstance(operand, (int, float)) and not isinstance(operand, bool)


def _tilelang_expr_key(
    var,
    var_bufs: dict,
    var_ops: dict,
    var_consts: Optional[dict] = None,
    _memo: Optional[dict[str, tuple]] = None,
) -> tuple:
    """Build a stable key for reusing already materialized TileLang expressions."""
    if _memo is None:
        _memo = {}

    var_name = str(var)
    if var_name in _memo:
        return _memo[var_name]
    if var_name in var_bufs:
        key = ("buf", var_bufs[var_name])
    elif var_consts and var_name in var_consts:
        key = ("const", str(var_consts[var_name]))
    elif var_name in var_ops:
        op_name, operands = var_ops[var_name]
        key = (
            "op",
            op_name,
            tuple(
                _tilelang_expr_key(op, var_bufs, var_ops, var_consts, _memo)
                for op in operands
            ),
        )
    else:
        key = ("literal", var_name)
    _memo[var_name] = key
    return key


def _build_vec_ops(
    var,
    target_buf: str,
    ops_list: list,
    var_bufs: dict,
    var_ops: dict,
    var_consts: Optional[dict] = None,
    reusable_expr_bufs: Optional[dict[tuple, str]] = None,
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
        if reusable_expr_bufs is not None:
            expr_key = _tilelang_expr_key(var, var_bufs, var_ops, var_consts)
            if expr_key in reusable_expr_bufs:
                return reusable_expr_bufs[expr_key]

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
                src = _build_vec_ops(
                    op,
                    inter_buf,
                    ops_list,
                    var_bufs,
                    var_ops,
                    var_consts,
                    reusable_expr_bufs,
                    _visited,
                )
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


def _is_reduction_symbol_name(name: str) -> bool:
    return name == "r" or name == "rindex" or re.match(r"r\d+(?:_|$)", name) is not None


def _has_reduction_index(index: sympy.Expr) -> bool:
    return any(
        _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
        for symbol in index.free_symbols
    )


def _has_pointwise_index(index: sympy.Expr) -> bool:
    return any(
        getattr(symbol, "name", str(symbol)).startswith("x")
        for symbol in index.free_symbols
    )


def _dense_reduction_matrix_expected_index(
    index: sympy.Expr,
    reduction_numel: sympy.Expr,
) -> Optional[sympy.Expr]:
    x_symbols = [
        symbol for symbol in index.free_symbols
        if getattr(symbol, "name", str(symbol)).startswith("x")
    ]
    r_symbols = [
        symbol for symbol in index.free_symbols
        if _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
    ]
    if len(x_symbols) != 1 or len(r_symbols) != 1:
        return None
    return x_symbols[0] * reduction_numel + r_symbols[0]


def _is_dense_reduction_matrix_index(
    index: sympy.Expr,
    reduction_numel: sympy.Expr,
) -> bool:
    expected = _dense_reduction_matrix_expected_index(index, reduction_numel)
    if expected is None:
        return False
    try:
        return bool(sympy.simplify(index - expected) == 0)
    except Exception:
        return index == expected


def _tilelang_index_expr(
    expr: sympy.Expr,
    pointwise_index: str = "cid",
    reduction_index: str = "_tl_i",
    symbol_replacements: Optional[dict[sympy.Symbol, str]] = None,
) -> str:
    if symbol_replacements is None:
        symbol_replacements = {}
    if isinstance(expr, (int, sympy.Integer)):
        return str(int(expr))
    if isinstance(expr, sympy.Symbol):
        if expr in symbol_replacements:
            return symbol_replacements[expr]
        name = getattr(expr, "name", str(expr))
        if name.startswith("x"):
            return pointwise_index
        if _is_reduction_symbol_name(name):
            return reduction_index
        return name
    if isinstance(expr, FloorDiv):
        return (
            f"({_tilelang_index_expr(expr.args[0], pointwise_index, reduction_index, symbol_replacements)} "
            f"// {_tilelang_index_expr(expr.args[1], pointwise_index, reduction_index, symbol_replacements)})"
        )
    if isinstance(expr, ModularIndexing):
        base, divisor, modulus = expr.args
        base_expr = _tilelang_index_expr(
            base,
            pointwise_index,
            reduction_index,
            symbol_replacements,
        )
        if divisor != 1:
            base_expr = (
                f"({base_expr} // "
                f"{_tilelang_index_expr(divisor, pointwise_index, reduction_index, symbol_replacements)})"
            )
        return (
            f"({base_expr} % "
            f"{_tilelang_index_expr(modulus, pointwise_index, reduction_index, symbol_replacements)})"
        )
    if isinstance(expr, sympy.Add):
        return "(" + " + ".join(
            _tilelang_index_expr(
                arg,
                pointwise_index,
                reduction_index,
                symbol_replacements,
            )
            for arg in expr.args
        ) + ")"
    if isinstance(expr, sympy.Mul):
        return "(" + " * ".join(
            _tilelang_index_expr(
                arg,
                pointwise_index,
                reduction_index,
                symbol_replacements,
            )
            for arg in expr.args
        ) + ")"
    return str(expr)


def _is_zero_index(index: sympy.Expr) -> bool:
    try:
        return bool(sympy.simplify(index) == 0)
    except Exception:
        return False


class TileLangKernel(NPUIndexTritonKernel):
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
        self._tl_input_numels: dict[str, sympy.Expr] = {}
        self._tl_static_axis_vars: dict[sympy.Symbol, tuple[str, sympy.Expr]] = {}
        self._tl_static_value_vars: dict[tuple[str, str], tuple[str, sympy.Expr]] = {}
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

    def want_no_x_dim(self) -> bool:
        return False

    def initialize_range_tree(self, pid_cache: Optional[dict[str, str]]) -> None:
        """
        Use NPUIndex range-tree entries so NPUTritonScheduling's indexing
        transform can remove/substitute axes, while keeping TileLang's existing
        x/r prefix convention.
        """
        if pid_cache is None:
            pid_cache = {}

        active_prefixes = [
            prefix for prefix in ("z", "y", "x", "r")
            if prefix in self.numels
        ]
        no_r_dim = not self.inside_reduction or not self.features.is_reduction()

        if self.no_x_dim:
            tensor_dims = ["r"]
        elif no_r_dim:
            tensor_dims = ["z", "y", "x"]
        else:
            tensor_dims = ["z", "y", "x", "r"]

        grid_dims = ["x", "y", "z"]
        tensor_dim_map = {
            prefix: idx
            for idx, prefix in enumerate(p for p in tensor_dims if p in active_prefixes)
        }
        grid_dim_map = {
            prefix: idx
            for idx, prefix in enumerate(p for p in grid_dims if p in active_prefixes)
        }

        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix.startswith("r")
            tensor_dim = tensor_dim_map.get(prefix)
            grid_dim = None if is_reduction else grid_dim_map.get(prefix)
            index = i if grid_dim is None else grid_dim
            self.range_trees.append(
                IterationRangesRootNPUIndex(
                    f"{prefix}index",
                    self.numels[prefix],
                    prefix,
                    index,
                    self,
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                )
            )

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
        if name in self.cse.store_cache:
            return self.cse.store_cache[name]

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
        try:
            self._tl_input_numels[local_name] = sympy_product(
                V.graph.get_buffer(name).get_size()
            )
        except Exception:
            pass
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
            # Unconditional overwrite: store() may run first (e.g. fused cast
            # epilogue) and register the wrong _out_ptr*_local for this name.
            # The reduction buffer must always win.
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
        if loc in self._tl_input_numels:
            return f"({self._static_value_var('B', self._tl_input_numels[loc])},)"
        index = self._tl_input_indices.get(loc, sympy.S.Zero)
        modular_terms = list(index.atoms(ModularIndexing))
        if len(modular_terms) == 1:
            return f"({self._static_value_var('B', modular_terms[0].args[2])},)"
        factor = self._pointwise_broadcast_factor(loc)
        if factor is None:
            return "(T.ceildiv(_xnumel, _XBLOCK),)"
        return f"(T.ceildiv(_xnumel, {self._static_value_var('B', factor)}),)"

    def _pointwise_broadcast_load_index(self, loc: str) -> str:
        factor = self._pointwise_broadcast_factor(loc)
        if factor is None:
            return "cid"
        return f"((cid * _XBLOCK) // {self._static_value_var('B', factor)})"

    def _pointwise_broadcast_lane_index(
        self,
        loc: str,
        pointwise_index: str = "(cid * _XBLOCK + _tl_i)",
    ) -> str:
        factor = self._pointwise_broadcast_factor(loc)
        if factor is not None:
            factor_var = self._static_value_var("B", factor)
            index = f"({pointwise_index} // {factor_var})"
            if loc in self._tl_input_numels:
                numel_var = self._static_value_var("B", self._tl_input_numels[loc])
                index = f"({index} % {numel_var})"
            return index
        index = self._tl_input_indices.get(loc, sympy.S.Zero)
        return _tilelang_index_expr(
            index,
            pointwise_index=pointwise_index,
        )

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

    def _is_dense_reduction_matrix_loc(self, loc: str) -> bool:
        reduction_numel = self.numels.get("r")
        if reduction_numel is None:
            return False
        return _is_dense_reduction_matrix_index(
            self._tl_input_indices.get(loc, sympy.S.Zero),
            reduction_numel,
        )

    def _reduction_symbol_replacements(
        self,
        index: sympy.Expr,
    ) -> dict[sympy.Symbol, str]:
        r_symbols = [
            symbol for symbol in index.free_symbols
            if _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
        ]
        if len(r_symbols) <= 1:
            return {}

        def sort_key(symbol: sympy.Symbol):
            coeff = sympy.expand(index).coeff(symbol)
            if coeff is not None and not getattr(coeff, "free_symbols", None):
                try:
                    return int(coeff)
                except Exception:
                    pass
            node = (
                self.range_tree_nodes.get(symbol)
                or self.range_tree_nodes_removed.get(symbol)
            )
            if node is not None:
                return node.sorted_order or 0
            return str(symbol)

        ordered = sorted(r_symbols, key=sort_key)
        replacements: dict[sympy.Symbol, str] = {}
        factor = 1
        for i, symbol in enumerate(ordered):
            node = (
                self.range_tree_nodes.get(symbol)
                or self.range_tree_nodes_removed.get(symbol)
            )
            if node is None:
                raise NotImplementedError(
                    "TileLang reduction could not determine range for "
                    f"index symbol {symbol}"
                )
            try:
                extent = int(node.length)
            except Exception as exc:
                raise NotImplementedError(
                    "TileLang reduction only supports static expanded "
                    f"reduction axes; {symbol} has length {node.length}"
                ) from exc

            if i == 0:
                replacements[symbol] = f"(_tl_i % {extent})"
            elif i == len(ordered) - 1:
                replacements[symbol] = f"(_tl_i // {factor})"
            else:
                replacements[symbol] = f"((_tl_i // {factor}) % {extent})"
            factor *= extent
        return replacements

    def _axis_extent(self, symbol: sympy.Symbol) -> sympy.Expr:
        node = (
            self.range_tree_nodes.get(symbol)
            or self.range_tree_nodes_removed.get(symbol)
        )
        if node is not None:
            return node.length
        name = getattr(symbol, "name", str(symbol))
        if name.startswith("x"):
            return self.numels.get("x", sympy.S.One)
        if _is_reduction_symbol_name(name):
            return self.numels.get("r", sympy.S.One)
        return sympy.S.One

    def _static_axis_var(self, symbol: sympy.Symbol) -> str:
        if symbol not in self._tl_static_axis_vars:
            name = f"_R{len(self._tl_static_axis_vars)}"
            self._tl_static_axis_vars[symbol] = (name, self._axis_extent(symbol))
        return self._tl_static_axis_vars[symbol][0]

    def _static_value_var(self, prefix: str, value: sympy.Expr) -> str:
        key = (prefix, str(value))
        if key not in self._tl_static_value_vars:
            name = f"_{prefix}{len(self._tl_static_value_vars)}"
            self._tl_static_value_vars[key] = (name, value)
        return self._tl_static_value_vars[key][0]

    def tilelang_static_axis_args(self) -> list[tuple[str, sympy.Expr]]:
        return list(dict.fromkeys([
            *self._tl_static_axis_vars.values(),
            *self._tl_static_value_vars.values(),
        ]))

    def _index_symbols_by_stride(self, index: sympy.Expr) -> list[sympy.Symbol]:
        def stride(symbol: sympy.Symbol):
            coeff = sympy.expand(index).coeff(symbol)
            if coeff is not None and not getattr(coeff, "free_symbols", None):
                try:
                    return int(coeff)
                except Exception:
                    pass
            node = (
                self.range_tree_nodes.get(symbol)
                or self.range_tree_nodes_removed.get(symbol)
            )
            if node is not None:
                return node.sorted_order or 0
            return 0

        symbols = [
            symbol for symbol in index.free_symbols
            if getattr(symbol, "name", str(symbol)).startswith("x")
            or _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
        ]
        return sorted(symbols, key=stride, reverse=True)

    def _reduction_matrix_symbols(self, index: sympy.Expr) -> list[sympy.Symbol]:
        return self._index_symbols_by_stride(index)

    def _reduction_matrix_dim_expr(self, symbol: sympy.Symbol, index: sympy.Expr) -> str:
        symbols = self._reduction_matrix_symbols(index)
        name = getattr(symbol, "name", str(symbol))
        if name.startswith("x") and len([
            s for s in symbols
            if getattr(s, "name", str(s)).startswith("x")
        ]) == 1:
            return "_xnumel"
        if _is_reduction_symbol_name(name) and len([
            s for s in symbols
            if _is_reduction_symbol_name(getattr(s, "name", str(s)))
        ]) == 1:
            return "_rnumel"
        return self._static_axis_var(symbol)

    def _reduction_matrix_local_shape_from_index(self, index: sympy.Expr) -> str:
        dims: list[str] = []
        for symbol in self._reduction_matrix_symbols(index):
            name = getattr(symbol, "name", str(symbol))
            dims.append("_XBLOCK" if name.startswith("x") else self._reduction_matrix_dim_expr(symbol, index))
        return f"({', '.join(dims)},)"

    def _reduction_scalar_local_shape_from_index(self, index: sympy.Expr) -> str:
        dims = [
            "_XBLOCK" if getattr(symbol, "name", str(symbol)).startswith("x") else "1"
            for symbol in self._reduction_matrix_symbols(index)
        ]
        return f"({', '.join(dims)},)"

    def _reduction_matrix_reduce_dims_from_index(self, index: sympy.Expr) -> str:
        dims = [
            str(i)
            for i, symbol in enumerate(self._reduction_matrix_symbols(index))
            if _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
        ]
        return f"[{', '.join(dims)}]"

    def _representative_reduction_matrix_index(self) -> Optional[sympy.Expr]:
        for loc in self._tl_reduction_input_locs:
            index = self._tl_input_indices.get(loc, sympy.S.Zero)
            if _has_pointwise_index(index) and _has_reduction_index(index):
                return index
        for loc, index in self._output_indices.items():
            if _has_pointwise_index(index) and _has_reduction_index(index):
                return index
        return None

    def _pointwise_symbol_replacements(
        self,
        index: sympy.Expr,
        pointwise_index: str = "cid",
    ) -> dict[sympy.Symbol, str]:
        x_symbols = [
            symbol for symbol in index.free_symbols
            if getattr(symbol, "name", str(symbol)).startswith("x")
        ]
        if len(x_symbols) <= 1:
            return {symbol: pointwise_index for symbol in x_symbols}

        ordered = self._index_symbols_by_stride(
            sum(symbol * sympy.expand(index).coeff(symbol) for symbol in x_symbols)
        )
        ordered = [
            symbol for symbol in ordered
            if getattr(symbol, "name", str(symbol)).startswith("x")
        ]
        replacements: dict[sympy.Symbol, str] = {}
        factor = 1
        for i, symbol in enumerate(reversed(ordered)):
            extent = self._axis_extent(symbol)
            try:
                extent_int = int(extent)
            except Exception as exc:
                raise NotImplementedError(
                    "TileLang reduction only supports static expanded "
                    f"pointwise axes; {symbol} has length {extent}"
                ) from exc
            if i == 0:
                replacements[symbol] = f"(cid % {extent_int})"
            elif i == len(ordered) - 1:
                replacements[symbol] = f"(cid // {factor})"
            else:
                replacements[symbol] = f"((cid // {factor}) % {extent_int})"
            factor *= extent_int
        return replacements

    def _reduction_matrix_shape_from_index(self, index: sympy.Expr) -> str:
        reduction_numel = self.numels.get("r", sympy.S.One)
        if _is_dense_reduction_matrix_index(index, reduction_numel):
            return "(_xnumel, _rnumel)"

        dims: list[str] = []
        symbols = self._index_symbols_by_stride(index)
        for symbol in symbols:
            dims.append(self._reduction_matrix_dim_expr(symbol, index))
        return f"({', '.join(dims)},)"

    def _reduction_matrix_index_for_output(self, loc: str) -> sympy.Expr:
        index = self._output_indices.get(loc, sympy.S.Zero)
        if not (_has_pointwise_index(index) and _has_reduction_index(index)):
            representative_index = self._representative_reduction_matrix_index()
            if representative_index is not None:
                return representative_index
        return index

    def _reduction_matrix_start_index_tuple(self, index: sympy.Expr) -> str:
        reduction_numel = self.numels.get("r", sympy.S.One)
        if _is_dense_reduction_matrix_index(index, reduction_numel):
            return "cid * _XBLOCK, 0"

        replacements = {}
        replacements.update(self._pointwise_symbol_replacements(index, "cid * _XBLOCK"))
        for symbol in index.free_symbols:
            if _is_reduction_symbol_name(getattr(symbol, "name", str(symbol))):
                replacements[symbol] = "0"
        symbols = self._index_symbols_by_stride(index)
        parts = [
            _tilelang_index_expr(symbol, symbol_replacements=replacements)
            for symbol in symbols
        ]
        return ", ".join(parts)

    def _emit_reduction_matrix_load(
        self,
        code: IndentedBuffer,
        var: str,
        loc: str,
    ) -> None:
        index = self._tl_input_indices.get(loc, sympy.S.Zero)
        reduction_numel = self.numels.get("r", sympy.S.One)
        if _is_dense_reduction_matrix_index(index, reduction_numel):
            code.writeline(f"T.copy({var}[cid * _XBLOCK, 0], {loc})")
            return
        code.writeline(
            f"T.copy({var}[{self._reduction_matrix_start_index_tuple(index)}], {loc})"
        )

    def _emit_reduction_matrix_store(
        self,
        code: IndentedBuffer,
        var: str,
        loc: str,
    ) -> None:
        index = self._reduction_matrix_index_for_output(loc)
        if (
            self._representative_reduction_matrix_index() is None
            and _has_reduction_index(index)
            and not _has_pointwise_index(index)
        ):
            code.writeline(f"T.copy({loc}, {var}[0, 0])")
            return
        reduction_numel = self.numels.get("r", sympy.S.One)
        if _is_dense_reduction_matrix_index(index, reduction_numel):
            code.writeline(f"T.copy({loc}, {var}[cid * _XBLOCK, 0])")
            return
        code.writeline(
            f"T.copy({loc}, {var}[{self._reduction_matrix_start_index_tuple(index)}])"
        )

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
        index = self._reduction_matrix_index_for_output(loc)
        if _is_zero_index(index):
            index = self._representative_reduction_matrix_index()
        if index is None:
            return "(1, _RBLOCK)" if self._reduction_output_extent(loc, tensor_arg_names) == "matrix" else "(1, 1)"
        if self._reduction_output_extent(loc, tensor_arg_names) == "matrix":
            if (
                self._representative_reduction_matrix_index() is None
                and _has_reduction_index(index)
                and not _has_pointwise_index(index)
            ):
                return "(1, _RBLOCK)"
            return self._reduction_matrix_local_shape_from_index(index)
        return self._reduction_scalar_local_shape_from_index(index)

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        if self._reduction_outputs:
            return self._codegen_reduction_kernel(name)

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

        def emit_pointwise_body(
            code: IndentedBuffer,
            vector_len: str,
            base_index: str,
            pointwise_index: str,
        ) -> None:
            # ---- allocate input buffers (L1/shared) ----
            for _, (var, loc, dtype) in self._tl_inputs.items():
                kind = self._pointwise_input_kind(loc)
                if kind == "scalar":
                    code.writeline(
                        f"{loc}_brc_src = T.alloc_shared((1,), "
                        f"'{tilelang_dtype(dtype)}')"
                    )
                code.writeline(
                    f"{loc} = T.alloc_shared(({vector_len},), '{tilelang_dtype(dtype)}')"
                )

            # ---- allocate output buffers (fragment) ----
            input_locs = {loc for _, loc, _ in self._tl_inputs.values()}
            for _, (var, loc, dtype) in self._tl_outputs.items():
                if loc not in input_locs:
                    code.writeline(
                        f"{loc} = T.alloc_shared(({vector_len},), '{tilelang_dtype(dtype)}')"
                    )
            code.writeline("")

            # ---- T.copy: GM -> L1 for every input ----
            for _, (var, loc, _) in self._tl_inputs.items():
                kind = self._pointwise_input_kind(loc)
                if kind == "scalar":
                    code.writeline(f"T.copy({var}[0], {loc}_brc_src)")
                    code.writeline(f"T.vbrc({loc}_brc_src, {loc})")
                elif kind == "broadcast":
                    load_index = self._pointwise_broadcast_lane_index(
                        loc,
                        pointwise_index=pointwise_index,
                    )
                    code.writeline(f"for _tl_i in T.Parallel({vector_len}):")
                    with code.indent():
                        code.writeline(f"{loc}[_tl_i] = {var}[{load_index}]")
                else:
                    code.writeline(f"T.copy({var}[{base_index}], {loc})")
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

                self._emit_vec_ops_with_lifetime_reuse(
                    code,
                    ops_list,
                    f"({vector_len},)",
                    dtype,
                    scalar_cache,
                    already_allocated,
                )

            code.writeline("")

            # ---- T.copy: fragment -> GM for every output ----
            for _, (var, loc, _) in self._tl_outputs.items():
                code.writeline(f"T.copy({loc}, {var}[{base_index}])")

        def emit_pointwise_prim_func(code: IndentedBuffer) -> None:
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
                    code.writeline(
                        "for _tl_block in T.serial(T.ceildiv(_XBLOCK, _XBLOCK_SUB)):"
                    )
                    with code.indent():
                        emit_pointwise_body(
                            code,
                            "_XBLOCK_SUB",
                            "cid * _XBLOCK + _tl_block * _XBLOCK_SUB",
                            "(cid * _XBLOCK + _tl_block * _XBLOCK_SUB + _tl_i)",
                        )

        code = IndentedBuffer()
        code.writeline("import tilelang.language as T")
        code.writeline("import math as _math")
        code.writeline("")
        code.writeline("_XBLOCK = X0BLOCK")
        code.writeline("_XBLOCK_SUB = X0BLOCK_SUB")
        code.writeline("")
        emit_pointwise_prim_func(code)

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
                input_shapes[var] = self._reduction_matrix_shape_from_index(
                    self._tl_input_indices.get(loc, sympy.S.Zero)
                )
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
                if vector_epilogue:
                    out_loc = next(
                        loc for _, (var, loc, _) in self._tl_outputs.items()
                        if var == argdef.name
                    )
                    out_index = self._reduction_matrix_index_for_output(out_loc)
                    if (
                        self._representative_reduction_matrix_index() is None
                        and _has_reduction_index(out_index)
                        and not _has_pointwise_index(out_index)
                    ):
                        shape = "(1, _rnumel)"
                    else:
                        shape = self._reduction_matrix_shape_from_index(out_index)
                else:
                    shape = "(_xnumel, 1)"
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
        code.writeline("_XBLOCK = X0BLOCK")
        matrix_index = self._representative_reduction_matrix_index()
        matrix_local_shape = (
            self._reduction_matrix_local_shape_from_index(matrix_index)
            if matrix_index is not None else "(1, _RBLOCK)"
        )
        scalar_reduce_shape = (
            self._reduction_scalar_local_shape_from_index(matrix_index)
            if matrix_index is not None else "(1, 1)"
        )
        reduce_dims = (
            self._reduction_matrix_reduce_dims_from_index(matrix_index)
            if matrix_index is not None else "[1]"
        )
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
                buffer_shapes: dict[str, str] = {}
                for _, (var, loc, dtype) in self._tl_inputs.items():
                    kind = self._reduction_input_kind(loc)
                    if kind in {"scalar", "row_broadcast"}:
                        code.writeline(
                            f"{loc}_brc_src = T.alloc_shared((1, 1), "
                            f"'{tilelang_dtype(dtype)}')"
                        )
                        buffer_shapes[loc + "_brc_src"] = "(1, 1)"
                    shape = (
                        self._reduction_matrix_local_shape_from_index(
                            self._tl_input_indices.get(loc, sympy.S.Zero)
                        )
                        if kind == "matrix" else "(1, _RBLOCK)"
                    )
                    code.writeline(
                        f"{loc} = T.alloc_shared({shape}, '{tilelang_dtype(dtype)}')"
                    )
                    buffer_shapes[loc] = shape

                input_locs = {loc for _, loc, _ in self._tl_inputs.values()}
                already_allocated = set(input_locs)
                scalar_cache: dict[tuple[str, str], str] = {}
                reusable_expr_bufs: dict[tuple, str] = {}
                for _, (var, loc, dtype) in self._tl_outputs.items():
                    if var not in tensor_arg_names:
                        continue
                    code.writeline(
                        f"{loc} = T.alloc_shared({self._reduction_output_shape(loc, tensor_arg_names)}, "
                        f"'{tilelang_dtype(dtype)}')"
                    )
                    buffer_shapes[loc] = self._reduction_output_shape(
                        loc, tensor_arg_names
                    )
                    already_allocated.add(loc)
                for loc, (_, dtype) in self._reduction_outputs.items():
                    if loc not in already_allocated:
                        code.writeline(
                            f"{loc} = T.alloc_shared({scalar_reduce_shape}, '{tilelang_dtype(dtype)}')"
                        )
                        already_allocated.add(loc)
                    buffer_shapes[loc] = scalar_reduce_shape
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
                        self._emit_reduction_matrix_load(code, var, loc)
                code.writeline("")

                for out_loc, (result_var, dtype) in self._reduction_outputs.items():
                    reduction_type, value, _ = self._reduction_vars[str(result_var)]
                    reduce_input = f"_{result_var}_reduce_in"
                    ops_list: list[tuple] = []

                    src = _build_vec_ops(
                        value,
                        reduce_input,
                        ops_list,
                        self._var_bufs,
                        self._var_ops,
                        self._var_consts,
                        reusable_expr_bufs,
                    )

                    if ops_list:
                        last_op, last_operands, _ = ops_list[-1]
                        if (
                            last_op == "exp"
                            and len(last_operands) == 1
                            and isinstance(last_operands[0], str)
                            and last_operands[0] not in input_locs
                            and last_operands[0] not in {
                                loc for _, loc, _ in self._tl_outputs.values()
                            }
                            and last_operands[0] not in self._reduction_outputs
                        ):
                            reduce_input = last_operands[0]
                        elif reduce_input not in already_allocated:
                            code.writeline(
                                f"{reduce_input} = T.alloc_shared({matrix_local_shape}, "
                                f"'{tilelang_dtype(dtype)}')"
                            )
                            already_allocated.add(reduce_input)
                            buffer_shapes[reduce_input] = matrix_local_shape
                        ops_list[-1] = (last_op, last_operands, reduce_input)
                        reusable_expr_bufs[
                            _tilelang_expr_key(
                                value,
                                self._var_bufs,
                                self._var_ops,
                                self._var_consts,
                            )
                        ] = reduce_input

                    self._emit_vec_ops_with_lifetime_reuse(
                        code,
                        ops_list,
                        matrix_local_shape,
                        dtype,
                        scalar_cache,
                        already_allocated,
                        buffer_shapes,
                    )

                    reduce_src = reduce_input if ops_list else src
                    code.writeline(
                        f"T.reduce({reduce_src}, {out_loc}, dims={reduce_dims}, "
                        f"reduce_mode='{reduction_type}')"
                    )

                for out_loc, (result_var, dtype) in self._output_vars.items():
                    if out_loc not in vector_epilogue_locs:
                        continue
                    out_kind = self._reduction_output_extent(out_loc, tensor_arg_names)
                    ops_list: list[tuple] = []

                    src = _build_vec_ops(
                        result_var,
                        out_loc,
                        ops_list,
                        self._var_bufs,
                        self._var_ops,
                        self._var_consts,
                        reusable_expr_bufs,
                    )

                    if not ops_list:
                        if src != out_loc:
                            code.writeline(f"T.copy({src}, {out_loc})")
                        continue

                    last_op, last_operands, _ = ops_list[-1]
                    ops_list[-1] = (last_op, last_operands, out_loc)

                    self._emit_vec_ops_with_lifetime_reuse(
                        code,
                        ops_list,
                        matrix_local_shape if out_kind == "matrix" else scalar_reduce_shape,
                        dtype,
                        scalar_cache,
                        already_allocated,
                        buffer_shapes,
                    )

                # ---- scalar epilogue: per-row outputs computed from reduction
                # results but with no reduction index in their output index
                # (e.g. mean = sum/N, or any f(reduction_result) per row).
                # These are in _output_vars but NOT in vector_epilogue_locs.
                scalar_epilogue_locs = {
                    loc
                    for loc in self._output_vars
                    if loc not in self._reduction_outputs
                    and loc not in vector_epilogue_locs
                    and self._is_tensor_output_loc(loc, tensor_arg_names)
                }
                for out_loc in scalar_epilogue_locs:
                    result_var, dtype = self._output_vars[out_loc]
                    ops_list: list[tuple] = []
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
                    self._emit_vec_ops_with_lifetime_reuse(
                        code,
                        ops_list,
                        "(1, 1)",
                        dtype,
                        scalar_cache,
                        already_allocated,
                        buffer_shapes,
                    )

                code.writeline("")
                for _, (var, loc, _) in self._tl_outputs.items():
                    if var not in tensor_arg_names:
                        continue
                    if loc in vector_epilogue_locs:
                        self._emit_reduction_matrix_store(code, var, loc)
                    else:
                        code.writeline(f"T.copy({loc}, {var}[cid * _XBLOCK, 0])")

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

    def _emit_vec_ops_with_lifetime_reuse(
        self,
        code: IndentedBuffer,
        ops_list: list[tuple],
        local_shape: str,
        dtype: torch.dtype,
        scalar_cache: dict[tuple[str, str], str],
        already_allocated: set[str],
        buffer_shapes: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Emit a linearized vector-op list while reusing temporary TileLang buffers.

        Buffers that exist before this emission starts are protected. Only buffers
        allocated by this op list can be returned to the free list and reused by a
        later op with the same shape/dtype.
        """
        protected_buffers = set(already_allocated)
        use_counts: dict[str, int] = {}
        buffer_map: dict[str, str] = {}
        free_buffers: dict[str, list[str]] = {}
        if buffer_shapes is None:
            buffer_shapes = {}

        for _op_name, operands, _out_buf in ops_list:
            for operand in operands:
                if isinstance(operand, str):
                    use_counts[operand] = use_counts.get(operand, 0) + 1

        def _resolve_buffer(buf):
            if isinstance(buf, str):
                return buffer_map.get(buf, buf)
            return buf

        def _is_reusable_temp(buf: str) -> bool:
            return (
                isinstance(buf, str)
                and buf.startswith("_")
                and buf not in protected_buffers
            )

        def _operand_shape(operand) -> Optional[str]:
            if not isinstance(operand, str):
                return None
            return buffer_shapes.get(buffer_map.get(operand, operand))

        def _result_shape(
            op_name: str,
            operands: list,
            logical_out_buf: str,
        ) -> str:
            if logical_out_buf in buffer_shapes:
                return buffer_shapes[logical_out_buf]
            operand_shapes = [
                shape for shape in (_operand_shape(operand) for operand in operands)
                if shape is not None
            ]
            if op_name in _UNARY_VEC_OPS and operand_shapes:
                return operand_shapes[0]
            if local_shape in operand_shapes:
                return local_shape
            if operand_shapes:
                return operand_shapes[0]
            return local_shape

        def _allocate_buffer(logical_buf: str, shape: str) -> str:
            if logical_buf in buffer_map:
                return buffer_map[logical_buf]
            if logical_buf in already_allocated:
                actual_buf = logical_buf
            elif free_buffers.get(shape):
                actual_buf = free_buffers[shape].pop()
            else:
                actual_buf = logical_buf
                code.writeline(
                    f"{actual_buf} = T.alloc_shared({shape}, "
                    f"'{tilelang_dtype(dtype)}')"
                )
                already_allocated.add(actual_buf)
            buffer_map[logical_buf] = actual_buf
            buffer_shapes[logical_buf] = shape
            buffer_shapes[actual_buf] = shape
            return actual_buf

        for op_name, operands, logical_out_buf in ops_list:
            resolved_operands = [_resolve_buffer(operand) for operand in operands]
            shape = _result_shape(op_name, resolved_operands, logical_out_buf)
            out_buf = _allocate_buffer(logical_out_buf, shape)
            materialized_operands = self._materialize_scalar_operands(
                code, op_name, resolved_operands, dtype, scalar_cache
            )
            code.writeline(self._emit_vec_op(op_name, materialized_operands, out_buf))

            for operand in operands:
                if not isinstance(operand, str):
                    continue
                use_counts[operand] = use_counts.get(operand, 0) - 1
                if use_counts[operand] != 0:
                    continue
                actual_operand = _resolve_buffer(operand)
                if (
                    _is_reusable_temp(operand)
                    and actual_operand not in protected_buffers
                    and actual_operand != out_buf
                    and actual_operand not in free_buffers.setdefault(
                        buffer_shapes.get(actual_operand, local_shape), []
                    )
                ):
                    free_buffers[buffer_shapes.get(actual_operand, local_shape)].append(
                        actual_operand
                    )

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
        # TileLang NPU codegen uses T.reduce on the full reduction dimension in one
        # shot (_RBLOCK = _rnumel). The persistent path in inductor matches this:
        # it only calls store_reduction() and never emits the extra store() calls
        # that the non-persistent path generates and that confuse our codegen.
        return True

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

    # ------------------------------------------------------------------
    # TileLang T.gemm (matmul) codegen
    # ------------------------------------------------------------------

    def define_kernel_matmul(
        self,
        src_code: str,
        node_schedule: list,
        kernel_name: str,
    ) -> str:
        """
        Splice a TileLang T.gemm kernel factory + cache + wrapper into
        ``wrapper.header``.

        Pattern emitted at module level::

            import tilelang as _tilelang_<N>

            def _prim_factory_<name>(M, N, K):
                _M, _N, _K = M, N, K
                # @T.prim_func definition (src_code)
                return <name>_prim_fn

            _<name>_cache = {}

            def <name>(A, B, C, M, N, K):
                _key = (int(M), int(N), int(K))
                if _key not in _<name>_cache:
                    _<name>_cache[_key] = _tilelang_<N>.compile(
                        _prim_factory_<name>(*_key), target='npuir')
                _<name>_cache[_key](A, B, C)
        """
        wrapper = V.graph.wrapper_code

        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        wrapper.src_to_kernel[src_code] = kernel_name
        src_code_final = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        tl_pkg_root: Optional[str] = None
        try:
            import tilelang as _tl
            import os as _os
            tl_pkg_root = _os.path.dirname(_os.path.dirname(_tl.__file__))
        except ImportError:
            pass

        suffix       = kernel_name.rsplit("_", 1)[-1]
        import_alias = f"_tilelang_{suffix}"
        cache_var    = f"_{kernel_name}_cache"
        factory_fn   = f"_prim_factory_{kernel_name}"
        prim_fn_name = f"{kernel_name}_prim_fn"

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        meta_comment = f"{origins}\n{detailed}".strip()

        code = IndentedBuffer()
        code.writeline(f"\n# TileLang T.gemm kernel - {meta_comment}")
        if tl_pkg_root:
            code.writeline("import sys as _sys")
            code.writeline(
                f"if {tl_pkg_root!r} not in _sys.path: "
                f"_sys.path.insert(0, {tl_pkg_root!r})"
            )
        code.writeline(f"import tilelang as {import_alias}")
        code.writeline("")

        code.writeline(f"def {factory_fn}(M, N, K):")
        with code.indent():
            code.writeline("_M = M")
            code.writeline("_N = N")
            code.writeline("_K = K")
            code.splice(src_code_final)
            code.writeline(f"return {prim_fn_name}")

        code.writeline("")
        code.writeline(f"{cache_var} = {{}}")
        code.writeline("")

        code.writeline(f"def {kernel_name}(A, B, C, M, N, K):")
        with code.indent():
            code.writeline("_key = (int(M), int(N), int(K))")
            code.writeline(f"if _key not in {cache_var}:")
            with code.indent():
                code.writeline(f"{cache_var}[_key] = {import_alias}.compile(")
                with code.indent():
                    code.writeline(
                        f"{factory_fn}(_key[0], _key[1], _key[2]), target='npuir', "
                        f"pass_configs={_TILELANG_NPUIR_PASS_CONFIGS!r}"
                    )
                code.writeline(")")
            code.writeline(f"{cache_var}[_key](A, B, C)")

        generated = code.getvalue()
        print("====== TileLang T.gemm wrapper ======")
        print(generated)
        wrapper.header.splice(generated)
        return kernel_name

    def codegen_template(
        self,
        template_node,
        epilogue_nodes: list,
        only_gen_src_code: bool = False,
    ):
        """
        Intercept TileLang T.gemm templates; delegate everything else to the
        NPU Triton scheduling backend.

        A template is recognised as TileLang when its ``TemplateBuffer``'s
        ``make_kernel_render`` attribute is a ``_TileLangGemmRender`` instance,
        which is set by ``TileLangGemmCaller.output_node()``.
        """
        ir_node = template_node.node
        render  = getattr(ir_node, "make_kernel_render", None)

        if not isinstance(render, _TileLangGemmRender):
            # Not a TileLang template — let the NPU Triton path handle it.
            return self._triton_scheduling.codegen_template(
                template_node, epilogue_nodes, only_gen_src_code
            )

        params = render.params

        # Build a unique kernel name.
        fused_name = (
            get_fused_kernel_name(
                [template_node, *epilogue_nodes],
                config.triton.descriptive_names,
            )
            if config.triton.descriptive_names
            else ""
        )
        suffix      = V.graph.wrapper_code.next_kernel_suffix()
        kernel_name = "_".join(filter(None, ["tilelang_mm", fused_name, suffix]))

        # Generate T.gemm prim_func source.
        src_code = codegen_tilelang_mm_src(
            kernel_name,
            block_M=params.block_M,
            block_N=params.block_N,
            block_K=params.block_K,
            dtype=params.dtype,
            accum_dtype=params.accum_dtype,
        )

        if only_gen_src_code:
            return src_code

        node_schedule = [template_node, *epilogue_nodes]
        self.define_kernel_matmul(src_code, node_schedule, kernel_name)

        # Allocate the output buffer and mark all nodes as run.
        for n in [template_node, *epilogue_nodes]:
            n.mark_run()

        # Resolve tensor argument names and matrix dimensions.
        wrapper  = V.graph.wrapper_code
        mat1_ref = ir_node.inputs[0].codegen_reference()
        mat2_ref = ir_node.inputs[1].codegen_reference()
        out_name = ir_node.get_name()
        M_expr   = str(params.M)
        N_expr   = str(params.N)
        K_expr   = str(params.K)

        wrapper.writeline(
            f"{kernel_name}({mat1_ref}, {mat2_ref}, {out_name}, "
            f"{M_expr}, {N_expr}, {K_expr})"
        )

        self.scheduler.free_buffers()

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
        if nodes is None:
            nodes = list(kernel_features.scheduler_nodes())
        return super().codegen_node_schedule(kernel_features, nodes)

    @classmethod
    def select_tiling(cls, nodes, numel, reduction_numel=1):
        if isinstance(numel, (list, tuple)):
            numel = sympy_product(numel)
        if isinstance(reduction_numel, (list, tuple)):
            reduction_numel = sympy_product(reduction_numel)
        if str(reduction_numel) == "1":
            return {"x": numel}
        return {"x": numel, "r": reduction_numel}

    def define_kernel(
        self,
        src_code: str,
        node_schedule,
        kernel: TileLangKernel,
        traced_graph_hash: Optional[str] = None,
    ) -> tuple[str, str]:
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
            kernel_name = wrapper.src_to_kernel[src_code]
            return kernel_name, src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

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
        output_arg_names = {
            var
            for var, _, _ in kernel._tl_outputs.values()
        }
        output_arg_indices = [
            i for i, arg in enumerate(tensor_call_args) if arg in output_arg_names
        ]
        if not output_arg_indices and tensor_call_args:
            output_arg_indices = [-1]
        normalized_output_arg_indices = {
            i if i >= 0 else len(tensor_call_args) + i
            for i in output_arg_indices
        }
        profile_input_args = [
            arg
            for i, arg in enumerate(tensor_call_args)
            if i not in normalized_output_arg_indices
        ]
        tensor_dtypes = [
            s.dtype for s in signature if isinstance(s, TensorArg)
        ]
        try:
            dtype_bytes = max(get_byte_per_numel(dtype) for dtype in tensor_dtypes)
        except Exception:
            dtype_bytes = 4
        buffer_count = min(
            max(
                len(kernel._tl_inputs)
                + len(kernel._tl_outputs)
                + len(kernel._reduction_outputs),
                1,
            ),
            3,
        )
        is_reduction_kernel = bool(kernel._reduction_outputs)
        fallback_xblock_hint = (
            _DEFAULT_REDUCTION_XBLOCK if is_reduction_kernel else _DEFAULT_XBLOCK
        )
        active_trees     = kernel.active_range_trees()
        numel_arg_names  = [f"{t.prefix}numel" for t in active_trees]
        static_axis_args = kernel.tilelang_static_axis_args()
        static_axis_arg_names = [name.lstrip("_") for name, _ in static_axis_args]
        shape_factory_arg_names = numel_arg_names + static_axis_arg_names
        factory_arg_names = shape_factory_arg_names + ["X0BLOCK", "X0BLOCK_SUB"]
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
        autotune_enabled_fn = f"_tilelang_autotune_enabled_{suffix}"
        autotune_warmup_fn = f"_tilelang_autotune_warmup_{suffix}"
        autotune_rep_fn = f"_tilelang_autotune_rep_{suffix}"
        autotune_timeout_fn = f"_tilelang_autotune_timeout_{suffix}"
        xblock_configs_fn = f"_tilelang_xblock_configs_{suffix}"
        code.writeline(
            "from torch_npu._inductor.codegen.tilelang import "
            f"_tilelang_autotune_enabled as {autotune_enabled_fn}, "
            f"_tilelang_autotune_warmup as {autotune_warmup_fn}, "
            f"_tilelang_autotune_rep as {autotune_rep_fn}, "
            f"_tilelang_autotune_timeout as {autotune_timeout_fn}, "
            f"_tilelang_xblock_configs as {xblock_configs_fn}"
        )
        code.writeline("")

        factory_params = (
            ", ".join(factory_arg_names) if factory_arg_names else "_dummy=None"
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
            for var_name, _extent in static_axis_args:
                code.writeline(f"{var_name} = {var_name.lstrip('_')}")
            code.splice(src_code)
            code.writeline(f"return {prim_fn_name}")

        code.writeline("")
        code.writeline(f"{cache_var} = {{}}")
        code.writeline("")

        code.writeline(f"def {kernel_name}({', '.join(outer_arg_list)}):")
        with code.indent():
            if numel_arg_names:
                key_items = [f"int({n})" for n in numel_arg_names]
                key_items.extend(f"int({extent})" for _name, extent in static_axis_args)
                code.writeline(
                    f"_key = ({', '.join(key_items)},)"
                )
            else:
                if static_axis_args:
                    code.writeline(
                        f"_key = ({', '.join(f'int({extent})' for _name, extent in static_axis_args)},)"
                    )
                else:
                    code.writeline("_key = ('static',)")
            code.writeline(f"if _key not in {cache_var}:")
            with code.indent():
                shape_factory_call = (
                    ", ".join(f"_key[{i}]" for i in range(len(shape_factory_arg_names)))
                    if shape_factory_arg_names else ""
                )
                xnumel_expr = "int(xnumel)" if "xnumel" in numel_arg_names else "1"
                rnumel_expr = "int(rnumel)" if "rnumel" in numel_arg_names else "1"
                code.writeline("_compiled_kernel = None")
                code.writeline(
                    f"_tilelang_configs = {xblock_configs_fn}("
                    f"{xnumel_expr}, {rnumel_expr}, "
                    f"is_reduction={is_reduction_kernel!r}, "
                    f"dtype_bytes={dtype_bytes!r}, "
                    f"buffer_count={buffer_count!r})"
                )
                code.writeline(
                    "_tilelang_fallback_x0block = ("
                    f"_tilelang_configs[0]['X0BLOCK'] "
                    f"if _tilelang_configs else {fallback_xblock_hint!r})"
                )
                code.writeline(
                    "_tilelang_fallback_x0block_sub = ("
                    "_tilelang_configs[0].get('X0BLOCK_SUB', _tilelang_fallback_x0block) "
                    f"if _tilelang_configs else {fallback_xblock_hint!r})"
                )
                code.writeline(
                    f"if {autotune_enabled_fn}() and len(_tilelang_configs) > 1:"
                )
                with code.indent():
                    code.writeline("try:")
                    with code.indent():
                        code.writeline("from tilelang.autotuner import AutoTuner as _TileLangAutoTuner")
                        code.writeline("def _autotune_factory(X0BLOCK, X0BLOCK_SUB):")
                        with code.indent():
                            autotune_factory_args = ", ".join(
                                [arg for arg in [shape_factory_call, "X0BLOCK", "X0BLOCK_SUB"] if arg]
                            )
                            code.writeline(f"return {factory_fn}({autotune_factory_args})")
                        code.writeline("def _tilelang_supply(_params, config=None):")
                        with code.indent():
                            code.writeline(f"return [{', '.join(profile_input_args)}]")
                        code.writeline(
                            "_tilelang_warmup = "
                            f"{autotune_warmup_fn}()"
                        )
                        code.writeline(
                            "_tilelang_rep = "
                            f"{autotune_rep_fn}()"
                        )
                        code.writeline(
                            "_tilelang_timeout = "
                            f"{autotune_timeout_fn}()"
                        )
                        code.writeline(
                            "_tilelang_tuner = _TileLangAutoTuner.from_kernel("
                            "_autotune_factory, _tilelang_configs)"
                        )
                        code.writeline("_tilelang_tuner.set_compile_args(")
                        with code.indent():
                            code.writeline(
                                f"out_idx={output_arg_indices!r}, target='npuir', "
                                f"pass_configs={_TILELANG_NPUIR_PASS_CONFIGS!r}"
                            )
                        code.writeline(")")
                        code.writeline("_tilelang_tuner.set_profile_args(")
                        with code.indent():
                            code.writeline(
                                "skip_check=True, cache_input_tensors=False, "
                                "supply_prog=_tilelang_supply, "
                                "warmup=_tilelang_warmup, rep=_tilelang_rep, "
                                "timeout=_tilelang_timeout"
                            )
                        code.writeline(")")
                        code.writeline(
                            "_tilelang_result = _tilelang_tuner.run("
                            "warmup=_tilelang_warmup, rep=_tilelang_rep, "
                            "timeout=_tilelang_timeout)"
                        )
                        code.writeline(
                            "_tilelang_selected_config = getattr(_tilelang_result, 'config', None) or {}"
                        )
                        code.writeline(
                            "_tilelang_selected_hint = _tilelang_selected_config.get("
                            "'X0BLOCK', _tilelang_fallback_x0block)"
                        )
                        code.writeline(
                            "_tilelang_selected_sub = _tilelang_selected_config.get("
                            "'X0BLOCK_SUB', _tilelang_fallback_x0block_sub)"
                        )
                        selected_factory_args = ", ".join(
                            [
                                arg for arg in [
                                    shape_factory_call,
                                    "_tilelang_selected_hint",
                                    "_tilelang_selected_sub",
                                ] if arg
                            ]
                        )
                        code.writeline(f"_compiled_kernel = {import_alias}.compile(")
                        with code.indent():
                            code.writeline(
                                f"{factory_fn}({selected_factory_args}), target='npuir', "
                                f"pass_configs={_TILELANG_NPUIR_PASS_CONFIGS!r}"
                            )
                        code.writeline(")")
                        code.writeline(
                            "print('TileLang autotune selected', "
                            "_tilelang_selected_config)"
                        )
                    code.writeline("except Exception as _tilelang_autotune_exc:")
                    with code.indent():
                        code.writeline(
                            "print('TileLang autotune failed, falling back to default compile:', "
                            "_tilelang_autotune_exc)"
                        )
                        code.writeline("_compiled_kernel = None")
                code.writeline("if _compiled_kernel is None:")
                with code.indent():
                    fallback_factory_args = ", ".join(
                        [
                            arg for arg in [
                                shape_factory_call,
                                "_tilelang_fallback_x0block",
                                "_tilelang_fallback_x0block_sub",
                            ] if arg
                        ]
                    )
                    code.writeline(f"_compiled_kernel = {import_alias}.compile(")
                    with code.indent():
                        code.writeline(
                            f"{factory_fn}({fallback_factory_args}), target='npuir', "
                            f"pass_configs={_TILELANG_NPUIR_PASS_CONFIGS!r}"
                        )
                    code.writeline(")")
                code.writeline(f"{cache_var}[_key] = _compiled_kernel")
            code.writeline(f"{cache_var}[_key]({', '.join(tensor_call_args)})")

        wrapper.header.splice(code.getvalue())
        return kernel_name, src_code
