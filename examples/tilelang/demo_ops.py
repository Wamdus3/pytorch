"""
TileLang inductor backend — exhaustive op coverage demo.

Tests every op in _BINARY_VEC_OPS / _UNARY_VEC_OPS with supported dtypes.
Per-op dtype support is validated in store() via _check_op_graph_dtype();
unsupported dtype+op combinations fall back to Triton automatically.

All T.v* ops lower to AIV instructions (hivm.func_core_type=AIV).

Run:
    TORCHINDUCTOR_NPU_BACKEND=tilelang python demo_ops.py
"""
import os
import torch
import torch_npu  # noqa: F401

DEVICE = "npu"
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "tilelang"

rtol, atol = 1e-2, 1e-2

PASS_COUNT = 0
FAIL_COUNT = 0


def _check(name, fn, *inputs, rtol=rtol, atol=atol):
    global PASS_COUNT, FAIL_COUNT
    compiled = torch.compile(fn, backend="inductor")
    ref = fn(*inputs)
    out = compiled(*inputs)
    # Cast ref to compiled output dtype (e.g. T.gemm accumulates fp16 → fp32)
    if ref.dtype != out.dtype:
        ref = ref.to(out.dtype)
    flat_ref = ref.flatten()
    flat_out = out.flatten()
    vals_ref = [f"{v:.4f}" for v in flat_ref[:4].tolist()]
    vals_out = [f"{v:.4f}" for v in flat_out[:4].tolist()]
    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        err = (out - ref).abs().max().item()
        print(f"  [PASS] {name:<38}  ref={vals_ref}  out={vals_out}  max_err={err:.6f}")
        PASS_COUNT += 1
    except AssertionError as e:
        print(f"  [FAIL] {name:<38}  ref={vals_ref}  out={vals_out}")
        print(f"         {e}")
        FAIL_COUNT += 1


# ---------------------------------------------------------------------------
# fp32 inputs
# ---------------------------------------------------------------------------
N   = 1024
x   = torch.randn(N, device=DEVICE, dtype=torch.float32)
xp  = torch.rand(N,  device=DEVICE, dtype=torch.float32) + 0.1   # strictly positive
xs  = torch.rand(N,  device=DEVICE, dtype=torch.float32) * 4.0 - 2.0  # in (-2, 2) for sin/cos
x_erf = torch.rand(N, device=DEVICE, dtype=torch.float32) * 2.0 - 1.0  # T.verf is accurate in [-1, 1]
y   = torch.randn(N, device=DEVICE, dtype=torch.float32)
yp  = torch.rand(N,  device=DEVICE, dtype=torch.float32) + 0.1

# fp16 inputs
x16  = torch.randn(N, device=DEVICE, dtype=torch.float16)
xp16 = torch.rand(N,  device=DEVICE, dtype=torch.float16).to(torch.float16) + 0.1
y16  = torch.randn(N, device=DEVICE, dtype=torch.float16)

# int32 inputs (for vmul / vdiv / vpow etc.)
xi32 = torch.randint(-100, 100, (N,), device=DEVICE, dtype=torch.int32)
yi32 = torch.randint(1,   10,   (N,), device=DEVICE, dtype=torch.int32)   # positive for div/pow


# ===========================================================================
# Binary ops — T.vXXX(A, B, C)
# Supported dtypes per op (from hardware docs):
#   vadd/vsub/vmax : fp16, fp32
#   vmul           : fp16, fp32, int16, int32, int64
#   vdiv           : fp16, fp32, int64
#   vmin           : fp16, fp32, bf16, int16, int32, int64
#   vpow           : int32 only
#   vand           : int8, int64, fp16, fp32, bool
# ===========================================================================
print("\n========== Binary ops (tensor OP tensor) ==========")

_check("add   fp32  x+y",       lambda x, y: x + y,              x,   y)
_check("add   fp16  x+y",       lambda x, y: x + y,              x16, y16)
_check("sub   fp32  x-y",       lambda x, y: x - y,              x,   y)
_check("sub   fp16  x-y",       lambda x, y: x - y,              x16, y16)
_check("mul   fp32  x*y",       lambda x, y: x * y,              x,   y)
_check("mul   fp16  x*y",       lambda x, y: x * y,              x16, y16)
_check("mul   int32 x*y",       lambda x, y: x * y,              xi32, yi32)
_check("truediv fp32 x/yp",     lambda x, y: x / y,              x,   yp)
_check("truediv fp16 x/yp",     lambda x, y: x / y,              x16, xp16)
_check("maximum fp32 max(x,y)", lambda x, y: torch.maximum(x, y),x,   y)
_check("maximum fp16 max(x,y)", lambda x, y: torch.maximum(x, y),x16, y16)
_check("minimum fp32 min(x,y)", lambda x, y: torch.minimum(x, y),x,   y)
_check("minimum fp16 min(x,y)", lambda x, y: torch.minimum(x, y),x16, y16)
_check("pow   int32 xi**yi",    lambda x, y: torch.pow(x, y),    xi32, yi32)

print("\n========== Binary ops (tensor OP scalar constant) ==========")

_check("add   fp32  x+2.0",     lambda x: x + 2.0,               x)
_check("add   fp16  x+2.0",     lambda x: x + 2.0,               x16)
_check("sub   fp32  x-0.5",     lambda x: x - 0.5,               x)
_check("mul   fp32  x*3.0",     lambda x: x * 3.0,               x)
_check("mul   int32 x*3",       lambda x: x * 3,                  xi32)
_check("truediv fp32 x/4.0",   lambda x: x / 4.0,               x)
_check("pow   int32 x**2",      lambda x: torch.pow(x, 2),        xi32)


# ===========================================================================
# Unary ops — T.vXXX(A, B)
# All support fp16 and fp32; vabs also supports uint8, int32, int64.
# ===========================================================================
print("\n========== Unary ops ==========")

_check("exp     fp32  exp(x)",    lambda x: torch.exp(x),          x)
_check("exp     fp16  exp(x)",    lambda x: torch.exp(x),          x16)
_check("log     fp32  log(xp)",   lambda x: torch.log(x),          xp)
_check("log     fp16  log(xp)",   lambda x: torch.log(x),          xp16)
_check("exp2    fp32  exp2(x)",   lambda x: torch.exp2(x),         x)
_check("log2    fp32  log2(xp)",  lambda x: torch.log2(x),         xp)
_check("relu    fp32  relu(x)",   lambda x: torch.relu(x),         x)
_check("relu    fp16  relu(x)",   lambda x: torch.relu(x),         x16)
_check("sigmoid fp32  sig(x)",    lambda x: torch.sigmoid(x),      x)
_check("sigmoid fp16  sig(x)",    lambda x: torch.sigmoid(x),      x16)
_check("sqrt    fp32  sqrt(xp)",  lambda x: torch.sqrt(x),         xp)
_check("rsqrt   fp32  rsqrt(xp)", lambda x: torch.rsqrt(x),        xp)
_check("abs     fp32  abs(x)",    lambda x: torch.abs(x),          x)
_check("abs     fp16  abs(x)",    lambda x: torch.abs(x),          x16)
_check("abs     int32 abs(x)",    lambda x: torch.abs(x),          xi32)
_check("cos     fp32  cos(x)",    lambda x: torch.cos(x),          xs)
_check("sin     fp32  sin(x)",    lambda x: torch.sin(x),          xs)
_check("erf     fp32  erf(x)",    lambda x: torch.erf(x),          x_erf)
_check("tanh    fp32  tanh(x)",   lambda x: torch.tanh(x),         x)
_check("neg     fp32  -x",        lambda x: -x,                    x)
_check("neg     fp16  -x",        lambda x: -x,                    x16)
_check("neg     int32 -x",        lambda x: -x,                    xi32)


# ===========================================================================
# Fused chains — multiple ops in one kernel
# ===========================================================================
print("\n========== Fused chains (fp32) ==========")

_check("silu      x*sigmoid(x)",
       lambda x: x * torch.sigmoid(x),                             x)

_check("gelu_tanh",
       lambda x: x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
       x)

_check("scale_bias  x*w+b",
       lambda x, y, z: x * y + z,                                  x, y, xp)

_check("exp_scale  exp(x)*2-1",
       lambda x: torch.exp(x) * 2.0 - 1.0,                        x)

_check("sqrt_relu  relu(sqrt(xp))",
       lambda x: torch.relu(torch.sqrt(x)),                        xp)

_check("log_sigmoid  log(sigmoid(x))",
       lambda x: torch.log(torch.sigmoid(x)),                      x)

_check("abs_sqrt  sqrt(abs(x)+eps)",
       lambda x: torch.sqrt(torch.abs(x) + 1e-6),                  x)

_check("neg_exp  exp(-x)",
       lambda x: torch.exp(-x),                                     x)

_check("cos_sin_add  cos(x)+sin(x)",
       lambda x: torch.cos(x) + torch.sin(x),                      xs)

_check("tanh_scale  tanh(x*0.5)*2",
       lambda x: torch.tanh(x * 0.5) * 2.0,                        x)

print("\n========== Fused chains (fp16) ==========")

_check("silu fp16",
       lambda x: x * torch.sigmoid(x),                             x16)

_check("scale_shift fp16  x*3+1",
       lambda x: x * 3.0 + 1.0,                                    x16)

_check("exp_add fp16  exp(x)+x",
       lambda x: torch.exp(x) + x,                                 x16)

print("\n========== Fused chains (int32) ==========")
# int32 supported ops: vmul, vabs, vneg, vmin, vpow
# NOT supported: vadd, vsub, vmax, vdiv (fp16/fp32 only for those)

_check("abs_mul int32  abs(x)*2",
       lambda x: torch.abs(x) * 2,                                  xi32)

_check("neg_mul int32  (-x)*y",
       lambda x, y: (-x) * y,                                       xi32, yi32)

_check("pow_mul int32  x**2 * y",
       lambda x, y: torch.pow(x, 2) * y,                           xi32, yi32)

_check("abs_min int32  min(abs(x),y)",
       lambda x, y: torch.minimum(torch.abs(x), y),                xi32, yi32)

_check("mul_mul int32  x*y*2",
       lambda x, y: x * y * 2,                                      xi32, yi32)


# ===========================================================================
# 2-D tensors
# ===========================================================================
print("\n========== 2-D tensors ==========")

x2  = torch.randn(32, 128, device=DEVICE, dtype=torch.float32)
y2  = torch.randn(32, 128, device=DEVICE, dtype=torch.float32)
x2h = torch.randn(32, 128, device=DEVICE, dtype=torch.float16)

_check("add 2D fp32",       lambda x, y: x + y,             x2,  y2)
_check("sigmoid 2D fp32",   lambda x: torch.sigmoid(x),     x2)
_check("silu 2D fp32",      lambda x: x * torch.sigmoid(x), x2)
_check("add 2D fp16",       lambda x, y: x + y,             x2h, x2h)
_check("relu 2D fp16",      lambda x: torch.relu(x),        x2h)


# ===========================================================================
# 3-D tensors — e.g. (batch, seq, hidden)
# TileLang flattens all dims into one contiguous xnumel; the kernel shape is
# the same as 1-D/2-D — only the total element count matters.
# ===========================================================================
print("\n========== 3-D tensors ==========")

x3  = torch.randn(4, 32, 128, device=DEVICE, dtype=torch.float32)
y3  = torch.randn(4, 32, 128, device=DEVICE, dtype=torch.float32)
xp3 = torch.rand(4, 32, 128,  device=DEVICE, dtype=torch.float32) + 0.1
x3h = torch.randn(4, 32, 128, device=DEVICE, dtype=torch.float16)

_check("add 3D fp32",           lambda x, y: x + y,                   x3,  y3)
_check("sub 3D fp32",           lambda x, y: x - y,                   x3,  y3)
_check("mul 3D fp32",           lambda x, y: x * y,                   x3,  y3)
_check("relu 3D fp32",          lambda x: torch.relu(x),              x3)
_check("sigmoid 3D fp32",       lambda x: torch.sigmoid(x),           x3)
_check("tanh 3D fp32",          lambda x: torch.tanh(x),              x3)
_check("exp 3D fp32",           lambda x: torch.exp(x),               x3)
_check("sqrt 3D fp32",          lambda x: torch.sqrt(x),              xp3)
_check("silu 3D fp32",          lambda x: x * torch.sigmoid(x),       x3)
_check("add 3D fp16",           lambda x, y: x + y,                   x3h, x3h)
_check("relu 3D fp16",          lambda x: torch.relu(x),              x3h)
_check("scale_bias 3D fp32",    lambda x, y, z: x * y + z,            x3, y3, xp3)


# ===========================================================================
# 4-D tensors — e.g. (batch, heads, seq, head_dim)
# ===========================================================================
print("\n========== 4-D tensors ==========")

x4  = torch.randn(2, 4, 32, 64, device=DEVICE, dtype=torch.float32)
y4  = torch.randn(2, 4, 32, 64, device=DEVICE, dtype=torch.float32)
xp4 = torch.rand(2, 4, 32, 64,  device=DEVICE, dtype=torch.float32) + 0.1
x4h = torch.randn(2, 4, 32, 64, device=DEVICE, dtype=torch.float16)

_check("add 4D fp32",           lambda x, y: x + y,                   x4,  y4)
_check("mul 4D fp32",           lambda x, y: x * y,                   x4,  y4)
_check("relu 4D fp32",          lambda x: torch.relu(x),              x4)
_check("sigmoid 4D fp32",       lambda x: torch.sigmoid(x),           x4)
_check("tanh 4D fp32",          lambda x: torch.tanh(x),              x4)
_check("exp 4D fp32",           lambda x: torch.exp(x),               x4)
_check("sqrt 4D fp32",          lambda x: torch.sqrt(x),              xp4)
_check("silu 4D fp32",          lambda x: x * torch.sigmoid(x),       x4)
_check("neg_exp 4D fp32",       lambda x: torch.exp(-x),              x4)
_check("add 4D fp16",           lambda x, y: x + y,                   x4h, x4h)
_check("sigmoid 4D fp16",       lambda x: torch.sigmoid(x),           x4h)
_check("scale_bias 4D fp32",    lambda x, y, z: x * y + z,            x4, y4, xp4)


# ===========================================================================
# Reductions — persistent (rnumel fits in SRAM)
#
# The inductor TileLang backend prints the generated @T.prim_func source to
# stdout when it compiles a reduction kernel, so each test below will also
# show the raw TileLang code that was produced.
#
# Layout:
#   Grid  = xnumel  (one AI Core per output element)
#   Input = T.alloc_shared((N,), dtype)       loaded via T.copy(ptr[cid*N], buf, size=[N])
#   Reduce= T.reduce(src, dst, dims=0, reduce_mode='sum'|'max'|…, clear=True)
#   Output= T.alloc_fragment((1,), dtype)     stored via T.copy(buf, ptr[cid], size=[1])
# ===========================================================================
print("\n========== Reductions (persistent, fp32) ==========")

# # Small N so the full row fits in L1 (< 4 KB for fp32)
M, N = 64, 128
xr  = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
xr16 = torch.randn(M, N, device=DEVICE, dtype=torch.float16)

_check("sum  2D fp32  dim=1 keepdim",
       lambda x: x.sum(dim=1, keepdim=True),   xr)

_check("sum  2D fp16  dim=1 keepdim",
       lambda x: x.sum(dim=1, keepdim=True),   xr16)

_check("sum  2D fp32  dim=1 no keepdim",
       lambda x: x.sum(dim=1),                  xr)

_check("max  2D fp32  dim=1 keepdim",
       lambda x: x.max(dim=1, keepdim=True).values,  xr)

_check("min  2D fp32  dim=1 keepdim",
       lambda x: x.min(dim=1, keepdim=True).values,  xr)

_check("mean 2D fp32  dim=1 keepdim",
       lambda x: x.mean(dim=1, keepdim=True),   xr)

print("\n========== Reductions (persistent, larger N) ==========")

M2, N2 = 32, 256
xr2 = torch.randn(M2, N2, device=DEVICE, dtype=torch.float16)

_check("sum  32x512 fp32  dim=1",
       lambda x: x.sum(dim=1, keepdim=True),   xr2)


# ===========================================================================
# GEMM / matmul — T.gemm path
#
# Supported input dtype : fp16  (accumulates in fp32)
# Tile sizes             : block_M=128, block_N=128, block_K=64
#
# TileLang is selected as the matmul backend when
#   TORCHINDUCTOR_NPU_BACKEND=tilelang  (already set at the top of this file)
#   and mat1.dtype is fp16 or int8.
#
# The codegen path:
#   add_tilelang_gemm_choices()  →  TileLangGemmCaller.output_node()
#   →  TileLangScheduling.codegen_template()  →  codegen_tilelang_mm_src()
#   →  tilelang.compile(..., target='npuir')
# ===========================================================================
print("\n========== GEMM — aten.mm fp16 (square) ==========")

# _check casts ref to out.dtype for fair comparison (T.gemm returns fp32)
for _sz in [256, 512, 1024]:
    _A = torch.randn(_sz, _sz, device=DEVICE, dtype=torch.float16)
    _B = torch.randn(_sz, _sz, device=DEVICE, dtype=torch.float16)
    _check(f"mm fp16 {_sz}x{_sz}@{_sz}x{_sz}",
           lambda a, b: torch.mm(a, b), _A, _B)

print("\n========== GEMM — aten.mm fp16 (non-square) ==========")

for _M, _N, _K in [(256, 512, 128), (128, 1024, 256), (512, 128, 256)]:
    _A = torch.randn(_M, _K, device=DEVICE, dtype=torch.float16)
    _B = torch.randn(_K, _N, device=DEVICE, dtype=torch.float16)
    _check(f"mm fp16 {_M}x{_K}@{_K}x{_N}",
           lambda a, b: torch.mm(a, b), _A, _B)

print("\n========== GEMM — fused mm + elementwise (fp16) ==========")

_M, _N, _K = 512, 512, 256
_A = torch.randn(_M, _K, device=DEVICE, dtype=torch.float16)
_B = torch.randn(_K, _N, device=DEVICE, dtype=torch.float16)

_check("mm+relu    fp16",
       lambda a, b: torch.relu(torch.mm(a, b)),    _A, _B)
_check("mm+sigmoid fp16",
       lambda a, b: torch.sigmoid(torch.mm(a, b)), _A, _B)
_check("mm+scale   fp16  out*0.5+1.0",
       lambda a, b: torch.mm(a, b) * 0.5 + 1.0,   _A, _B)

print("\n========== GEMM — torch.nn.Linear (fp16, no bias) ==========")

for _in_f, _out_f, _batch in [(256, 512, 128), (512, 1024, 64)]:
    _linear = torch.nn.Linear(_in_f, _out_f, bias=False).half().to(DEVICE)
    _x = torch.randn(_batch, _in_f, device=DEVICE, dtype=torch.float16)
    _check(f"Linear no-bias {_batch}x{_in_f}→{_batch}x{_out_f}",
           _linear, _x)


# ===========================================================================
# Summary
# ===========================================================================
total = PASS_COUNT + FAIL_COUNT
print(f"\n{'='*55}")
print(f"  PASSED: {PASS_COUNT}/{total}   FAILED: {FAIL_COUNT}/{total}")
print(f"{'='*55}")
