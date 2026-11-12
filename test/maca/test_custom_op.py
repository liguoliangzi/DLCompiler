"""
Triton add kernel using dl.custom() with bitcode auto-resolution on MACA.

Verifies that the custom op `add` produces correct results on the MetaX
(MACA) backend. The kernel is lowered through:

    hivm.custom (dl.custom frontend op)
      -> triton_gpu.custom  (TritonToTritonGPUCustom pass, carries layouts)
      -> tt.extern_elementwise (LowerTritonGPUCustom pass, calls user symbol)
      -> linked user bitcode (maca_add.bc, via metadata["bitcodes"])
"""

import pytest
import torch
import triton
import triton.language as tl

import triton.backends.dicp_triton.driver as dicp_driver

triton.runtime.driver.set_active(dicp_driver.DICPDriver("maca"))

import triton.language.extra.deeplink.cann.extension as dl  # noqa: E402


# ======================================================================
# DSL custom op registration — bitcode auto-resolved by name
# ======================================================================


@dl.register_custom_op
class add:
    core = dl.CORE.MXU
    pipe = dl.PIPE.MX_COMPUTE
    mode = dl.MODE.MX_SIMD

    def __init__(self, a, b, out=None):
        assert out is not None, "dl.custom() requires out= parameter"
        self.symbol = "custom_add_" + str(a.dtype)
        # Auto-resolved to maca_add.bc (see dlcompiler/bitcode/src/maca_add.cc)
        self.bitcode = "maca_add"


# ======================================================================
# Triton kernel
# ======================================================================


@triton.jit
def custom_add_kernel(output_ptr, a_ptr, b_ptr, L: tl.constexpr):
    idx = tl.arange(0, L)
    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)
    buf = tl.full([L], 0, a.dtype)
    res = dl.custom("add", a, b, out=buf)
    tl.store(output_ptr + idx, res)


# ======================================================================
# Tests
# ======================================================================


@pytest.mark.parametrize("L", [32, 128, 1024])
def test_custom_add_fp32(L):
    a = torch.randn(L, device="cuda")
    b = torch.randn(L, device="cuda")
    out = torch.empty(L, device="cuda")

    custom_add_kernel[1, 1, 1](out, a, b, L=L)

    ref = a + b
    assert torch.allclose(
        out, ref, atol=1e-6
    ), f"L={L}: max_diff={(out - ref).abs().max().item()}"


@pytest.mark.parametrize("L", [32, 128, 1024])
def test_custom_add_int32(L):
    a = torch.randint(0, 1000, (L,), dtype=torch.int32, device="cuda")
    b = torch.randint(0, 1000, (L,), dtype=torch.int32, device="cuda")
    out = torch.empty(L, dtype=torch.int32, device="cuda")

    custom_add_kernel[1, 1, 1](out, a, b, L=L)

    ref = a + b
    assert torch.equal(out, ref), f"L={L}: out={out}, ref={ref}"


if __name__ == "__main__":
    for L in [32, 128, 1024]:
        test_custom_add_fp32(L)
        test_custom_add_int32(L)
        print(f"[PASS] L={L}")
    print("Done.")
