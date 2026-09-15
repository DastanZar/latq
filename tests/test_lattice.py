import math

import torch

from latq import (
    HAS_TRITON,
    build_codebook,
    dequantize,
    latq_gemv_reference,
    quantize,
    run_gemv,
    theoretical_floor,
)
from latq.lattice import MODES

torch.manual_seed(0)


def test_codebook_shapes_deterministic():
    for mode, (dim, _) in MODES.items():
        cb = build_codebook(mode)
        assert cb.shape == (256, dim)
        assert torch.equal(cb, build_codebook(mode)), f"{mode} not deterministic"


def test_d_lattice_invariant():
    for mode in ("d4", "d8"):
        cb = build_codebook(mode)
        sums = cb.sum(dim=1)
        assert torch.equal(sums % 2, torch.zeros_like(sums)), f"{mode} even-sum broken"


def test_d8_at_rate_distortion_floor():
    """1 bpw on Gaussian rows must land ON the 1-bit R-D floor (not above it)."""
    w = torch.randn(32, 2048)
    rec = dequantize(quantize(w, "d8"))
    rel = ((rec - w).norm() / w.norm()).item()
    floor = theoretical_floor(1)  # sqrt(1 - 2/pi) ~ 0.602
    assert rel < floor * 1.05, f"rel {rel:.3f} above floor {floor:.3f} — codebook suboptimal"
    assert rel > floor * 0.85, f"rel {rel:.3f} suspiciously BELOW floor — format leak?"


def test_fidelity_ladder_monotone():
    """More bytes per block must monotonically reduce quantization error."""
    w = torch.randn(32, 1024)
    rels = []
    for mode in ("d8", "d4", "z2"):
        rec = dequantize(quantize(w, mode))
        rels.append(((rec - w).norm() / w.norm()).item())
    assert rels[0] > rels[1] > rels[2], f"ladder not monotone: {rels}"


def test_compression_ratios():
    w = torch.randn(64, 2048)
    for mode, bpw in (("d8", 1), ("d4", 2), ("z2", 4)):
        pack = quantize(w, mode)
        qbytes = pack["idx"].numel() + pack["scale"].numel() * 4
        eff_bpw = qbytes * 8 / w.numel()
        assert eff_bpw <= bpw * 1.15, f"{mode}: {eff_bpw:.2f} bpw exceeds budget"


def test_gemv_kernel_parity_all_modes():
    if not HAS_TRITON:
        import pytest

        pytest.skip("triton unavailable")
    for mode in ("d8", "d4", "z2"):
        w = (torch.randn(24, 512) * 0.05).to(torch.float16)
        x = torch.randn(512)
        out = run_gemv(x, w, mode)
        ref = latq_gemv_reference(x, out["pack"])
        err = (out["y"] - ref).abs().max().item()
        assert err < 1e-5, f"{mode}: kernel vs reference {err:.2e}"


def test_z2_competitive_with_int4():
    """Z2 (4 bpw vector VQ) should be competitive with INT4 element-wise."""
    w = torch.randn(32, 1024)
    x = torch.randn(1024)
    y_vq = run_gemv(x, w, "z2")["y"]
    y16 = x @ w.T
    rel_vq = ((y_vq - y16).norm() / y16.norm()).item()
    # INT4 baseline: 16 uniform levels, per-row absmax scale
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.round(w / s).clamp(-8, 7) * s
    y_i4 = x @ q.T
    rel_i4 = ((y_i4 - y16).norm() / y16.norm()).item()
    assert rel_vq <= rel_i4 * 1.35, f"z2 rel {rel_vq:.3f} vs int4 {rel_i4:.3f}"


def test_theoretical_floor_value():
    assert abs(theoretical_floor(1) - math.sqrt(1 - 2 / math.pi)) < 1e-12
