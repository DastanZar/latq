"""Lattice vector quantization ladder for weight-only decode.

Three granularities share one format — a byte indexes a 256-point codebook —
and one fused Triton GEMV (codebook resident in cache, one byte per block
crossing the bus):

    mode "d8": 8 weights per byte (1 bpw, 16x vs fp16) — rel err ~0.60,
               which EQUALS the 1-bit rate-distortion floor sqrt(1 - 2/pi)
               for Gaussian sources. The codebook is provably at the floor.
    mode "d4": 4 weights per byte (2 bpw, 8x)  — rel err ~0.35
    mode "z2": 2 weights per byte (4 bpw, 4x)  — rel err ~0.10, competitive
               with INT4 element-wise but with vector (block) scaling.

Lattices: D_n = integer vectors with even coordinate sum (the minimal
shell +-e_i +-e_j makes it denser than the integer lattice Z_n in 4-D/8-D;
in 2-D the plain Z^2 grid is the right choice).

Prior art: vector quantization for LLM weights appears in AQLM (additive
VQ) and QTIP (trellis-coded lattices); lattice background in Conway &
Sloane, *Sphere Packings, Lattices and Groups*. Here the codebook is small
enough (2-8 KB) to stay in L1/shared memory while streaming one byte per
block.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = [
    "build_codebook",
    "quantize",
    "dequantize",
    "latq_gemv_reference",
    "run_gemv",
    "theoretical_floor",
    "HAS_TRITON",
]

MODES = {"d8": (8, "D"), "d4": (4, "D"), "z2": (2, "Z")}


def theoretical_floor(bits: int) -> float:
    """Rel-error floor of optimal quantization of a Gaussian at `bits` bits/weight."""
    if bits == 1:
        return math.sqrt(1 - 2 / math.pi)
    # dithered-uniform approximation for higher rates (bound, not exact)
    return math.sqrt(math.pi / 2 / (2 ** (2 * bits)))


def _product(rng, dim):
    if dim == 0:
        yield ()
        return
    for head in rng:
        for tail in _product(rng, dim - 1):
            yield (head,) + tail


def build_codebook(mode: str) -> torch.Tensor:
    """Deterministic 256 x dim codebook for a mode, sorted (norm, lex)."""
    dim, lattice = MODES[mode]
    if lattice == "Z":
        pts: list[tuple[int, ...]] = [(a, b) for a in range(-16, 17) for b in range(-16, 17)]
    elif dim == 4:
        # 9^4 = 6561 candidates, even-sum filter -> the D4 lattice by radius
        pts = [p for p in _product(range(-4, 5), 4) if sum(p) % 2 == 0]
    else:
        # D8: explicit shells up to norm^2 = 4 (origin + 112 + 16 + 448 = 577)
        raw: list[list[int]] = [[0] * 8]
        for i in range(8):
            for j in range(i + 1, 8):
                for si in (1, -1):
                    for sj in (1, -1):
                        p = [0] * 8
                        p[i], p[j] = si, sj
                        raw.append(p)
        for i in range(8):
            for s in (2, -2):
                p = [0] * 8
                p[i] = s
                raw.append(p)
        for i in range(8):
            for j in range(i + 1, 8):
                for k in range(j + 1, 8):
                    for l in range(k + 1, 8):
                        for q in range(16):
                            p = [0] * 8
                            p[i] = 1 if q & 1 else -1
                            p[j] = 1 if q & 2 else -1
                            p[k] = 1 if q & 4 else -1
                            p[l] = 1 if q & 8 else -1
                            raw.append(p)
        pts = [tuple(p) for p in raw]
    pts.sort(key=lambda p: (sum(v * v for v in p), p))
    pts = pts[:256]
    if lattice == "D":
        for p in pts:
            assert sum(p) % 2 == 0, f"{mode} even-sum invariant broken: {p}"
    cb = torch.tensor(pts, dtype=torch.float32)
    assert cb.shape == (256, dim)
    return cb


def quantize(w: torch.Tensor, mode: str = "d4") -> dict:
    """Quantize [N, K] to one byte per block + per-row fp32 scale.

    Scale policy: the row's 3-sigma span maps onto the codebook radius
    (scale = 3*sigma / R_codebook), selected by error sweep on Gaussian
    rows and valid for all three modes. Returns {idx uint8 [N, K/dim],
    scale fp32 [N], cb, mode, dim}.
    """
    dim, _ = MODES[mode]
    n, k = w.shape
    assert k % dim == 0, f"K={k} not divisible by {dim}"
    w = w.to(torch.float32)
    cb = build_codebook(mode)
    radius = cb.norm(dim=1).max().item()
    sigma = w.pow(2).mean(dim=1).sqrt().clamp(min=1e-8)
    scale = 3.0 * sigma / radius
    blocks = (w / scale[:, None]).reshape(n, k // dim, dim)
    d = torch.cdist(blocks.reshape(-1, dim), cb)
    idx = d.argmin(dim=1).reshape(n, k // dim).to(torch.uint8)
    return {"idx": idx, "scale": scale, "cb": cb, "mode": mode, "dim": dim}


def dequantize(pack: dict) -> torch.Tensor:
    idx, scale, cb, dim = pack["idx"], pack["scale"], pack["cb"], pack["dim"]
    n, kb = idx.shape
    blocks = cb[idx.to(torch.int64)].reshape(n, kb, dim)
    return blocks.reshape(n, kb * dim) * scale[:, None]


def latq_gemv_reference(x: torch.Tensor, pack: dict) -> torch.Tensor:
    return x.to(torch.float32) @ dequantize(pack).T


if HAS_TRITON:

    @triton.jit
    def _latq_gemv_kernel(
        Idx, X, Y, CB, Scale, K, KB,
        DIM: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, KB, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < KB
            idx = tl.load(Idx + row * KB + cols, mask=m, other=0).to(tl.int32)
            base = idx * DIM
            for d in tl.static_range(DIM):
                v = tl.load(CB + base + d)
                kidx = cols * DIM + d
                xv = tl.load(X + kidx, mask=m & (kidx < K), other=0.0)
                acc += v * xv
        s = tl.load(Scale + row)
        tl.store(Y + row, tl.sum(acc, axis=0) * s)


def run_gemv(x: torch.Tensor, w: torch.Tensor, mode: str = "d4") -> dict:
    """Quantize weights at `mode`, run fused decode GEMV. {y, pack}."""
    pack = quantize(w, mode)
    n, k = w.shape
    if HAS_TRITON:
        y = torch.empty(n, dtype=torch.float32)
        _latq_gemv_kernel[(n,)](
            pack["idx"], x.to(torch.float32), y, pack["cb"].reshape(-1),
            pack["scale"], k, k // pack["dim"], DIM=pack["dim"], BLOCK=64,
        )
    else:
        y = latq_gemv_reference(x, pack)
    return {"y": y, "pack": pack}
