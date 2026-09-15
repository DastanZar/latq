"""latq — lattice vector quantization ladder for weight-only decode."""
import os

if not os.environ.get("LATQ_DISABLE_INTERPRET"):
    import torch as _torch

    if not _torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")

from .lattice import (
    HAS_TRITON,
    MODES,
    build_codebook,
    dequantize,
    latq_gemv_reference,
    quantize,
    run_gemv,
    theoretical_floor,
)

__all__ = [
    "MODES",
    "HAS_TRITON",
    "build_codebook",
    "quantize",
    "dequantize",
    "latq_gemv_reference",
    "run_gemv",
    "theoretical_floor",
]
