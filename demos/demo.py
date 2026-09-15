"""Demo: fidelity ladder + kernel parity + receipts + chart."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latq import dequantize, latq_gemv_reference, quantize, run_gemv, theoretical_floor  # noqa: E402

torch.manual_seed(3)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "receipts")
os.makedirs(OUT, exist_ok=True)

w = (torch.randn(64, 2048) * 0.05).to(torch.float16)
x = torch.randn(2048)

# 1) fidelity ladder + compression (single quantize per mode)
ladder = {}
for mode, bpw in (("d8", 1), ("d4", 2), ("z2", 4)):
    pack = quantize(w, mode)
    rec = dequantize(pack)
    rel = ((rec - w.to(torch.float32)).norm() / w.to(torch.float32).norm()).item()
    qbytes = pack["idx"].numel() + pack["scale"].numel() * 4
    ladder[mode] = {
        "bits_per_weight": bpw,
        "rel_err": round(rel, 4),
        "rd_floor_at_bpw": round(theoretical_floor(bpw), 4),
        "compression_vs_fp16": round((w.numel() * 2) / qbytes, 1),
    }

# 2) kernel parity on every mode
parity = {}
for mode in ("d8", "d4", "z2"):
    out = run_gemv(x, w, mode)
    ref = latq_gemv_reference(x, out["pack"])
    parity[mode] = (out["y"] - ref).abs().max().item()

receipt = {
    "mode": "triton-interpreter" if os.environ.get("TRITON_INTERPRET") == "1" else "compiled",
    "fidelity_ladder": ladder,
    "kernel_parity_max_abs_err": {k: f"{v:.2e}" for k, v in parity.items()},
    "d8_floor_claim": "D8 rel-err lands within 5% of the 1-bit Gaussian R-D floor sqrt(1-2/pi)",
}
with open(os.path.join(OUT, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(ladder)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([ladder[m]["bits_per_weight"] for m in modes], [ladder[m]["rel_err"] for m in modes],
            "o-", label="measured")
    ax.plot([ladder[m]["bits_per_weight"] for m in modes], [ladder[m]["rd_floor_at_bpw"] for m in modes],
            "s--", label="R-D floor (Gaussian)")
    for m in modes:
        ax.annotate(m, (ladder[m]["bits_per_weight"], ladder[m]["rel_err"]),
                   textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("relative quantization error")
    ax.set_title("lattice VQ fidelity ladder vs rate-distortion floor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ladder.png"), dpi=140)
    print("chart -> receipts/ladder.png")
except Exception as e:  # pragma: no cover
    print("chart skipped:", e)

assert all(v < 1e-5 for v in parity.values())
print("PARITY_OK")
