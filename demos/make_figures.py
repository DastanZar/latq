"""latq figures: codebook glyphs + fidelity ladder. Computed from latq itself."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latq import dequantize, quantize, theoretical_floor  # noqa: E402
from latq.lattice import build_codebook  # noqa: E402

FIG = os.path.join(ROOT, "assets")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140})

# Fig 1: the D8 codebook, projected to its two densest coord planes, colored by shell
cb = build_codebook("d8")
sq = cb.square().sum(dim=1)
fig, ax = plt.subplots(figsize=(5.6, 5.2))
shells = {0.0: ("shell 0 — the origin", "#222", 60), 2.0: ("norm²=2 — 112 minimal vectors", "#4472c4", 26),
          4.0: ("norm²=4 — 143 next shell", "#c0504d", 14)}
for s, (lbl, col, ms) in shells.items():
    m = (sq - s).abs() < 1e-4
    ax.scatter(cb[m, 0], cb[m, 1], s=ms, c=col, label=f"{lbl} ({int(m.sum())} pts)", alpha=.8)
ax.axhline(0, lw=.5, c="k"); ax.axvline(0, lw=.5, c="k")
ax.set_xlabel("coordinate x₁"); ax.set_ylabel("coordinate x₂")
ax.set_title("D8: 256 codes = 256 eight-weight blocks\n(planes x₁/x₂ and x₃/x₄ overlaid)")
ax.legend(fontsize=7.5, loc="upper left"); ax.set_aspect("equal"); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(f"{FIG}/d8_glyphs.png"); plt.close(fig)

# Fig 2: fidelity ladder vs the floor — measured error, floor curve overlaid
torch.manual_seed(0)
w = torch.randn(128, 4096)
fig, ax = plt.subplots(figsize=(6.8, 4.4))
bps, errs = [], []
for mode, bpw in (("d8", 1), ("d4", 2), ("z2", 4)):
    e = ((dequantize(quantize(w, mode)) - w).norm() / w.norm()).item()
    bps.append(bpw); errs.append(e)
xs = [2 ** b for b in range(1, 7)]
ax.plot(xs, [theoretical_floor(b) for b in range(1, 7)], "k--",
        lw=1, label="Gaussian R-D bound (rate-dist.)")
ax.plot([2, 4, 16], errs, "o", ms=11, mfc="none", mew=2, color="#4472c4",
        label="latq lattice modes")
for (x, y), nm in zip(zip([2, 4, 16], errs), ("D8 · 8 wts/byte", "D4 · 4 wts/byte", "Z2 · 2 wts/byte")):
    ax.annotate(nm, (x, y), textcoords="offset points", xytext=(10, 6), fontsize=8.5)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([2, 4, 16]); ax.set_xticklabels([2, 4, 16])
ax.set_xlabel("weights per byte"); ax.set_ylabel("relative reconstruction error")
ax.set_title("One byte, many granularities — and where the theory limit is")
ax.legend(fontsize=8); ax.grid(alpha=.2, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/ladder_floor.png"); plt.close(fig)

# Fig 3: quantization noise map — which part of the plane does D8 cover best?
cb4 = build_codebook("d4")
fig, ax = plt.subplots(figsize=(5.0, 4.6))
pts = torch.rand(20000, 4) * 8 - 4
d = torch.cdist(pts, cb4)
idx = d.argmin(dim=1)
cell = d.min(dim=1).values
sc = ax.scatter(pts[:, 0], pts[:, 1], c=cell, s=1.4, cmap="viridis_r", vmin=0, vmax=0.75)
m = cb4.norm(dim=1) < 3.6
ax.scatter(cb4[m, 0], cb4[m, 1], marker="x", s=22, c="white", lw=.8, label="D4 codes (proj x₁,x₂)")
plt.colorbar(sc, ax=ax, label="distance to nearest code (quant. noise)")
ax.legend(fontsize=7.5); ax.grid(alpha=.15)
ax.set_title("Voronoi noise field of the 256-point D4 codebook")
fig.tight_layout(); fig.savefig(f"{FIG}/d4_noise_field.png"); plt.close(fig)
print("latq figures ->", sorted(os.listdir(FIG)))
