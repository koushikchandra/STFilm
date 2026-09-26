"""Compact spatial-domain recovery figure: ARI (left) and NMI (right), KMeans k=6,
predicted vs. measured domains over n=58 held-out six-organ slides (mean over 3 seeds).
One short, narrow figure with two small panels; coral hero bar for MIST.

Values: ARI@6 from the per-method domain-recovery results; NMI@6 from tab:krobust.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "xtick.color": "black", "ytick.color": "black",
    "axes.labelcolor": "black", "text.color": "black",
})

BLUE, CORAL, ANNOT = "#aeb3b8", "#aeb3b8", "#444444"
MODELS = ["ST-Net", "Hist2ST", "BLEEP", "STEM", "MIST"]
ARI = [0.065, 0.073, 0.081, 0.031, 0.091]
NMI = [0.091, 0.087, 0.092, 0.028, 0.110]


def _panel(ax, vals, ylabel, ymax):
    x = np.arange(len(MODELS))
    cols = [BLUE] * (len(MODELS) - 1) + [CORAL]
    bars = ax.bar(x, vals, 0.72, color=cols, edgecolor="none")
    for b, v in zip(bars, vals):
        hero = b is bars[-1]
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if hero else "normal", color=ANNOT if hero else "#444")
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", ls=":", lw=0.5, color="#D5D5D5", zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, length=2)


def main(outdir):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.4, 2.0))
    _panel(axL, ARI, r"ARI $\uparrow$", 0.12)
    _panel(axR, NMI, r"NMI $\uparrow$", 0.13)
    fig.tight_layout(w_pad=1.4)
    out = Path(outdir) / "spatial_recovery.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
