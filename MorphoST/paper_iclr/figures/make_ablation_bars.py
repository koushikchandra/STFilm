"""Context-stream ablation bars, styled after encoder_bars.py (coral hero for the full config,
muted gray + hatch for reduced configs, value labels, dotted y-grid, despined).

Global self-attention is the vanilla-Transformer backbone (not our contribution); the added levers
are the coordinate-invariant LOCAL spatial pathway and the SLIDE-pooling pathway. Bars show, on top
of the backbone: Vanilla (G), +Local (G+L), +Slide (G+S), +Local+Slide (full).

Values (mean +/- std over 3 seeds), computed from the ablation result dirs:
  POOLED = 6-organ pooled, UNI+CONCH (LOOO/results_ablation6, complete factorial)
  SKCM / LUNG = intra-cohort, UNI features (results_ablation_intra, complete 2^3 factorial)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.9,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "hatch.linewidth": 0.6,
})

CORAL, CORAL_E, ANNOT = "#E07B5D", "#B5533A", "#C0552F"

GROUPS = ["POOLED\n(UNI+CONCH)", "SKCM\n(UNI)", "LUNG\n(UNI)"]
# config -> per-group (mean, std); order: Vanilla, +Local, +Slide, +Local+Slide (hero)
MEAN = {
    "Vanilla (G)":        [0.590, 0.641, 0.577],
    "+Local (G+L)":       [0.596, 0.693, 0.584],
    "+Slide (G+S)":       [0.570, 0.639, 0.583],
    "+Local+Slide (full)":[0.605, 0.696, 0.589],
}
STD = {
    "Vanilla (G)":        [0.005, 0.008, 0.004],
    "+Local (G+L)":       [0.007, 0.007, 0.004],
    "+Slide (G+S)":       [0.037, 0.021, 0.004],
    "+Local+Slide (full)":[0.006, 0.001, 0.005],
}
# (label, facecolor, edgecolor, hatch) — hero last so coral pops on the right of each group
CFGS = [("Vanilla (G)",         "#8a9096", "#6f757b", "////"),
        ("+Local (G+L)",        "#aeb3b8", "#8f959b", "...."),
        ("+Slide (G+S)",        "#cdd1d5", "#a9aeb3", "\\\\"),
        ("+Local+Slide (full)", CORAL,     CORAL_E,   "")]


def main(outdir):
    x = np.arange(len(GROUPS)); SLOT = 0.21; BW = 0.202  # SLOT>BW leaves a small gap between bars
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for j, (label, col, ec, hh) in enumerate(CFGS):
        hero = "full" in label
        xs = x + (j - 1.5) * SLOT
        bars = ax.bar(xs, MEAN[label], BW, yerr=STD[label], color=col, edgecolor="none",
                      linewidth=0, label=label,
                      error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#666"))
        for b, v in zip(bars, MEAN[label]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.4 if hero else 6.6, fontweight="bold" if hero else "normal",
                    color=ANNOT if hero else "#666", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(GROUPS, fontsize=10)
    ax.set_ylabel("Mean PCC", fontsize=10.5)
    ax.set_ylim(0.50, 0.76)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=k) for k, c, e, h in CFGS],
              loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False,
              fontsize=8.2, handlelength=1.2, columnspacing=1.1)
    ax.grid(axis="y", ls=":", lw=0.6, color="#D5D5D5", zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    out = Path(outdir) / "ablation_bars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
