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
    "axes.edgecolor": "black", "axes.linewidth": 0.9,
    "xtick.color": "black", "ytick.color": "black",
    "axes.labelcolor": "black", "text.color": "black",
    "hatch.linewidth": 0.6,
})

CORAL, CORAL_E, ANNOT = "#E07B5D", "#B5533A", "#C0552F"
ERRC = "#BBBBBB"  # light-gray error bars

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


# ---- right panel: spatial kNN vs random-k neighbors (full model, UNI+CONCH) ----
NB_GROUPS = ["POOLED", "SKCM"]
NB_MEAN = {"Random-$k$":  [0.577, 0.671], "Spatial $k$NN": [0.605, 0.709]}
NB_STD  = {"Random-$k$":  [0.009, 0.004], "Spatial $k$NN": [0.006, 0.002]}
NB_CFGS = [("Random-$k$", "#aeb3b8"), ("Spatial $k$NN", CORAL)]  # kNN = hero (coral)


def _grid(ax):
    ax.grid(axis="y", ls=":", lw=0.6, color="#D5D5D5", zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10)


# shared geometry so bars and group gaps are identical in both panels
BW = 0.24        # bar width (same in both panels)
SLOT = 0.26      # bar pitch within a group (small gap between bars)
GAP_G = 0.42     # gap between groups (same in both panels)
MARGIN = 0.24    # left/right padding inside each panel


def _centers(nbars, ngroups):
    gw = nbars * SLOT                       # width occupied by one group
    pitch = gw + GAP_G
    c = np.arange(ngroups) * pitch
    xlim = (c[0] - gw / 2 - MARGIN, c[-1] + gw / 2 + MARGIN)
    return c, xlim


def _label(ax, bars, vals, stds, hero):
    for b, v, s in zip(bars, vals, stds):
        ax.text(b.get_x() + b.get_width() / 2, v + s + 0.008,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if hero else "normal",
                color="black", rotation=90)


def main(outdir):
    cL, xlimL = _centers(len(CFGS), len(GROUPS))
    cR, xlimR = _centers(len(NB_CFGS), len(NB_GROUPS))
    spanL, spanR = xlimL[1] - xlimL[0], xlimR[1] - xlimR[0]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 3.1),
                                   gridspec_kw={"width_ratios": [spanL, spanR], "wspace": 0.18})

    # ----- (a) context-stream ablation -----
    for j, (label, col, ec, hh) in enumerate(CFGS):
        hero = "full" in label
        xs = cL + (j - (len(CFGS) - 1) / 2) * SLOT
        bars = axL.bar(xs, MEAN[label], BW, yerr=STD[label], color=col, edgecolor="none",
                       linewidth=0, label=label,
                       error_kw=dict(elinewidth=0.9, capsize=2, ecolor=ERRC))
        _label(axL, bars, MEAN[label], STD[label], hero)
    axL.set_xticks(cL); axL.set_xticklabels(GROUPS, fontsize=10)
    axL.set_ylabel("Mean PCC@50", fontsize=10)
    axL.set_ylim(0.50, 0.80); axL.set_xlim(*xlimL)
    axL.legend(handles=[Patch(facecolor=c, edgecolor="none", label=k) for k, c, e, h in CFGS],
               loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False,
               fontsize=10, handlelength=1.1, columnspacing=0.9, handletextpad=0.4)
    axL.set_title("(a) Context streams", fontsize=10, pad=24)
    _grid(axL)

    # ----- (b) spatial kNN vs random-k -----
    for j, (label, col) in enumerate(NB_CFGS):
        hero = "kNN" in label
        xs = cR + (j - (len(NB_CFGS) - 1) / 2) * SLOT
        bars = axR.bar(xs, NB_MEAN[label], BW, yerr=NB_STD[label], color=col, edgecolor="none",
                       linewidth=0, label=label,
                       error_kw=dict(elinewidth=0.9, capsize=2, ecolor=ERRC))
        _label(axR, bars, NB_MEAN[label], NB_STD[label], hero)
    axR.set_xticks(cR); axR.set_xticklabels(NB_GROUPS, fontsize=10)
    axR.set_ylabel("Mean PCC@50", fontsize=10)
    axR.set_ylim(0.50, 0.80); axR.set_xlim(*xlimR)
    axR.legend(handles=[Patch(facecolor=c, edgecolor="none", label=k) for k, c in NB_CFGS],
               loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
               fontsize=10, handlelength=1.1, columnspacing=0.9, handletextpad=0.4)
    axR.set_title("(b) Neighborhood (UNI+CONCH)", fontsize=10, pad=24)
    _grid(axR)

    out = Path(outdir) / "ablation_bars.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
