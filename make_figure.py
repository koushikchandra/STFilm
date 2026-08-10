"""Qualitative figure for the paper: per-organ LOOO Pearson (3-seed mean +/- std) for the
three main conditioners (none / desc / local). Shows where FiLM helps and visualizes the
metric floor (kidney/prostate/liver). Reads fold_<organ>_results.json pearson_mean; no re-run.
Writes paper/figs/per_organ_looo.pdf (+ .png)."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (1, 2, 3)
# (label, dir, tag builder)
METHODS = [
    ("STFlow (none)", "results_final_uni8", lambda s: f"LOOO_none_seed{s}"),
    ("STFiLM (desc)", "results_final_uni8", lambda s: f"LOOO_desc_seed{s}"),
    ("STFiLM (local)", "results_local_uni8", lambda s: f"LOOO_local_seed{s}"),
]
COLORS = ["#9e9e9e", "#4c72b0", "#c44e52"]

def organ_values(root, tagfn):
    """{organ: [pearson_mean over seeds]}"""
    per = {}
    for s in SEEDS:
        d = os.path.join(ROOT, root, tagfn(s))
        for f in glob.glob(os.path.join(d, "fold_*_results.json")):
            organ = os.path.basename(f)[len("fold_"):-len("_results.json")]
            pm = json.load(open(f)).get("pearson_mean")
            if pm is not None:
                per.setdefault(organ, []).append(float(pm))
    return per

data = [organ_values(r, fn) for _, r, fn in METHODS]
# organ order: by 'none' mean ascending, so the floor organs sit on the left
organs = sorted(data[0], key=lambda o: np.mean(data[0][o]))
means = np.array([[np.mean(d[o]) for o in organs] for d in data])
stds  = np.array([[np.std(d[o])  for o in organs] for d in data])

# overall (mean across organs) appended as a summary group
organs_disp = [o.capitalize() for o in organs] + ["Overall"]
means = np.hstack([means, means.mean(axis=1, keepdims=True)])
stds  = np.hstack([stds,  np.zeros((len(METHODS), 1))])

x = np.arange(len(organs_disp))
w = 0.26
fig, ax = plt.subplots(figsize=(9, 3.4))
for i, (label, _, _) in enumerate(METHODS):
    ax.bar(x + (i - 1) * w, means[i], w, yerr=stds[i], capsize=2,
           label=label, color=COLORS[i], edgecolor="black", linewidth=0.4,
           error_kw=dict(lw=0.7))

# shade the metric-floor organs
floor = {"kidney", "prostate", "liver"}
for j, o in enumerate(organs):
    if o in floor:
        ax.axvspan(j - 0.5, j + 0.5, color="0.92", zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(organs_disp, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Pearson (3-seed mean)", fontsize=10)
ax.set_title("Per-organ leave-one-organ-out transfer", fontsize=11)
ax.axvline(len(organs) - 0.5, color="black", lw=0.6, ls=":")
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
ax.set_ylim(0, max(0.9, means.max() * 1.15))
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
ax.text(0.5, 0.02, "shaded = low-variance metric floor (near-silent leakage-safe panels)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="0.35")
fig.tight_layout()

outdir = os.path.join(ROOT, "paper", "figs")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, f"per_organ_looo.{ext}"), dpi=200, bbox_inches="tight")
print("wrote", os.path.join(outdir, "per_organ_looo.pdf"))
# also print the numbers for the caption/sanity
print("\norgan        none    desc   local")
for j, o in enumerate(organs_disp):
    print(f"{o:11s} " + " ".join(f"{means[i,j]:6.3f}" for i in range(len(METHODS))))
