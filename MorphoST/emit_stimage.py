"""Emit Table 2 (tab:stimage): STImage-1K4M LOOO per-organ Pearson comparison."""
import os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
SEEDS = [1, 2, 3]
ORGANS = ["breast", "kidney", "liver", "pancreas", "skin", "prostate", "brain", "heart"]
ORGAN_LABELS = {
    "breast": "Breast", "kidney": "Kidney", "liver": "Liver",
    "pancreas": "Pancreas", "skin": "Skin", "prostate": "Prostate",
    "brain": "Brain", "heart": "Heart",
}
METHODS = [
    ("ST-Net",    "results_corrected_stimage_baselines", "LOOO", "stnet"),
    ("HisToGene", "results_corrected_stimage_baselines", "LOOO", "histogene"),
    ("Hist2ST",   "results_corrected_stimage_baselines", "LOOO", "hist2st"),
    ("TRIPLEX",   "results_corrected_stimage_baselines", "LOOO", "triplex"),
    ("Gene-DML",  "results_corrected_stimage_baselines", "LOOO", "genedml"),
    ("HyperST",   "results_corrected_stimage_baselines", "LOOO", "hyperst"),
    ("MorphoST",  "MorphoST/results_corrected_stimage",  "LOOO", "V3"),
]

def organ_stat(d, regime, tag, organ):
    ps = []
    for s in SEEDS:
        f = os.path.join(ROOT, d, f"{regime}_{tag}_seed{s}", f"fold_{organ}_results.json")
        try: ps.append(json.load(open(f))["pearson_mean"])
        except Exception: pass
    return (np.mean(ps), np.std(ps)) if ps else None

print("%%% TABLE 2 (STImage-1K4M LOOO per-organ Pearson) %%%")
colavg = {m: [] for m, _, _, _ in METHODS}
for organ in ORGANS:
    cells = {m: organ_stat(d, r, t, organ) for m, d, r, t in METHODS}
    avail = {m: v for m, v in cells.items() if v}
    best = max(avail, key=lambda k: avail[k][0]) if avail else None
    row = f"{ORGAN_LABELS[organ]:9s}"
    for m, _, _, _ in METHODS:
        st = cells[m]
        s = f"${st[0]:.3f}_{{{st[1]:.2f}}}$" if st else r"\running"
        if m == best and st: s = f"\\best{{{s}}}"
        row += f" & {s}"
        if st: colavg[m].append(st[0])
    row += r" \\"
    print(row)

# Average row
avgs = {m: (np.mean(v) if v else None) for m, v in colavg.items()}
best_avg = max(avgs, key=lambda k: avgs[k] if avgs[k] is not None else -1)
ar = r"\textbf{Average}"
for m, _, _, _ in METHODS:
    a = avgs[m]
    s = f"${a:.3f}$" if a is not None else r"\running"
    if m == best_avg and a is not None: s = f"\\best{{{s}}}"
    ar += f" & {s}"
ar += r" \\"
print(ar)
