"""Emit Table 11 (tab:percohort): full per-cohort HEST-1k Pearson, ungrouped, nested validation.
Reuses the per-cohort/per-seed loader from emit_final.py. STFlow left as \\running (no corrected
reproduction yet)."""
import os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
SEEDS = [1, 2, 3]
COHORTS = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "HCC", "LUNG", "LYMPH_IDC"]
# display label -> (dir relative to repo root, tag)
METHODS = [("ST-Net", "results_corrected_hest_baselines", "stnet"),
           ("HisToGene", "results_corrected_hest_baselines", "histogene"),
           ("Hist2ST", "results_corrected_hest_baselines", "hist2st"),
           ("TRIPLEX", "results_corrected_hest_baselines", "triplex"),
           ("Gene-DML", "results_corrected_hest_baselines", "genedml"),
           ("HyperST", "results_corrected_hest_baselines", "hyperst"),
           ("MorphoST", "MorphoST/results_corrected_hest_corr", "V3")]

def cohort_stat(d, tag, c):
    """mean_std over seeds of per-fold-mean Pearson for one (dir, tag, cohort)."""
    ps = []
    for s in SEEDS:
        v = []
        for f in glob.glob(os.path.join(ROOT, d, f"{c}_{tag}_seed{s}", "fold_*_results.json")):
            try: v.append(json.load(open(f))["pearson_mean"])
            except Exception: pass
        if v: ps.append(np.mean(v))
    return (np.mean(ps), np.std(ps)) if ps else None

print("%%% TABLE 11 (per-cohort, 10 cohorts) %%%")
for c in COHORTS:
    cells = {m: cohort_stat(d, t, c) for m, d, t in METHODS}
    avail = {m: v for m, v in cells.items() if v}
    best = max(avail, key=lambda k: avail[k][0]) if avail else None
    disp = c.replace("_", r"\_")
    row = f"{disp:9s}"
    for m, _, _ in METHODS:   # all baselines + MorphoST, in column order
        st = cells[m]; s = f"${st[0]:.3f}_{{{st[1]:.2f}}}$" if st else r"\running"
        if m == best and st: s = f"\\best{{{s}}}"
        row += f" & {s}"
    row += r" \\"
    print(row)
