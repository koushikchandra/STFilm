"""8-ORGAN grouped view of the per-cohort HEST results. Same underlying runs as aggregate_hest.py,
but the 10 cohorts are collapsed to 8 organ groups (COAD+READ -> colorectal, IDC+LYMPH_IDC -> breast),
matching our cross-organ organ grouping. For each method/organ we average the constituent cohorts'
k-fold means WITHIN each seed, then report mean+/-std over the 3 seeds.
"""
import os, json, glob
import numpy as np

HERE = os.path.dirname(__file__)
MORPHO_ROOT = os.path.join(HERE, "results_hest")
BASE_ROOT = os.path.join(HERE, "..", "results_hest_baselines")
STFLOW_ROOT = os.path.join(HERE, "..", "results_hest_stflow")
SEEDS = [1, 2, 3]

ORGANS = [  # display order: (organ label, constituent cohorts)
    ("Breast",     ["IDC", "LYMPH_IDC"]),
    ("Prostate",   ["PRAD"]),
    ("Pancreas",   ["PAAD"]),
    ("Skin",       ["SKCM"]),
    ("Colorectal", ["COAD", "READ"]),
    ("Kidney",     ["CCRCC"]),
    ("Liver",      ["HCC"]),
    ("Lung",       ["LUNG"]),
]
BASELINES = ["histogene", "hist2st", "stnet", "deepspace", "mlpprobe", "triplex", "bleep"]
methods = ["MorphoST-V3", "STFlow"] + BASELINES
roots = {"MorphoST-V3": (MORPHO_ROOT, "V3"), "STFlow": (STFLOW_ROOT, "STFlow")}
for b in BASELINES:
    roots[b] = (BASE_ROOT, b)
short = {"MorphoST-V3": "Morpho-V3", "STFlow": "STFlow", "histogene": "HisToGene", "hist2st": "Hist2ST",
         "stnet": "STNet", "deepspace": "DeepSpaCE", "mlpprobe": "MLPprobe", "triplex": "TRIPLEX",
         "bleep": "BLEEP"}


def cohort_seed_mean(root, cohort, tag, seed):
    if tag == "STFlow":
        # STFlow's train.py nests results under a timestamped exp dir; take the latest run.
        base = os.path.join(root, f"{cohort}_STFlow_seed{seed}")
        fs = glob.glob(f"{base}/*/{cohort}/results_kfold.json") + glob.glob(f"{base}/*/results_kfold.json")
        if not fs:
            return None
        fs.sort(key=os.path.getmtime)
        try:
            return json.load(open(fs[-1]))["pearson_mean"]
        except Exception:
            return None
    f = os.path.join(root, f"{cohort}_{tag}_seed{seed}", "results_kfold.json")
    if os.path.isfile(f):
        try:
            return json.load(open(f))["pearson_mean"]
        except Exception:
            return None
    return None


def organ_cell(root, cohorts, tag):
    """Per seed: mean over constituent cohorts (needs all present that seed). Then mean+/-std/seeds."""
    per_seed = []
    for s in SEEDS:
        vs = [cohort_seed_mean(root, c, tag, s) for c in cohorts]
        if all(v is not None for v in vs):
            per_seed.append(np.mean(vs))
    if not per_seed:
        return float("nan"), float("nan"), 0
    return float(np.mean(per_seed)), float(np.std(per_seed)), len(per_seed)


colw = 11
hdr = f"{'organ':11s} |" + "".join(f"{short[m]:>{colw}s}" for m in methods) + "  | win"
print(hdr); print("-" * len(hdr))
col_means = {m: [] for m in methods}
for organ, cohorts in ORGANS:
    vals, cells = {}, {}
    for m in methods:
        root, tag = roots[m]
        mu, sd, n = organ_cell(root, cohorts, tag)
        vals[m] = mu if mu == mu else -1
        if mu == mu:
            col_means[m].append(mu)
            cells[m] = f"{mu:.3f}" + (f"±{sd:.2f}" if n > 1 else f"~{n}")
        else:
            cells[m] = "-"
    win = max(vals, key=vals.get) if any(v > -1 for v in vals.values()) else "-"
    line = f"{organ:11s} |"
    for m in methods:
        mark = "*" if m == win and vals[m] > -1 else " "
        line += f"{cells[m]+mark:>{colw}s}"
    print(line + f"  | {short.get(win, win)}")

print("-" * len(hdr))
avg = {m: (np.mean(col_means[m]) if col_means[m] else -1) for m in methods}
win = max(avg, key=avg.get)
line = f"{'Average':11s} |"
for m in methods:
    s = f"{avg[m]:.3f}" if avg[m] > -1 else "-"
    line += f"{s + ('*' if m == win else ' '):>{colw}s}"
print(line + f"  | {short.get(win, win)}")
print(f"\n(8-organ grouping: Breast=IDC+LYMPH_IDC, Colorectal=COAD+READ; others 1:1. "
      f"Average is over the 8 organ rows.)")
