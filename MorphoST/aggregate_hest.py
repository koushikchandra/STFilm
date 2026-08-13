"""STFlow-Table-1-style per-cohort table centered on MorphoST V3 (HEST-1k within-organ k-fold).

Two comparison sets, all 3-seed, same UNI features / splits / metric:
  (1) FEATURE-MATCHED (apples-to-apples, the ones we can rigorously claim to beat): our own
      histogene/hist2st/stnet/deepspace/mlpprobe/triplex/bleep run under the SAME per-cohort protocol
      (results_hest_baselines/), plus MorphoST-V3 (results_hest/).
  (2) PUBLISHED reference: STFlow / TRIPLEX / UNI / BLEEP numbers from the STFlow paper Table 1.

MorphoST V3 cell = mean+/-std over 3 seeds of the cohort's k-fold mean. Bold-equivalent '*' marks the
row winner among the feature-matched columns.
"""
import os, json
import numpy as np

HERE = os.path.dirname(__file__)
MORPHO_ROOT = os.path.join(HERE, "results_hest")
BASE_ROOT = os.path.join(HERE, "..", "results_hest_baselines")
STFLOW_ROOT = os.path.join(HERE, "..", "results_hest_stflow")
SEEDS = [1, 2, 3]
COHORTS = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "HCC", "LUNG", "LYMPH_IDC"]
LABEL = {"LYMPH_IDC": "LYMPH"}
BASELINES = ["histogene", "hist2st", "stnet", "deepspace", "mlpprobe", "triplex", "bleep"]

# STFlow paper Table 1, HEST block averages (reference only).
PAPER_AVG = {"UNI": 0.344, "BLEEP": 0.368, "TRIPLEX": 0.395, "STFlow": 0.415}


def cell(root, cohort, model_tag):
    """(mean, std, nseeds) over the per-seed kfold means for one method/cohort."""
    import glob
    vals = []
    for s in SEEDS:
        if model_tag == "STFlow":  # STFlow nests results under a timestamped exp dir
            base = os.path.join(root, f"{cohort}_STFlow_seed{s}")
            fs = glob.glob(f"{base}/*/{cohort}/results_kfold.json") + glob.glob(f"{base}/*/results_kfold.json")
            if not fs:
                continue
            fs.sort(key=os.path.getmtime); f = fs[-1]
        else:
            f = os.path.join(root, f"{cohort}_{model_tag}_seed{s}", "results_kfold.json")
        if os.path.isfile(f):
            try:
                vals.append(json.load(open(f))["pearson_mean"])
            except Exception:
                pass
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


methods = ["MorphoST-V3", "STFlow"] + BASELINES
roots = {"MorphoST-V3": (MORPHO_ROOT, "V3"), "STFlow": (STFLOW_ROOT, "STFlow")}
for b in BASELINES:
    roots[b] = (BASE_ROOT, b)

# header
short = {"MorphoST-V3": "Morpho-V3", "STFlow": "STFlow", "histogene": "HisToGene", "hist2st": "Hist2ST",
         "stnet": "STNet", "deepspace": "DeepSpaCE", "mlpprobe": "MLPprobe",
         "triplex": "TRIPLEX", "bleep": "BLEEP"}
colw = 11
hdr = f"{'cohort':8s} |" + "".join(f"{short[m]:>{colw}s}" for m in methods) + "  | win"
print(hdr); print("-" * len(hdr))

col_means = {m: [] for m in methods}
for c in COHORTS:
    vals, cells = {}, {}
    for m in methods:
        root, tag = roots[m]
        mu, sd, n = cell(root, c, tag)
        vals[m] = mu if mu == mu else -1
        if mu == mu:
            col_means[m].append(mu)
            cells[m] = f"{mu:.3f}" + (f"±{sd:.2f}" if n > 1 else f"~{n}")
        else:
            cells[m] = "-"
    win = max(vals, key=vals.get) if any(v > -1 for v in vals.values()) else "-"
    line = f"{LABEL.get(c, c):8s} |"
    for m in methods:
        mark = "*" if m == win and vals[m] > -1 else " "
        line += f"{cells[m]+mark:>{colw}s}"
    print(line + f"  | {short.get(win, win)}")

print("-" * len(hdr))
avg = {m: (np.mean(col_means[m]) if col_means[m] else -1) for m in methods}
win = max(avg, key=avg.get)
line = f"{'Average':8s} |"
for m in methods:
    s = f"{avg[m]:.3f}" if avg[m] > -1 else "-"
    line += f"{s + ('*' if m == win else ' '):>{colw}s}"
print(line + f"  | {short.get(win, win)}")
print(f"\nPublished ref (STFlow Table 1 HEST avg):  STFlow {PAPER_AVG['STFlow']}  "
      f"TRIPLEX {PAPER_AVG['TRIPLEX']}  BLEEP {PAPER_AVG['BLEEP']}  UNI {PAPER_AVG['UNI']}")

# completeness
tot = len(COHORTS) * len(SEEDS)
print("\ncompleteness (cohort,seed kfolds / %d):" % tot)
for m in methods:
    root, tag = roots[m]
    done = sum(cell(root, c, tag)[2] for c in COHORTS)
    print(f"  {short[m]:10s} {done}/{tot}  (avg over {len(col_means[m])}/10 cohorts)")
