"""Emit Tables 3 (LOOO per-organ) and 4 (POOLED summary) for both HEST-1k and STImage-1K4M."""
import os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
SEEDS = [1, 2, 3]

HEST_ORGANS  = [("Breast",["breast"]), ("Colorectal",["colorectal"]), ("Kidney",["kidney"]),
                ("Liver",["liver"]), ("Lung",["lung"]), ("Pancreas",["pancreas"]),
                ("Prostate",["prostate"]), ("Skin",["skin"])]
STI_ORGANS   = ["breast","kidney","liver","pancreas","skin","prostate","brain","heart"]
STI_LABELS   = {"breast":"Breast","kidney":"Kidney","liver":"Liver","pancreas":"Pancreas",
                "skin":"Skin","prostate":"Prostate","brain":"Brain","heart":"Heart"}
POOLED_FOLDS = [0, 1, 2, 3, 4]

# (name, hest_dir, stimage_dir, hest_tag, stimage_tag)
BASELINES = [
    ("ST-Net",    "results_corrected_crossorgan_baselines", "results_corrected_stimage_crossorgan_baselines", "stnet",     "stnet"),
    ("HisToGene", "results_corrected_crossorgan_baselines", "results_corrected_stimage_crossorgan_baselines", "histogene", "histogene"),
    ("Hist2ST",   "results_corrected_crossorgan_baselines", "results_corrected_stimage_crossorgan_baselines", "hist2st",   "hist2st"),
    ("TRIPLEX",   "results_corrected_crossorgan_baselines", "results_corrected_stimage_crossorgan_baselines", "triplex",   "triplex"),
    ("Gene-DML",  "results_corrected_crossorgan_baselines", "results_corrected_stimage_crossorgan_baselines", "genedml",   "genedml"),
    ("HyperST",   "results_corrected_crossorgan_baselines", "results_corrected_stimage_crossorgan_baselines", "hyperst",   "hyperst"),
    ("MorphoST",  "MorphoST/results_corrected_factorial",   "MorphoST/results_corrected_stimage",            "C111",      "V3"),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def hest_looo_organ(dp, tag, organ_cohs):
    """Mean ± std over seeds of the mean over cohorts in one organ group."""
    ps = []
    for s in SEEDS:
        cv = []
        for coh in organ_cohs:
            f = os.path.join(ROOT, dp, f"LOOO_{tag}_seed{s}", f"fold_{coh}_results.json")
            try: cv.append(json.load(open(f))["pearson_mean"])
            except: pass
        if len(cv) == len(organ_cohs): ps.append(np.mean(cv))
    return (np.mean(ps), np.std(ps)) if ps else None

def sti_looo_organ(dp, tag, organ, fallback_dp="results_corrected_stimage_baselines"):
    """Check dp first, fall back to fallback_dp (nova LOOO-only run) if needed."""
    ps = []
    for s in SEEDS:
        f = os.path.join(ROOT, dp, f"LOOO_{tag}_seed{s}", f"fold_{organ}_results.json")
        if not os.path.isfile(f):
            f = os.path.join(ROOT, fallback_dp, f"LOOO_{tag}_seed{s}", f"fold_{organ}_results.json")
        try: ps.append(json.load(open(f))["pearson_mean"])
        except: pass
    return (np.mean(ps), np.std(ps)) if ps else None

def pooled_stat(dp, tag, fld="pearson_mean"):
    """Mean ± std over seeds of the mean over 5 folds."""
    ps = []
    for s in SEEDS:
        fvs = []
        for k in POOLED_FOLDS:
            f = os.path.join(ROOT, dp, f"POOLED_{tag}_seed{s}", f"fold_{k}_results.json")
            try: fvs.append(json.load(open(f))[fld])
            except: pass
        if fvs: ps.append(np.mean(fvs))
    return (np.mean(ps), np.std(ps)) if ps else None

def fmt(st):
    return f"${st[0]:.3f}_{{{st[1]:.2f}}}$" if st else r"\running"

def best_mark(s, is_best): return f"\\best{{{s}}}" if is_best else s

# ── TABLE 3: LOOO per-organ ───────────────────────────────────────────────────

print("%%% TABLE 3: LOOO per-organ (HEST-1k then STImage-1K4M) %%%")

# HEST section
hest_colavg = {m: [] for m, *_ in BASELINES}
print("% --- HEST-1k LOOO ---")
for oname, cohs in HEST_ORGANS:
    cells = {m: hest_looo_organ(hd, ht, cohs) for m, hd, _, ht, _ in BASELINES}
    avail = {m: v for m, v in cells.items() if v}
    best = max(avail, key=lambda k: avail[k][0]) if avail else None
    row = f"{oname:11s}"
    for m, *_ in BASELINES:
        s = best_mark(fmt(cells[m]), m == best and cells[m])
        row += f" & {s}"
        if cells[m]: hest_colavg[m].append(cells[m][0])
    print(row + r" \\")
# Average
ha = {m: (np.mean(v) if v else None) for m, v in hest_colavg.items()}
ba = max(ha, key=lambda k: ha[k] if ha[k] else -1)
ar = r"\textbf{Average}"
for m, *_ in BASELINES:
    s = f"${ha[m]:.3f}$" if ha[m] is not None else r"\running"
    s = best_mark(s, m == ba and ha[m] is not None)
    ar += f" & {s}"
print(ar + r" \\")

print("% --- STImage-1K4M LOOO ---")
sti_colavg = {m: [] for m, *_ in BASELINES}
for organ in STI_ORGANS:
    cells = {m: sti_looo_organ(sd, st, organ) for m, _, sd, _, st in BASELINES}
    avail = {m: v for m, v in cells.items() if v}
    best = max(avail, key=lambda k: avail[k][0]) if avail else None
    row = f"{STI_LABELS[organ]:11s}"
    for m, *_ in BASELINES:
        s = best_mark(fmt(cells[m]), m == best and cells[m])
        row += f" & {s}"
        if cells[m]: sti_colavg[m].append(cells[m][0])
    print(row + r" \\")
sa = {m: (np.mean(v) if v else None) for m, v in sti_colavg.items()}
bs = max(sa, key=lambda k: sa[k] if sa[k] else -1)
ar = r"\textbf{Average}"
for m, *_ in BASELINES:
    s = f"${sa[m]:.3f}$" if sa[m] is not None else r"\running"
    s = best_mark(s, m == bs and sa[m] is not None)
    ar += f" & {s}"
print(ar + r" \\")

# ── TABLE 4: POOLED summary ───────────────────────────────────────────────────

print("\n%%% TABLE 4: POOLED summary (HEST-1k | STImage-1K4M) %%%")
hest_pool = {m: pooled_stat(hd, ht) for m, hd, _, ht, _ in BASELINES}
sti_pool  = {m: pooled_stat(sd, st) for m, _, sd, _, st in BASELINES}
bh = max(hest_pool, key=lambda k: hest_pool[k][0] if hest_pool[k] else -1)
bs2 = max(sti_pool, key=lambda k: sti_pool[k][0] if sti_pool[k] else -1)
for m, *_ in BASELINES:
    sh = fmt(hest_pool[m]); ss = fmt(sti_pool[m])
    if m == bh and hest_pool[m]: sh = f"\\best{{{sh}}}"
    if m == bs2 and sti_pool[m]: ss = f"\\best{{{ss}}}"
    print(f"{m:11s} & {sh} & {ss} \\\\")
