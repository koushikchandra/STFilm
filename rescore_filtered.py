"""Variance/detection-filtered LOOO re-scoring (leakage-safe: filter is on the TARGET's
own distribution, never used for training). For each LOOO organ fold we compute per-gene
detection rate (fraction of held-out spots with count>0); genes detected in < MIN_DETECT of
spots are dropped, and pearson_mean is recomputed over surviving genes only. Pure
re-aggregation of saved per-gene pearson_corrs (aligned 1:1 to the fold gene panel) — no
model re-run. Emits a raw-vs-filtered LOOO table across all models + baselines."""
import json, glob, csv, os, numpy as np
import scanpy as sc, warnings; warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
MIN_DETECT = 0.10   # keep genes expressed in >=10% of held-out spots

ORGAN_COHORT = {'kidney':'CCRCC','prostate':'PRAD','liver':'HCC','lung':'LUNG',
                'skin':'SKCM','colorectal':'COAD','breast':'IDC','pancreas':'PAAD'}
ORGANS = list(ORGAN_COHORT)

MODELS = [
    ("ST-Net",              "results_spatial_uni8", lambda s: f"LOOO_stnet_seed{s}",     (1,2,3)),
    ("HisToGene",           "results_spatial_uni8", lambda s: f"LOOO_histogene_seed{s}", (1,2,3)),
    ("Hist2ST",             "results_spatial_uni8", lambda s: f"LOOO_hist2st_seed{s}",   (1,2,3)),
    ("BLEEP",               "results_spatial_uni8", lambda s: f"LOOO_bleep_seed{s}",     (1,2,3)),
    ("TRIPLEX",             "results_spatial_uni8", lambda s: f"LOOO_triplex_seed{s}",   (1,2,3)),
    ("STFlow none",         "results_final_uni8",   lambda s: f"LOOO_none_seed{s}",       (1,2,3)),
    ("STFiLM desc",         "results_final_uni8",   lambda s: f"LOOO_desc_seed{s}",       (1,2,3)),
    ("STFiLM local",        "results_local_uni8",   lambda s: f"LOOO_local_seed{s}",      (1,2,3)),
    ("STFlow MoE",          "results_moe_uni8",     lambda s: f"LOOO_moe_seed{s}",        (1,2,3)),
]

def panel_for(organ):
    d = json.load(open(f"{ROOT}/cross_organ_splits8/LOOO/genes_{organ}.json"))
    if isinstance(d, dict): d = d.get("genes", list(d.keys()))
    return list(d)

def detect_rates(organ):
    """Per-gene detection rate across all held-out spots for this organ (panel order)."""
    panel = panel_for(organ)
    rows = list(csv.DictReader(open(f"{ROOT}/cross_organ_splits8/LOOO/splits/test_{organ}.csv")))
    counts = np.zeros(len(panel)); tot = 0
    for r in rows:
        p = os.path.join(ROOT, "dataset", r["expr_path"])
        if not os.path.exists(p): continue
        a = sc.read_h5ad(p)
        idx = [panel.index(g) for g in panel if g in a.var_names]
        gp  = [g for g in panel if g in a.var_names]
        X = a[:, gp].X; X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        nz = (X > 0).sum(0)
        for j, g in zip(idx, range(len(gp))): counts[j] += nz[g]
        tot += X.shape[0]
    return panel, counts / max(tot, 1)

# precompute keep-mask per organ
KEEP = {}
for o in ORGANS:
    panel, det = detect_rates(o)
    KEEP[o] = (panel, det >= MIN_DETECT, det)

def fold_pearsons(path, organ, panel):
    f = os.path.join(path, f"fold_{organ}_results.json")
    if not os.path.exists(f): return None
    pc = json.load(open(f))["pearson_corrs"]   # list of {'name','pearson_corr'}
    m = {e["name"]: e["pearson_corr"] for e in pc}
    return np.array([float(m[g]) if g in m and m[g] is not None else np.nan
                     for g in panel], dtype=float)

print(f"Detection filter: keep genes expressed in >= {MIN_DETECT:.0%} of held-out spots\n")
print("Genes kept per organ:")
for o in ORGANS:
    panel, keep, det = KEEP[o]
    print(f"  {o:11s} {keep.sum():2d}/{len(panel):2d} kept  (dropped: "
          f"{[g for g,k in zip(panel,keep) if not k]})")

print(f"\n{'model':15s} {'LOOO raw':>9s} {'LOOO filt':>10s}   per-organ filtered (kept-gene mean)")
for label, root, tagfn, seeds in MODELS:
    raw_folds, filt_folds = {}, {}
    for o in ORGANS:
        panel, keep, _ = KEEP[o]
        raws, filts = [], []
        for s in seeds:
            pc = fold_pearsons(os.path.join(ROOT, root, tagfn(s)), o, panel)
            if pc is None or len(pc) != len(keep): continue
            if np.isnan(pc).all(): continue
            raws.append(np.nanmean(pc))
            filts.append(np.nanmean(pc[keep]) if keep.any() else np.nan)
        if raws:  raw_folds[o]  = np.mean(raws)
        if filts: filt_folds[o] = np.nanmean(filts)
    raw_ov  = np.mean(list(raw_folds.values()))  if raw_folds  else np.nan
    filt_ov = np.mean(list(filt_folds.values())) if filt_folds else np.nan
    cells = " ".join(f"{o[:3]}={filt_folds.get(o,float('nan')):.2f}" for o in ORGANS)
    print(f"{label:15s} {raw_ov:9.4f} {filt_ov:10.4f}   {cells}")
