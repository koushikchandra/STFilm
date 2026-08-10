"""Phase 3: seed-ensemble. For each film/regime/fold, average the best-epoch predictions
across seeds 1-3 (dumped as fold_<f>_preds.npz by --dump_preds), then recompute the metric.
Reports single-seed-mean vs 3-seed-ensemble, raw and (LOOO) detection-filtered.
Run AFTER results_ens_uni8 is populated by ens_sweep.sbatch."""
import os, glob, json, csv
import numpy as np
import scanpy as sc, warnings; warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
ENS  = os.path.join(ROOT, "results_ens_uni8")
FILMS = ["none", "desc", "local"]
SEEDS = (1, 2, 3)
LOOO = ["kidney","liver","lung","pancreas","prostate","skin","breast","colorectal"]
POOLED = ["0","1","2","3","4"]
MIN_DETECT = 0.10
ORGAN_COHORT = {'kidney':'CCRCC','prostate':'PRAD','liver':'HCC','lung':'LUNG',
                'skin':'SKCM','colorectal':'COAD','breast':'IDC','pancreas':'PAAD'}

def per_gene_pearson(pred, gt):
    """Column-wise Pearson r between pred and gt (n_spots x n_genes)."""
    out = []
    for j in range(gt.shape[1]):
        a, b = pred[:, j], gt[:, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out)

def load_seed(film, regime, seed, fold):
    p = os.path.join(ENS, f"{regime}_{film}_seed{seed}", f"fold_{fold}_preds.npz")
    if not os.path.exists(p): return None
    d = np.load(p, allow_pickle=True)
    return d["pred"], d["gt"], list(d["genes"])

def detect_keep(fold):
    """Detection-rate keep-mask (panel order) for a LOOO organ fold; None for POOLED."""
    if fold not in ORGAN_COHORT: return None
    panel = json.load(open(f"{ROOT}/cross_organ_splits8/LOOO/genes_{fold}.json"))
    panel = panel.get("genes", panel) if isinstance(panel, dict) else panel
    rows = list(csv.DictReader(open(f"{ROOT}/cross_organ_splits8/LOOO/splits/test_{fold}.csv")))
    counts = np.zeros(len(panel)); tot = 0
    for r in rows:
        pth = os.path.join(ROOT, "dataset", r["expr_path"])
        if not os.path.exists(pth): continue
        a = sc.read_h5ad(pth)
        gp = [g for g in panel if g in a.var_names]
        X = a[:, gp].X; X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        nz = (X > 0).sum(0)
        for g, c in zip(gp, nz): counts[panel.index(g)] += c
        tot += X.shape[0]
    return panel, (counts / max(tot, 1)) >= MIN_DETECT

for regime, folds in [("LOOO", LOOO), ("POOLED", POOLED)]:
    print(f"\n===== {regime} =====")
    hdr = f"{'film':6s} {'single(mean sd)':>16s} {'ensemble(3sd)':>14s}"
    if regime == "LOOO": hdr += f" {'ens+filter':>11s}"
    print(hdr)
    for film in FILMS:
        single_fold, ens_fold, ensf_fold = [], [], []
        for fold in folds:
            seeds_data = [load_seed(film, regime, s, fold) for s in SEEDS]
            seeds_data = [x for x in seeds_data if x is not None]
            if not seeds_data: continue
            gt = seeds_data[0][1]
            # single-seed mean of per-seed metric
            per_seed = [np.nanmean(per_gene_pearson(p, g)) for p, g, _ in seeds_data]
            single_fold.append(np.mean(per_seed))
            # ensemble = mean of predictions across seeds
            ens_pred = np.mean([p for p, _, _ in seeds_data], axis=0)
            pg = per_gene_pearson(ens_pred, gt)
            ens_fold.append(np.nanmean(pg))
            if regime == "LOOO":
                keep = detect_keep(fold)
                if keep is not None and len(keep[1]) == len(pg):
                    ensf_fold.append(np.nanmean(pg[keep[1]]))
        if not ens_fold:
            print(f"{film:6s}  (no dumps yet)"); continue
        line = f"{film:6s} {np.mean(single_fold):16.4f} {np.mean(ens_fold):14.4f}"
        if regime == "LOOO" and ensf_fold: line += f" {np.mean(ensf_fold):11.4f}"
        print(line)
