"""Aggregate corrected (nested-validation) results into the paper's table numbers.
Metric per (cohort,seed) = mean over that seed's fold_*_results.json of a field; organ = mean over
member cohorts; report mean_std over seeds. Prints per-organ PCC (Table 1), 3-metric summary (Table 2),
cross-organ (Table 3), and coverage so we know which cells are real vs still running."""
import os, glob, json, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
COHORTS = ["CCRCC","COAD","READ","HCC","IDC","LYMPH_IDC","LUNG","PAAD","PRAD","SKCM"]
ORGANS = [("Breast",["IDC","LYMPH_IDC"]),("Prostate",["PRAD"]),("Pancreas",["PAAD"]),
          ("Skin",["SKCM"]),("Colorectal",["COAD","READ"]),("Kidney",["CCRCC"]),
          ("Liver",["HCC"]),("Lung",["LUNG"])]
SEEDS = [1,2,3]

def cohort_seed(dirpath, cohort, tag, seed, field):
    d = os.path.join(dirpath, f"{cohort}_{tag}_seed{seed}")
    fs = glob.glob(os.path.join(d, "fold_*_results.json"))
    vals=[]
    for f in fs:
        try: vals.append(json.load(open(f))[field])
        except Exception: pass
    return float(np.mean(vals)) if vals else None

def organ_stat(dirpath, tag, cohorts, field):
    """mean_std over seeds of the organ value (organ = mean over member cohorts, seeds present)."""
    per_seed=[]
    for s in SEEDS:
        cv=[cohort_seed(dirpath,c,tag,s,field) for c in cohorts]
        cv=[x for x in cv if x is not None]
        if len(cv)==len(cohorts):   # organ complete for this seed
            per_seed.append(np.mean(cv))
    if not per_seed: return None
    return (float(np.mean(per_seed)), float(np.std(per_seed)), len(per_seed))

# intra methods: label -> (dir, tag)
INTRA = {
 "ST-Net":("results_corrected_hest_baselines","stnet"),
 "HisToGene":("results_corrected_hest_baselines","histogene"),
 "Hist2ST":("results_corrected_hest_baselines","hist2st"),
 "BLEEP":("results_corrected_hest_baselines","bleep"),
 "TRIPLEX":("results_corrected_hest_baselines","triplex"),
 "STFlow":(None,None),  # corrected reproduction pending
 "MorphoST":("MorphoST/results_corrected_hest_corr","V3"),
}
def resolve(d): return os.path.join(ROOT,d) if d and d.startswith("MorphoST") else (os.path.join(ROOT,d) if d else None)

print("="*70); print("TABLE 1 — intra per-organ PCC (mean±sd, n seeds)"); print("="*70)
hdr="organ      "+"".join(f"{m:>12s}" for m in INTRA); print(hdr)
for oname,cs in ORGANS:
    row=f"{oname:11s}"
    for m,(d,tag) in INTRA.items():
        if d is None: row+=f"{'run':>12s}"; continue
        st=organ_stat(resolve(d),tag,cs,"pearson_mean")
        row+= f"{('%.3f/%d'%(st[0],st[2])) if st else 'run':>12s}"
    print(row)

print("\n"+"="*70); print("TABLE 2 — intra 3-metric averages over organs (PCC/SCC/MSE)"); print("="*70)
for m,(d,tag) in INTRA.items():
    if d is None: print(f"{m:12s}  PCC=run  SCC=run  MSE=run"); continue
    out={}
    for fld in ["pearson_mean","spearman_mean","mse"]:
        vals=[organ_stat(resolve(d),tag,cs,fld) for _,cs in ORGANS]
        vals=[v[0] for v in vals if v]
        out[fld]=(np.mean(vals), len(vals)) if vals else None
    def f(k): return f"{out[k][0]:.3f}(o{out[k][1]})" if out[k] else "run"
    print(f"{m:12s}  PCC={f('pearson_mean'):>10s}  SCC={f('spearman_mean'):>10s}  MSE={f('mse'):>10s}")

print("\n"+"="*70); print("ABLATION — factorial (cross-organ LOOO), Pearson per component"); print("="*70)
FACT=os.path.join(ROOT,"MorphoST","results_corrected_factorial")
for code in ["000","001","010","011","100","101","110","111"]:
    # subdir naming: look for *C{code}* or components in results_kfold
    ds=glob.glob(os.path.join(FACT,f"*{code}*seed*"))
    per_seed=[]
    for s in SEEDS:
        cand=[d for d in ds if d.endswith(f"seed{s}")]
        if not cand: continue
        fs=glob.glob(os.path.join(cand[0],"fold_*_results.json"))
        v=[json.load(open(f))["pearson_mean"] for f in fs if os.path.isfile(f)]
        if v: per_seed.append(np.mean(v))
    if per_seed:
        print(f"  C{code}: {np.mean(per_seed):.4f}±{np.std(per_seed):.4f}  (n={len(per_seed)}, folds/seed varies)")
    else:
        print(f"  C{code}: (none) dirs={[os.path.basename(d) for d in ds][:3]}")

print("\n"+"="*70); print("STImage (cross-organ), MorphoST V3"); print("="*70)
STI=os.path.join(ROOT,"MorphoST","results_corrected_stimage")
for reg in ["LOOO","POOLED"]:
    per_seed=[]
    for s in SEEDS:
        d=glob.glob(os.path.join(STI,f"{reg}_*seed{s}"))
        if not d: continue
        fs=glob.glob(os.path.join(d[0],"fold_*_results.json"))
        v=[json.load(open(f))["pearson_mean"] for f in fs if os.path.isfile(f)]
        if v: per_seed.append(np.mean(v))
    print(f"  {reg}: {('%.4f±%.4f (n=%d)'%(np.mean(per_seed),np.std(per_seed),len(per_seed))) if per_seed else 'run'}")

print("\n"+"="*70); print("LOSS CONTROL — MorphoST MSE vs MSE+PCC (avg over complete organs)"); print("="*70)
for label,d,tag in [("MorphoST MSE","MorphoST/results_corrected_hest_mse","V3"),
                    ("MorphoST MSE+PCC","MorphoST/results_corrected_hest_corr","V3")]:
    vals=[organ_stat(resolve(d),tag,cs,"pearson_mean") for _,cs in ORGANS]
    vals=[v[0] for v in vals if v]
    print(f"  {label:18s}: {np.mean(vals):.4f} (over {len(vals)} organs)")
print("\nfactorial dir sample:", [os.path.basename(x) for x in glob.glob(os.path.join(FACT,'*'))[:4]])
