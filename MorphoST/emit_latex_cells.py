import os, glob, json
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,"..")
ORGANS=[("Breast",["IDC","LYMPH_IDC"]),("Prostate",["PRAD"]),("Pancreas",["PAAD"]),
        ("Skin",["SKCM"]),("Colorectal",["COAD","READ"]),("Kidney",["CCRCC"]),
        ("Liver",["HCC"]),("Lung",["LUNG"])]
SEEDS=[1,2,3]
def resolve(d): return os.path.join(ROOT,d)
def cs(dp,c,tag,s,fld):
    fs=glob.glob(os.path.join(dp,f"{c}_{tag}_seed{s}","fold_*_results.json")); v=[]
    for f in fs:
        try:v.append(json.load(open(f))[fld])
        except:pass
    return np.mean(v) if v else None
def organ(dp,tag,cohs,fld):
    ps=[]
    for s in SEEDS:
        cv=[cs(dp,c,tag,s,fld) for c in cohs]; cv=[x for x in cv if x is not None]
        if len(cv)==len(cohs): ps.append(np.mean(cv))
    return (np.mean(ps),np.std(ps),len(ps)) if ps else None
INTRA=[("ST-Net","results_corrected_hest_baselines","stnet"),
 ("HisToGene","results_corrected_hest_baselines","histogene"),
 ("Hist2ST","results_corrected_hest_baselines","hist2st"),
 ("BLEEP","results_corrected_hest_baselines","bleep"),
 ("TRIPLEX","results_corrected_hest_baselines","triplex"),
 ("MorphoST","MorphoST/results_corrected_hest_corr","V3")]
def sub(mu,sd): return f"${mu:.3f}_{{{sd:.2f}}}$"
print("---- TABLE 1 rows (complete organs only) ----")
for oname,cohs in ORGANS:
    cells={}; done=True
    for m,d,tag in INTRA:
        st=organ(resolve(d),tag,cohs,"pearson_mean")
        cells[m]=st
        if st is None: done=False
    if not done: 
        print(f"% {oname}: incomplete -> keep running"); continue
    best=max(cells,key=lambda k:cells[k][0])
    s=f"{oname:11s}"
    for m,_,_ in INTRA:
        mu,sd,n=cells[m]; c=sub(mu,sd)
        if m==best: c=f"\\best{{{c}}}"
        s+=f" & {c}"
    print(s+r" & \running \\   % STFlow pending")
print("\n---- TABLE 2 (3-metric, avg over complete organs) ----")
comp=[ (o,cohs) for o,cohs in ORGANS if organ(resolve("MorphoST/results_corrected_hest_corr"),"V3",cohs,"pearson_mean")]
print("% complete organs:", [o for o,_ in comp])
rows={}
for m,d,tag in INTRA:
    r={}
    for fld in ["pearson_mean","spearman_mean","mse"]:
        vals=[organ(resolve(d),tag,cohs,fld) for _,cohs in comp]; vals=[v[0] for v in vals if v]
        r[fld]=np.mean(vals) if vals else None
    rows[m]=r
bestp=max(rows,key=lambda k:rows[k]["pearson_mean"]); bests=max(rows,key=lambda k:rows[k]["spearman_mean"]); bestm=min(rows,key=lambda k:rows[k]["mse"])
for m,_,_ in INTRA:
    r=rows[m]
    p=f"{r['pearson_mean']:.3f}"; sc=f"{r['spearman_mean']:.3f}"; ms=f"{r['mse']:.3f}"
    if m==bestp:p=f"\\best{{{p}}}"
    if m==bests:sc=f"\\best{{{sc}}}"
    if m==bestm:ms=f"\\best{{{ms}}}"
    print(f"{m:11s} & {p} & {sc} & {ms} \\\\")
print("\n---- STImage rows ----")
STI=resolve("MorphoST/results_corrected_stimage")
for reg in ["LOOO","POOLED"]:
    ps=[]
    for s in SEEDS:
        d=glob.glob(os.path.join(STI,f"{reg}_*seed{s}"))
        if d:
            fs=glob.glob(os.path.join(d[0],"fold_*_results.json")); v=[json.load(open(f))["pearson_mean"] for f in fs]
            if v: ps.append(np.mean(v))
    print(f"STImage {reg}: {sub(np.mean(ps),np.std(ps))} (n={len(ps)})" if ps else f"{reg}: run")
print("\n---- LOSS CONTROL over COMMON organs ----")
common=[cohs for o,cohs in comp]
for label,d,tag in [("MorphoST MSE","MorphoST/results_corrected_hest_mse","V3"),("MorphoST MSE+PCC","MorphoST/results_corrected_hest_corr","V3")]:
    vals=[organ(resolve(d),tag,cohs,"pearson_mean") for cohs in common]; vals=[v[0] for v in vals if v]
    print(f"{label:18s}: {np.mean(vals):.4f} over {len(vals)} common organs")
