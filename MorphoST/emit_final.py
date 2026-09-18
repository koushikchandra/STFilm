import os, glob, json
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,"..")
ORGANS=[("Breast",["IDC","LYMPH_IDC"]),("Prostate",["PRAD"]),("Pancreas",["PAAD"]),
        ("Skin",["SKCM"]),("Colorectal",["COAD","READ"]),("Kidney",["CCRCC"]),
        ("Liver",["HCC"]),("Lung",["LUNG"])]
SEEDS=[1,2,3]
def resolve(d): return os.path.join(ROOT,d)
def cseed(dp,c,tag,s,fld):
    v=[]
    for f in glob.glob(os.path.join(dp,f"{c}_{tag}_seed{s}","fold_*_results.json")):
        try:v.append(json.load(open(f))[fld])
        except:pass
    return np.mean(v) if v else None
def organ(dp,tag,cohs,fld):
    ps=[]
    for s in SEEDS:
        cv=[cseed(dp,c,tag,s,fld) for c in cohs]; cv=[x for x in cv if x is not None]
        if len(cv)==len(cohs): ps.append(np.mean(cv))
    return (np.mean(ps),np.std(ps),len(ps)) if ps else None
INTRA=[("ST-Net","results_corrected_hest_baselines","stnet"),
 ("HisToGene","results_corrected_hest_baselines","histogene"),
 ("Hist2ST","results_corrected_hest_baselines","hist2st"),
 ("TRIPLEX","results_corrected_hest_baselines","triplex"),
 ("Gene-DML","results_corrected_hest_baselines","genedml"),
 ("HyperST","results_corrected_hest_baselines","hyperst"),
 ("MorphoST","MorphoST/results_corrected_hest_corr","V3")]
BASELINE_ORDER=["ST-Net","HisToGene","Hist2ST","TRIPLEX","Gene-DML","HyperST"]
def sub(st): return f"${st[0]:.3f}_{{{st[1]:.2f}}}$"
# TABLE 1
print("%%% TABLE 1 (8 organs) %%%")
colavg={m:[] for m,_,_ in INTRA}
for oname,cohs in ORGANS:
    cells={m:organ(resolve(d),t,cohs,"pearson_mean") for m,d,t in INTRA}
    best=max(cells,key=lambda k:cells[k][0] if cells[k] else -1)
    row=f"{oname:11s}"
    for m in BASELINE_ORDER:
        st=cells[m]; c=sub(st) if st else r"\running"
        if m==best and st: c=f"\\best{{{c}}}"
        row+=f" & {c}"
        if st: colavg[m].append(st[0])
    stm=cells["MorphoST"]; c=sub(stm) if stm else r"\running"
    if best=="MorphoST" and stm: c=f"\\best{{{c}}}"
    row+=f" & {c} \\\\"
    if stm: colavg["MorphoST"].append(stm[0])
    print(row)
avg={m:(np.mean(v) if v else None) for m,v in colavg.items()}
bestavg=max(avg,key=lambda k:avg[k] if avg[k] else -1)
ar="\\textbf{Average}"
for m in BASELINE_ORDER:
    a=avg[m]; c=f"${a:.3f}$" if a else r"\running"; c=f"\\best{{{c}}}" if m==bestavg else c; ar+=f" & {c}"
a=avg["MorphoST"]; c=f"${a:.3f}$" if a else r"\running"; c=f"\\best{{{c}}}" if bestavg=="MorphoST" else c
print(ar+f" & {c} \\\\")
# TABLE 2 (8-organ 3-metric)
print("\n%%% TABLE 2 %%%")
rows={}
for m,d,t in INTRA:
    r={fld:np.mean([organ(resolve(d),t,cohs,fld)[0] for _,cohs in ORGANS if organ(resolve(d),t,cohs,fld)]) for fld in ["pearson_mean","spearman_mean","mse"]}
    rows[m]=r
bp=max(rows,key=lambda k:rows[k]["pearson_mean"]); bs=max(rows,key=lambda k:rows[k]["spearman_mean"]); bm=min(rows,key=lambda k:rows[k]["mse"])
for m,_,_ in INTRA:
    r=rows[m]; p=f"{r['pearson_mean']:.3f}"; sc=f"{r['spearman_mean']:.3f}"; ms=f"{r['mse']:.3f}"
    if m==bp:p=f"\\best{{{p}}}"
    if m==bs:sc=f"\\best{{{sc}}}"
    if m==bm:ms=f"\\best{{{ms}}}"
    print(f"{m:11s} & {p} & {sc} & {ms} \\\\")
# ABLATION
print("\n%%% ABLATION (factorial LOOO, C000-C111) %%%")
FACT=resolve("MorphoST/results_corrected_factorial")
for code in ["000","001","010","011","100","101","110","111"]:
    ps=[]
    for s in SEEDS:
        d=os.path.join(FACT,f"LOOO_C{code}_seed{s}")
        v=[json.load(open(f))["pearson_mean"] for f in glob.glob(os.path.join(d,"fold_*_results.json"))]
        if v: ps.append(np.mean(v))
    print(f"C{code}: ${np.mean(ps):.3f}_{{{np.std(ps):.2f}}}$  (n={len(ps)})")
# LOSS CONTROL (8 organs)
print("\n%%% LOSS CONTROL (8 organs) %%%")
for lbl,d,t in [("MorphoST MSE","MorphoST/results_corrected_hest_mse","V3"),
                ("MorphoST MSE+PCC","MorphoST/results_corrected_hest_corr","V3"),
                ("TRIPLEX MSE","results_corrected_hest_baselines","triplex"),
                ("TRIPLEX MSE+PCC","results_corrected_hest_baselines_corr","triplex")]:
    vals=[organ(resolve(d),t,cohs,"pearson_mean") for _,cohs in ORGANS]; vals=[v[0] for v in vals if v]
    print(f"{lbl:18s}: {np.mean(vals):.3f} ({len(vals)} organs)")
# CROSS-ORGAN MorphoST LOOO (factorial C111)
print("\n%%% CROSS-ORGAN MorphoST LOOO = factorial C111 above %%%")
