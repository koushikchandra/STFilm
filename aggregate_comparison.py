"""Aggregate all cross-organ baselines + our model into one comparison table.

Same protocol for every row: UNI features, cross_organ_splits8 LOOO+POOLED folds,
per-fold 50-gene panel, log1p, pearson_mean metric. STFlow rows average over seeds 1-3;
single-seed baselines use their one run. Emits comparison_all.{csv,md}.
"""
import json, glob, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

# label, is_ours, root, name-builder(regime) -> list of exp dirs to average
MODELS = [
    ("Ridge probe",         False, "results_probe_uni8",   lambda r: [f"{r}_ridge"]),
    ("RandomForest probe",  False, "results_probe_uni8",   lambda r: [f"{r}_rf"]),
    ("ST-Net",              False, "results_spatial_uni8", lambda r: [f"{r}_stnet_seed{s}" for s in (1,2,3)]),
    ("HisToGene",           False, "results_spatial_uni8", lambda r: [f"{r}_histogene_seed{s}" for s in (1,2,3)]),
    ("Hist2ST",             False, "results_spatial_uni8", lambda r: [f"{r}_hist2st_seed{s}" for s in (1,2,3)]),
    ("BLEEP",               False, "results_spatial_uni8", lambda r: [f"{r}_bleep_seed{s}" for s in (1,2,3)]),
    ("TRIPLEX",             False, "results_spatial_uni8", lambda r: [f"{r}_triplex_seed{s}" for s in (1,2,3)]),
    ("STFlow (ICML'25)",    False, "results_final_uni8", lambda r: [f"{r}_none_seed{s}" for s in (1,2,3)]),
    ("STFiLM (ours)",       True,  "results_final_uni8", lambda r: [f"{r}_desc_seed{s}" for s in (1,2,3)]),
    ("STFiLM-local (ours)", True,  "results_local_uni8", lambda r: [f"{r}_local_seed{s}" for s in (1,2,3)]),
    ("STFlow+MoE (ours)",   True,  "results_moe_uni8",   lambda r: [f"{r}_moe_seed{s}" for s in (1,2,3)]),
]

LOOO_FOLDS  = ["kidney","liver","lung","pancreas","prostate","skin","breast","colorectal"]
POOLED_FOLDS= ["0","1","2","3","4"]

# accepted-model publication year (venue in comments); probes/ours have no external year
YEAR = {
    "ST-Net": "2020",            # Nat. Biomed. Eng.
    "HisToGene": "2021",         # bioRxiv
    "Hist2ST": "2022",           # Brief. Bioinform.
    "BLEEP": "2023",             # NeurIPS
    "TRIPLEX": "2024",           # CVPR
    "STFlow (ICML'25)": "2025",  # ICML
}

def fold_means(path):
    out={}
    for f in sorted(glob.glob(os.path.join(path,"fold_*_results.json"))):
        d=json.load(open(f)); out[str(d["fold"])]=d["pearson_mean"]
    return out

def collect(root, expdirs):
    per={}; seed_overall=[]
    for e in expdirs:
        p=os.path.join(ROOT,root,e)
        if not os.path.isdir(p): continue
        fm=fold_means(p)
        if not fm: continue
        for k,v in fm.items(): per.setdefault(k,[]).append(v)
        seed_overall.append(np.mean(list(fm.values())))
    permean={k:float(np.mean(v)) for k,v in per.items()}
    ov  = float(np.mean(seed_overall)) if seed_overall else float("nan")
    std = float(np.std(seed_overall))  if len(seed_overall)>1 else float("nan")
    return permean, ov, std, len(seed_overall)

for regime, folds in [("LOOO",LOOO_FOLDS),("POOLED",POOLED_FOLDS)]:
    rows=[]
    for label, ours, root, builder in MODELS:
        permean, ov, std, nseed = collect(root, builder(regime))
        row={"model":label, "year":YEAR.get(label,""), "ours":"*" if ours else ""}
        for fo in folds: row[fo]=round(permean.get(fo,float("nan")),4)
        row["OVERALL"]=round(ov,4)
        row["overall_std"]=round(std,4) if not np.isnan(std) else ""
        row["n_seeds"]=nseed
        rows.append(row)
    df=pd.DataFrame(rows)
    csv=os.path.join(ROOT,f"comparison_{regime}.csv")
    df.to_csv(csv,index=False)
    md=os.path.join(ROOT,f"comparison_{regime}.md")
    cols=list(df.columns)
    with open(md,"w") as f:
        f.write(f"### {regime} — pearson_mean (higher = better). `*` = our model.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for _,r in df.iterrows():
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"[{regime}] wrote {os.path.basename(csv)}, {os.path.basename(md)}")
    print(df.to_string(index=False)); print()
