"""Aggregate the STImage-1K4M second benchmark into full comparison tables (mirrors
aggregate_comparison.py). Same protocol for every row: UNI features, stimage_splits LOOO+POOLED
folds, per-fold 50-gene panel, log1p, pearson_mean metric, seeds 1-3. Baselines live in
results_stimage_spatial_uni8; STFlow/STFiLM films in results_stimage_uni8.
Emits comparison_stimage_{LOOO,POOLED}.{csv,md}."""
import json, glob, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "results_stimage_spatial_uni8"   # feature-matched baselines
FILM = "results_stimage_uni8"           # STFlow(none) + STFiLM variants

# label, is_ours, root, name-builder(regime) -> list of exp dirs to average over seeds
MODELS = [
    ("ST-Net",              False, BASE, lambda r: [f"{r}_stnet_seed{s}" for s in (1, 2, 3)]),
    ("HisToGene",           False, BASE, lambda r: [f"{r}_histogene_seed{s}" for s in (1, 2, 3)]),
    ("Hist2ST",             False, BASE, lambda r: [f"{r}_hist2st_seed{s}" for s in (1, 2, 3)]),
    ("BLEEP",               False, BASE, lambda r: [f"{r}_bleep_seed{s}" for s in (1, 2, 3)]),
    ("TRIPLEX",             False, BASE, lambda r: [f"{r}_triplex_seed{s}" for s in (1, 2, 3)]),
    ("STFlow (ICML'25)",    False, FILM, lambda r: [f"{r}_none_seed{s}" for s in (1, 2, 3)]),
    ("STFiLM (ours)",       True,  FILM, lambda r: [f"{r}_desc_seed{s}" for s in (1, 2, 3)]),
    ("STFiLM-local (ours)", True,  FILM, lambda r: [f"{r}_local_seed{s}" for s in (1, 2, 3)]),
]

LOOO_FOLDS = ["breast", "kidney", "liver", "pancreas", "skin", "prostate", "brain", "heart"]
POOLED_FOLDS = ["0", "1", "2", "3", "4"]

# accepted-model publication year (venue in comments); ours have no external year
YEAR = {
    "ST-Net": "2020",            # Nat. Biomed. Eng.
    "HisToGene": "2021",         # bioRxiv
    "Hist2ST": "2022",           # Brief. Bioinform.
    "BLEEP": "2023",             # NeurIPS
    "TRIPLEX": "2024",           # CVPR
    "STFlow (ICML'25)": "2025",  # ICML
}


def fold_means(path):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "fold_*_results.json"))):
        d = json.load(open(f)); out[str(d["fold"])] = d["pearson_mean"]
    return out


def collect(root, expdirs):
    per = {}; seed_overall = []
    for e in expdirs:
        p = os.path.join(ROOT, root, e)
        if not os.path.isdir(p): continue
        fm = fold_means(p)
        if not fm: continue
        for k, v in fm.items(): per.setdefault(k, []).append(v)
        seed_overall.append(np.mean(list(fm.values())))
    permean = {k: float(np.mean(v)) for k, v in per.items()}
    ov = float(np.mean(seed_overall)) if seed_overall else float("nan")
    std = float(np.std(seed_overall)) if len(seed_overall) > 1 else float("nan")
    return permean, ov, std, len(seed_overall)


for regime, folds in [("LOOO", LOOO_FOLDS), ("POOLED", POOLED_FOLDS)]:
    rows = []
    for label, ours, root, builder in MODELS:
        permean, ov, std, nseed = collect(root, builder(regime))
        row = {"model": label, "year": YEAR.get(label, ""), "ours": "*" if ours else ""}
        for fo in folds: row[fo] = round(permean.get(fo, float("nan")), 4)
        row["OVERALL"] = round(ov, 4)
        row["overall_std"] = round(std, 4) if not np.isnan(std) else ""
        row["n_seeds"] = nseed
        rows.append(row)
    df = pd.DataFrame(rows)
    csv = os.path.join(ROOT, f"comparison_stimage_{regime}.csv")
    df.to_csv(csv, index=False)
    md = os.path.join(ROOT, f"comparison_stimage_{regime}.md")
    cols = list(df.columns)
    with open(md, "w") as f:
        f.write(f"### STImage-1K4M {regime} — pearson_mean (higher = better). `*` = our model.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"[{regime}] wrote {os.path.basename(csv)}, {os.path.basename(md)}")
    print(df.to_string(index=False)); print()
