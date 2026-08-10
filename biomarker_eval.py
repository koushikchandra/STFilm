"""Biomarker-level evaluation (STFlow Table-2 style): per-gene Pearson for the clinically
relevant marker genes in our panels, per method, 3-seed mean. Shows whether FiLM conditioning
improves individual biomarkers, not just the average. Free: reads saved per-gene pearson_corrs
({'name','pearson_corr'}) from existing result dirs — no re-run, no prediction dumps needed."""
import json, glob, os, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# The 11 genes in our cross-organ panels, with clinical annotation.
MARKERS = {
    "MKI67":    "Ki-67 proliferation index (prognostic; grading)",
    "CD8A":     "cytotoxic T-cell (immune infiltration; immunotherapy)",
    "CD3E":     "pan T-cell (immune infiltration)",
    "CD79A":    "B-cell",
    "KIT":      "c-Kit (GIST/melanoma drug target)",
    "TNFRSF17": "BCMA (myeloma target)",
    "ACTA2":    "myofibroblast / stroma (aSMA)",
    "GZMK":     "cytotoxic granzyme-K T-cell",
    "CCR7":     "naive/central-memory T-cell homing",
    "CXCR4":    "chemokine receptor (metastasis)",
    "GPR183":   "immune cell positioning (EBI2)",
}

MODELS = [
    ("ST-Net",       "results_spatial_uni8", lambda r,s: f"{r}_stnet_seed{s}"),
    ("HisToGene",    "results_spatial_uni8", lambda r,s: f"{r}_histogene_seed{s}"),
    ("Hist2ST",      "results_spatial_uni8", lambda r,s: f"{r}_hist2st_seed{s}"),
    ("BLEEP",        "results_spatial_uni8", lambda r,s: f"{r}_bleep_seed{s}"),
    ("TRIPLEX",      "results_spatial_uni8", lambda r,s: f"{r}_triplex_seed{s}"),
    ("STFlow none",  "results_final_uni8",   lambda r,s: f"{r}_none_seed{s}"),
    ("STFiLM desc",  "results_final_uni8",   lambda r,s: f"{r}_desc_seed{s}"),
    ("STFiLM local", "results_local_uni8",   lambda r,s: f"{r}_local_seed{s}"),
]
REGIMES = ["LOOO", "POOLED"]
SEEDS = (1, 2, 3)

def gene_pcc(root, tagfn):
    """Return {gene: mean per-gene Pearson} over all folds/seeds/regimes for one model."""
    per = {g: [] for g in MARKERS}
    for regime in REGIMES:
        for s in SEEDS:
            d = os.path.join(ROOT, root, tagfn(regime, s))
            for f in glob.glob(os.path.join(d, "fold_*_results.json")):
                pc = json.load(open(f)).get("pearson_corrs", [])
                if not isinstance(pc, list): continue
                for e in pc:
                    if isinstance(e, dict) and e.get("name") in per and e.get("pearson_corr") is not None:
                        per[e["name"]].append(float(e["pearson_corr"]))
    return {g: (np.mean(v) if v else np.nan) for g, v in per.items()}, \
           {g: len(v) for g, v in per.items()}

results = {label: gene_pcc(root, fn)[0] for label, root, fn in MODELS}
counts  = gene_pcc(MODELS[2][1], MODELS[2][2])[1]

order = sorted(MARKERS, key=lambda g: -results["STFiLM local"][g])
labels = [m[0] for m in MODELS]

print("Per-biomarker Pearson (mean over all cross-organ folds x 3 seeds x both regimes)\n")
hdr = f"{'gene':9s} " + " ".join(f"{l.split()[-1][:6]:>7s}" for l in labels) + f"  {'Δ(loc-none)':>11s}  annotation"
print(hdr); print("-"*len(hdr))
for g in order:
    cells = " ".join(f"{results[l][g]:7.3f}" for l in labels)
    dloc = results["STFiLM local"][g] - results["STFlow none"][g]
    print(f"{g:9s} {cells}  {dloc:+11.3f}  {MARKERS[g]}")

print("\nMean over biomarkers:")
for l in labels:
    vals = [results[l][g] for g in MARKERS if not np.isnan(results[l][g])]
    print(f"  {l:14s} {np.mean(vals):.4f}")

# how many biomarkers each FiLM variant improves vs none
for variant in ("STFiLM desc", "STFiLM local"):
    d = [results[variant][g] - results["STFlow none"][g] for g in MARKERS]
    wins = sum(x > 0 for x in d)
    print(f"\n{variant} beats none on {wins}/{len(MARKERS)} biomarkers "
          f"(mean Δ = {np.mean(d):+.4f})")
