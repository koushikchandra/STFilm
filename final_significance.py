"""Final results table + paired significance of STFiLM-local+bigcap (our best) vs every baseline.
Pairs are per-fold (LOOO: 8 organs, POOLED: 5 folds), seed-averaged. Wilcoxon signed-rank
(primary, no normality assumption) + paired t (cross-check), Holm-corrected within each regime."""
import json, glob, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
LOOO = ["kidney","liver","lung","pancreas","prostate","skin","breast","colorectal"]
POOLED = ["0","1","2","3","4"]

# label -> (result_root, tag builder given regime), seeds averaged
MODELS = [
    ("ST-Net",              "results_spatial_uni8", lambda r,s: f"{r}_stnet_seed{s}"),
    ("HisToGene",           "results_spatial_uni8", lambda r,s: f"{r}_histogene_seed{s}"),
    ("Hist2ST",             "results_spatial_uni8", lambda r,s: f"{r}_hist2st_seed{s}"),
    ("BLEEP",               "results_spatial_uni8", lambda r,s: f"{r}_bleep_seed{s}"),
    ("TRIPLEX",             "results_spatial_uni8", lambda r,s: f"{r}_triplex_seed{s}"),
    ("STFlow (ICML'25)",    "results_final_uni8",   lambda r,s: f"{r}_none_seed{s}"),
    ("STFiLM-local",        "results_local_uni8",   lambda r,s: f"{r}_local_seed{s}"),
    ("STFiLM-local+bigcap", "results_bigcap_uni8",  lambda r,s: f"{r}_local_bigcap_seed{s}"),
]
BEST = "STFiLM-local+bigcap"

def fold_means(path):
    out = {}
    for f in glob.glob(os.path.join(path, "fold_*_results.json")):
        d = json.load(open(f)); out[str(d["fold"])] = d["pearson_mean"]
    return out

def fold_vec(root, mk, regime, folds, seeds=(1,2,3)):
    per = {f: [] for f in folds}
    for s in seeds:
        fm = fold_means(os.path.join(ROOT, root, mk(regime, s)))
        for f in folds:
            if f in fm: per[f].append(fm[f])
    return np.array([np.mean(per[f]) if per[f] else np.nan for f in folds])

def holm(pvals):
    m = len(pvals); order = sorted(range(m), key=lambda i: pvals[i])
    corr = [0.0]*m; running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i]); corr[i] = min(running, 1.0)
    return corr

for regime, folds in [("LOOO", LOOO), ("POOLED", POOLED)]:
    vecs = {lab: fold_vec(root, mk, regime, folds) for lab, root, mk in MODELS}
    print(f"\n===== {regime}: overall pearson_mean (seed-avg over folds) =====")
    for lab, root, mk in MODELS:
        ov = np.nanmean(vecs[lab])
        star = "  *" if lab.startswith("STFiLM") else ""
        print(f"  {lab:22s} {ov:.4f}{star}")

    print(f"\n----- paired test: {BEST} vs each baseline ({len(folds)} folds) -----")
    rows = []
    vb = vecs[BEST]
    for lab, root, mk in MODELS:
        if lab == BEST: continue
        va = vecs[lab]
        d = vb - va
        wins = int((d > 0).sum()); losses = int((d < 0).sum())
        try: _, wp = stats.wilcoxon(vb, va)
        except ValueError: wp = float("nan")
        _, tp = stats.ttest_rel(vb, va)
        rows.append([lab, d.mean(), wins, losses, wp, tp])
    hw = holm([r[4] for r in rows]); ht = holm([r[5] for r in rows])
    print(f"{'vs baseline':22s} {'Δ(best-base)':>12s} {'w/l':>6s} {'Wilcox_p':>9s} {'W_Holm':>8s} {'t_Holm':>8s}  sig")
    for (lab,dm,w,l,wp,tp), wholm, tholm in zip(rows, hw, ht):
        star = "***" if wholm < 0.05 else ("(.)" if wholm < 0.10 else "")
        print(f"{lab:22s} {dm:>+12.4f} {f'{w}/{l}':>6s} {wp:9.4f} {wholm:8.4f} {tholm:8.4f}  {star}")
