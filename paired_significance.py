"""Paired significance on per-fold pearson_mean (seed-averaged) for the margins that matter.
Pairs are per-fold (LOOO: 8 organs, POOLED: 5 folds). Uses Wilcoxon signed-rank (n small,
no normality assumption) + paired t as a cross-check. Reports mean delta and win-count."""
import json, glob, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
LOOO = ["kidney","liver","lung","pancreas","prostate","skin","breast","colorectal"]
POOLED = ["0","1","2","3","4"]

def fold_means(path):
    out = {}
    for f in glob.glob(os.path.join(path, "fold_*_results.json")):
        d = json.load(open(f)); out[str(d["fold"])] = d["pearson_mean"]
    return out

def model_fold_vec(root, tag_fn, folds, seeds=(1,2,3)):
    """Per-fold value averaged over available seeds."""
    per = {f: [] for f in folds}
    for s in seeds:
        fm = fold_means(os.path.join(ROOT, root, tag_fn(s)))
        for f in folds:
            if f in fm: per[f].append(fm[f])
    return np.array([np.mean(per[f]) for f in folds])

MODELS = {
    "none":  ("results_final_uni8", lambda r: lambda s: f"{r}_none_seed{s}"),
    "desc":  ("results_final_uni8", lambda r: lambda s: f"{r}_desc_seed{s}"),
    "local": ("results_local_uni8", lambda r: lambda s: f"{r}_local_seed{s}"),
    "moe":   ("results_moe_uni8",   lambda r: lambda s: f"{r}_moe_seed{s}"),
    "histogene": ("results_spatial_uni8", lambda r: lambda s: f"{r}_histogene_seed{s}"),
    "hist2st":   ("results_spatial_uni8", lambda r: lambda s: f"{r}_hist2st_seed{s}"),
}

COMPARISONS = [("local","desc"), ("local","none"), ("local","histogene"), ("desc","none")]

def holm(pvals):
    """Holm-Bonferroni step-down. Returns corrected p-values in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    corr = [0.0]*m
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pvals[i]
        running = max(running, adj)          # enforce monotonicity
        corr[i] = min(running, 1.0)
    return corr

for regime, folds in [("LOOO", LOOO), ("POOLED", POOLED)]:
    print(f"\n===== {regime}  (paired over {len(folds)} folds, seed-averaged) =====")
    vecs = {m: model_fold_vec(root, mk(regime), folds) for m,(root,mk) in MODELS.items()}
    rows = []
    for a, b in COMPARISONS:
        va, vb = vecs[a], vecs[b]
        d = va - vb
        wins = int((d > 0).sum()); losses = int((d < 0).sum())
        try:
            _, w_p = stats.wilcoxon(va, vb)
        except ValueError:
            w_p = float("nan")
        _, t_p = stats.ttest_rel(va, vb)
        rows.append([a, b, d.mean(), wins, losses, w_p, t_p])
    # Holm correction over the family of contrasts within this regime (paired-t p-values)
    holm_t = holm([r[6] for r in rows])
    holm_w = holm([r[5] for r in rows])
    print(f"{'contrast':22s} {'Δ':>8s} {'w/l':>6s} "
          f"{'t_p':>7s} {'t_Holm':>8s} {'Wilcox':>8s} {'W_Holm':>8s}  sig(Holm,t)")
    for (a,b,dm,w,l,wp,tp), ht, hw in zip(rows, holm_t, holm_w):
        star = "***" if ht < 0.05 else ("(.)" if ht < 0.10 else "")
        print(f"{a+' vs '+b:22s} {dm:+.4f} {f'{w}/{l}':>6s} "
              f"{tp:7.4f} {ht:8.4f} {wp:8.4f} {hw:8.4f}  {star}")
