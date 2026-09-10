"""Aggregate the 'is FiLM general?' 3-seed grid into a table with error bars + significance.

Reads results_film_baselines/LOOO_<model>-<film>_seed<s>/fold_<fold>_results.json (per-fold
pearson_mean). For each backbone: mean +/- std across seeds for none/desc/local, the delta of the
BEST FiLM variant vs none, and a Wilcoxon signed-rank test on the paired per-fold values
(none vs each FiLM variant, pooled over the 8 LOOO folds x available seeds).
"""
import os, sys, json, glob
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(__file__), "results_film_baselines")
REGIME = sys.argv[1] if len(sys.argv) > 1 else "LOOO"   # LOOO | POOLED
MODELS = ["stnet", "deepspace", "histogene", "hist2st", "mlpprobe", "triplex", "bleep"]
FILMS = ["none", "desc", "local"]
SEEDS = [1, 2, 3]


def fold_means(model, film, seed):
    """Return dict fold->pearson_mean for one (model,film,seed), or {} if missing."""
    d = os.path.join(ROOT, f"{REGIME}_{model}-{film}_seed{seed}")
    out = {}
    for f in glob.glob(os.path.join(d, "fold_*_results.json")):
        fold = os.path.basename(f)[len("fold_"):-len("_results.json")]
        try:
            out[fold] = json.load(open(f))["pearson_mean"]
        except Exception:
            pass
    return out


def seed_means(model, film):
    """kfold mean per seed (mean over that seed's folds); list over seeds present."""
    vals = []
    for s in SEEDS:
        fm = fold_means(model, film, s)
        if fm:
            vals.append(np.mean(list(fm.values())))
    return np.array(vals)


def paired(model, film_a, film_b):
    """Paired per-(fold,seed) arrays for two films, aligned on common keys."""
    A, B = [], []
    for s in SEEDS:
        fa, fb = fold_means(model, film_a, s), fold_means(model, film_b, s)
        for k in sorted(set(fa) & set(fb)):
            A.append(fa[k]); B.append(fb[k])
    return np.array(A), np.array(B)


def fmt(v):
    return f"{v:.4f}" if v == v else "  -  "


print(f"{'backbone':10s} | {'none':>16s} {'desc':>16s} {'local':>16s} | "
      f"{'best-Δ':>8s} {'p(Wilcox)':>10s}  seeds")
print("-" * 92)
for m in MODELS:
    cells, means = {}, {}
    for f in FILMS:
        sm = seed_means(m, f)
        means[f] = sm
        if len(sm):
            cells[f] = f"{sm.mean():.4f}±{sm.std():.3f}" if len(sm) > 1 else f"{sm.mean():.4f}"
        else:
            cells[f] = "-"
    if not len(means["none"]):
        continue
    base = means["none"].mean()
    # best available FiLM variant
    cand = [f for f in ("desc", "local") if len(means[f])]
    if cand:
        best_f = max(cand, key=lambda f: means[f].mean())
        delta = means[best_f].mean() - base
        A, Bp = paired(m, "none", best_f)
        try:
            p = wilcoxon(A, Bp).pvalue if len(A) >= 6 and np.any(A != Bp) else float("nan")
        except Exception:
            p = float("nan")
        dstr, pstr = f"{delta:+.4f}", (f"{p:.4f}" if p == p else "  -  ")
    else:
        best_f, dstr, pstr = "-", "  -  ", "  -  "
    nseed = max(len(means[f]) for f in FILMS)
    tag = "HELP" if (cand and delta > 0.004) else ("~" if cand else "")
    print(f"{m:10s} | {cells['none']:>16s} {cells['desc']:>16s} {cells['local']:>16s} | "
          f"{dstr:>8s} {pstr:>10s}  n={nseed}  best={best_f} {tag}")

# completeness
done = sum(len(seed_means(m, f)) for m in MODELS for f in FILMS
          if not (m == "bleep" and f == "local"))
total = (len(MODELS) - 1) * len(FILMS) * len(SEEDS) + 2 * len(SEEDS)  # bleep has no local
print("-" * 92)
print(f"completeness: {done}/{total} (model,film,seed) kfolds present")
