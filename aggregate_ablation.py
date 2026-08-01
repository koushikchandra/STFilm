"""Aggregate the FiLM ablation: V0 (film=none) vs V1 (film=context).

Reads every results_ablation/<run>/<COHORT>/{config.json,results_kfold.json},
groups by (cohort, film) across seeds, and reports mean pearson_mean and the
per-seed V1-V0 delta with error bars.
"""
import json, glob, os
import numpy as np
from collections import defaultdict

data = defaultdict(dict)  # (cohort, film) -> {seed: pearson_mean}
for cfg_path in glob.glob("results_ablation/*/*/config.json"):
    cohort_dir = os.path.dirname(cfg_path)
    cohort = os.path.basename(cohort_dir)
    rk = os.path.join(cohort_dir, "results_kfold.json")
    if not os.path.exists(rk):
        continue
    cfg = json.load(open(cfg_path))
    data[(cohort, cfg.get("film", "?"))][cfg.get("seed", "?")] = json.load(open(rk))["pearson_mean"]

cohorts = sorted(set(c for c, _ in data))
print(f"{'cohort':8s} {'V0 none':>16s} {'V1 context':>16s} {'mean delta':>14s}  seeds")
print("-" * 70)
rows = []
for cohort in cohorts:
    none, ctx = data.get((cohort, "none"), {}), data.get((cohort, "context"), {})
    seeds = sorted(set(none) & set(ctx))
    if not seeds:
        print(f"{cohort:8s}  (incomplete: none={sorted(none)} context={sorted(ctx)})")
        continue
    v0 = np.array([none[s] for s in seeds])
    v1 = np.array([ctx[s] for s in seeds])
    d = v1 - v0
    rows.append((cohort, d.mean()))
    print(f"{cohort:8s} {v0.mean():7.4f} ± {v0.std():.4f} {v1.mean():7.4f} ± {v1.std():.4f} "
          f"{d.mean():+7.4f} ± {d.std():.4f}  {seeds}")

if rows:
    deltas = np.array([d for _, d in rows])
    print("-" * 70)
    print(f"Cohorts where FiLM helped: {sum(d > 0 for _, d in rows)}/{len(rows)}")
    print(f"Mean delta across cohorts: {deltas.mean():+.4f} ± {deltas.std():.4f}")
