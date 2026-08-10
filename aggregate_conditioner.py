"""T3 — conditioner ablation table. Isolates the FiLM/adaLN conditioner on top of the same STFlow
backbone: none (base) -> descriptor -> hybrid -> local -> MoE. Same protocol as the main tables
(UNI, cross_organ_splits8 LOOO+POOLED, per-fold 50-gene panel, log1p, pearson_mean, seeds 1-3).
Reports Overall pearson_mean +/- seed-std and delta vs the none baseline. Emits comparison_conditioner.{csv,md}."""
import json, glob, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

# label, root, film-tag  (order = increasing conditioner richness)
VARIANTS = [
    ("STFlow (none)",        "results_final_uni8",  "none"),
    ("+ descriptor (desc)",  "results_final_uni8",  "desc"),
    ("+ hybrid",             "results_hybrid_uni8", "hybrid"),
    ("+ local (STFiLM)",     "results_local_uni8",  "local"),
    ("+ MoE",                "results_moe_uni8",    "moe"),
]
REGIMES = {"LOOO": 8, "POOLED": 5}


def overall(root, film, regime):
    """Seed-mean of the per-run overall (each run averages its folds); returns (mean, std, nseed)."""
    per_seed = []
    for s in (1, 2, 3):
        d = os.path.join(ROOT, root, f"{regime}_{film}_seed{s}")
        folds = glob.glob(os.path.join(d, "fold_*_results.json"))
        vals = [json.load(open(f)).get("pearson_mean") for f in folds]
        vals = [v for v in vals if v is not None]
        if vals:
            per_seed.append(np.mean(vals))
    if not per_seed:
        return float("nan"), float("nan"), 0
    return float(np.mean(per_seed)), float(np.std(per_seed)), len(per_seed)


rows = []
base = {r: overall("results_final_uni8", "none", r)[0] for r in REGIMES}
for label, root, film in VARIANTS:
    row = {"conditioner": label}
    for regime in REGIMES:
        m, sd, n = overall(root, film, regime)
        row[regime] = m; row[regime + "_std"] = sd
        row[regime + "_d"] = m - base[regime]  # delta vs none
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "comparison_conditioner.csv"), index=False)

md = os.path.join(ROOT, "comparison_conditioner.md")
with open(md, "w") as f:
    f.write("### Conditioner ablation — Overall pearson_mean (UNI, 3-seed). "
            "Δ = vs STFlow(none). Best bold.\n\n")
    f.write("| conditioner | LOOO | Δ | POOLED | Δ |\n|---|---|---|---|---|\n")
    best = {r: max(rows, key=lambda x: (x[r] if not np.isnan(x[r]) else -1))["conditioner"] for r in REGIMES}
    for r in rows:
        def cell(reg):
            v = f"{r[reg]:.4f} ± {r[reg+'_std']:.4f}"
            return f"**{v}**" if r["conditioner"] == best[reg] else v
        dl = "—" if r["conditioner"] == "STFlow (none)" else f"{r['LOOO_d']:+.4f}"
        dp = "—" if r["conditioner"] == "STFlow (none)" else f"{r['POOLED_d']:+.4f}"
        f.write(f"| {r['conditioner']} | {cell('LOOO')} | {dl} | {cell('POOLED')} | {dp} |\n")

print(open(md).read())
print("wrote comparison_conditioner.{csv,md}")
