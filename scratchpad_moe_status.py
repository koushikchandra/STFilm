"""Live partial comparison of moe/local sweeps vs the finished none/desc runs.
Reads whatever fold_*_results.json exist so far; averages over available seeds."""
import json, glob, os, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LOOO = ["kidney","liver","lung","pancreas","prostate","skin","breast","colorectal"]
POOLED = ["0","1","2","3","4"]

def fold_means(path):
    out = {}
    for f in glob.glob(os.path.join(path, "fold_*_results.json")):
        try: d = json.load(open(f))
        except Exception: continue
        out[str(d["fold"])] = d["pearson_mean"]
    return out

def collect(root, tag_fn, seeds=(1,2,3)):
    per = {}
    for s in seeds:
        fm = fold_means(os.path.join(ROOT, root, tag_fn(s)))
        for k, v in fm.items(): per.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in per.items()}, {k: len(v) for k, v in per.items()}

REFS = [
    ("none (STFlow) ", "results_final_uni8", lambda r: lambda s: f"{r}_none_seed{s}"),
    ("desc (STFiLM) ", "results_final_uni8", lambda r: lambda s: f"{r}_desc_seed{s}"),
    ("local (new)   ", "results_local_uni8", lambda r: lambda s: f"{r}_local_seed{s}"),
    ("moe (new)     ", "results_moe_uni8",   lambda r: lambda s: f"{r}_moe_seed{s}"),
]

for regime, folds in [("LOOO", LOOO), ("POOLED", POOLED)]:
    print(f"\n===== {regime} (pearson_mean; partial) =====")
    print(f"{'model':16s} " + " ".join(f"{f[:6]:>7s}" for f in folds) + f" {'OVER*':>7s}  seeds")
    for label, root, mk in REFS:
        pm, nseed = collect(root, mk(regime))
        cells = " ".join(f"{pm[f]:7.3f}" if f in pm else f"{'-':>7s}" for f in folds)
        done = [pm[f] for f in folds if f in pm]
        ov = np.mean(done) if done else float("nan")
        smax = max(nseed.values()) if nseed else 0
        print(f"{label:16s} {cells} {ov:7.3f}  ({len(done)}/{len(folds)} folds, {smax} sd)")
    print("* OVER = mean over COMPLETED folds only (not comparable across rows until full).")
