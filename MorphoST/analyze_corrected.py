"""Aggregate corrected nested-validation runs and compute paired comparisons."""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import paired_bootstrap


RUN_RE = re.compile(r"(?P<prefix>.*)_seed(?P<seed>\d+)$")


def collect(root):
    rows = []
    for path in glob.glob(os.path.join(root, "**", "fold_*_results.json"), recursive=True):
        run = Path(path).parent.name
        if not (Path(path).parent / "results_kfold.json").is_file():
            continue
        match = RUN_RE.match(run)
        result = json.load(open(path))
        rows.append({
            "run": run,
            "configuration": match.group("prefix") if match else run,
            "seed": int(match.group("seed")) if match else np.nan,
            "fold": str(result.get("fold", Path(path).stem)),
            **{k: result.get(k) for k in ["pearson_mean", "pearson_median", "pearson_slide_macro",
                                          "spearman_mean", "mse", "rmse", "mae", "ccc_mean",
                                          "n_spots", "n_train_slides", "n_val_slides", "n_test_slides"]},
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+")
    p.add_argument("--output_dir", default="analysis_corrected")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for root in args.roots:
        frame = collect(root)
        if len(frame):
            frame.insert(0, "result_root", root)
            frames.append(frame)
    if not frames:
        raise SystemExit("No corrected fold result files found")
    folds = pd.concat(frames, ignore_index=True)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    metrics = [c for c in ["pearson_mean", "pearson_slide_macro", "spearman_mean", "mse", "rmse",
                            "mae", "ccc_mean"] if folds[c].notna().any()]
    seeds = folds.groupby(["result_root", "configuration", "seed"], dropna=False)[metrics].mean().reset_index()
    seeds.to_csv(out / "seed_metrics.csv", index=False)
    summary = seeds.groupby(["result_root", "configuration"])[metrics].agg(["mean", "std", "count"])
    summary.to_csv(out / "summary.csv")
    print(summary.to_string())
    if args.compare:
        a, b = args.compare
        left = folds[folds.configuration == a][["seed", "fold", "pearson_mean"]]
        right = folds[folds.configuration == b][["seed", "fold", "pearson_mean"]]
        paired = left.merge(right, on=["seed", "fold"], suffixes=("_a", "_b"))
        result = paired_bootstrap(paired.pearson_mean_a, paired.pearson_mean_b)
        result.update({"configuration_a": a, "configuration_b": b})
        (out / "paired_bootstrap.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
