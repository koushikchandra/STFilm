"""Create ground-truth, prediction, and error maps from exported fold predictions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions")
    p.add_argument("--output", required=True)
    p.add_argument("--gene", default=None)
    p.add_argument("--slide", default=None)
    p.add_argument("--point_size", type=float, default=7)
    args = p.parse_args()
    data = np.load(args.predictions)
    pred, target, coords = data["pred"], data["target"], data["coords"]
    genes, slides = data["genes"].astype(str), data["slide_ids"].astype(str)
    slide = args.slide or np.unique(slides)[0]
    mask = slides == slide
    if args.gene:
        matches = np.where(genes == args.gene)[0]
        if not len(matches): raise SystemExit(f"Gene {args.gene!r} not found")
        gene_idx = int(matches[0])
    else:
        scores = [np.corrcoef(pred[mask, i], target[mask, i])[0, 1]
                  if np.std(target[mask, i]) > 1e-8 else np.nan for i in range(len(genes))]
        gene_idx = int(np.nanargmax(scores))
    values = [target[mask, gene_idx], pred[mask, gene_idx],
              np.abs(pred[mask, gene_idx] - target[mask, gene_idx])]
    titles = ["Ground truth", "MorphoST", "Absolute error"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    vmin = min(values[0].min(), values[1].min()); vmax = max(values[0].max(), values[1].max())
    for i, (ax, val, title) in enumerate(zip(axes, values, titles)):
        im = ax.scatter(coords[mask, 0], -coords[mask, 1], c=val, s=args.point_size,
                        cmap="magma" if i < 2 else "viridis",
                        vmin=vmin if i < 2 else None, vmax=vmax if i < 2 else None)
        ax.set_title(title); ax.set_aspect("equal"); ax.axis("off"); fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle(f"{genes[gene_idx]} — {slide}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
