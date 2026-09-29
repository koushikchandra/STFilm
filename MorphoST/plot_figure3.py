"""Figure 3 analog for MorphoST: cross-organ LOOO spatial gene expression maps.

Layout: rows = test slides, columns = Ground Truth + baselines + MorphoST.
Each cell is a spatial scatter of spot expression; SCC annotated below each
prediction panel.  Mimics STevs Figure 3 (Liu et al., CVPR 2026).

Default: breast LOOO, CD3E gene, slides NCBI785 + TENX95.
These slides show the full contrast: Hist2ST fails (SCC=-0.03), ST-Net near
noise (0.13-0.28), while MorphoST robustly recovers spatial structure (0.63-0.68).

Usage:
  python MorphoST/plot_figure3.py --output MorphoST/paper/figures/figure3.pdf
  python MorphoST/plot_figure3.py --organ lung --gene TNFRSF17 \\
      --slides TENX118 TENX141 --output MorphoST/paper/figures/figure3_lung.pdf
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = [
    ("ST-Net",    "results_corrected_crossorgan_baselines", "LOOO_stnet_seed{seed}"),
    ("HisToGene", "results_corrected_crossorgan_baselines", "LOOO_histogene_seed{seed}"),
    ("Hist2ST",   "results_corrected_crossorgan_baselines", "LOOO_hist2st_seed{seed}"),
    ("TRIPLEX",   "results_corrected_crossorgan_baselines", "LOOO_triplex_seed{seed}"),
    ("MorphoST",  "MorphoST/results_corrected_factorial",   "LOOO_C111_seed{seed}"),
]

CMAP_EXPR  = "RdBu_r"
CMAP_ERROR = "OrRd"
PT_SIZE    = 3   # scatter point size (lowered for dense slides)


def load_npz(organ, result_dir, run_template, seed):
    path = os.path.join(
        ROOT, result_dir,
        run_template.format(seed=seed),
        f"fold_{organ}", "test_predictions.npz"
    )
    if not os.path.isfile(path):
        return None
    return np.load(path)


def gene_index(data, gene):
    genes = list(data["genes"].astype(str))
    if gene not in genes:
        return None
    return genes.index(gene)


def scc_str(pred, target):
    if np.std(pred) < 1e-8:
        return "SCC: —"
    return f"SCC: {spearmanr(pred, target)[0]:.3f}"


def plot_figure3(organ, gene, slides, seed, output, point_size=PT_SIZE):
    # load all method data
    method_data = []
    for name, result_dir, run_tmpl in METHODS:
        d = load_npz(organ, result_dir, run_tmpl, seed)
        if d is None:
            print(f"  MISSING: {name}")
            method_data.append((name, None))
        else:
            method_data.append((name, d))

    # use MorphoST as reference for gene list and slide IDs
    ref = next(d for _, d in method_data if d is not None)
    gene_idx_ref = gene_index(ref, gene)
    if gene_idx_ref is None:
        raise SystemExit(f"Gene '{gene}' not found in reference predictions. "
                         f"Available: {list(ref['genes'].astype(str))}")

    # select slides
    all_slides = list(np.unique(ref["slide_ids"].astype(str)))
    if slides:
        missing = [s for s in slides if s not in all_slides]
        if missing:
            raise SystemExit(f"Slides {missing} not in test set. Available: {all_slides}")
        show_slides = slides
    else:
        # auto: top-2 by GT expression variance
        slide_var = sorted(
            [(float(np.std(ref["target"][ref["slide_ids"].astype(str) == sl, gene_idx_ref])), sl)
             for sl in all_slides],
            reverse=True
        )
        show_slides = [sl for _, sl in slide_var[:2]]
    print(f"Showing slides: {show_slides}")

    n_slides  = len(show_slides)
    n_cols    = 1 + len(METHODS)   # GT + baselines + MorphoST
    col_labels = ["Ground\nTruth"] + [name for name, _ in method_data]

    fig_w = n_cols * 2.2
    fig_h = n_slides * 2.2 + 0.6   # +0.6 for header row
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06,
                        hspace=0.35, wspace=0.12)

    gs = gridspec.GridSpec(n_slides, n_cols, figure=fig,
                           hspace=0.35, wspace=0.12)

    # column header text
    for ci, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_title(label, fontsize=8, fontweight="bold" if label not in ("Ground\nTruth",) else "normal",
                     pad=2)

    for ri, slide in enumerate(show_slides):
        # compute shared vmin/vmax from GT and all valid predictions on this slide
        gt_vals_list = []
        pred_vals_list = []
        for name, d in method_data:
            if d is None:
                continue
            gi = gene_index(d, gene)
            if gi is None:
                continue
            mask = d["slide_ids"].astype(str) == slide
            if not mask.any():
                continue
            gt_vals_list.append(d["target"][mask, gi])
            pred_vals_list.append(d["pred"][mask, gi])

        if not gt_vals_list:
            continue
        all_gt   = np.concatenate(gt_vals_list)
        all_pred = np.concatenate(pred_vals_list)
        vmin = float(np.percentile(np.concatenate([all_gt, all_pred]), 2))
        vmax = float(np.percentile(np.concatenate([all_gt, all_pred]), 98))

        # ── Ground truth column ─────────────────────────────────────────────
        ref_d = next((d for _, d in method_data if d is not None), None)
        gi_ref = gene_index(ref_d, gene)
        mask_ref = ref_d["slide_ids"].astype(str) == slide
        coords = ref_d["coords"][mask_ref]
        gt = ref_d["target"][mask_ref, gi_ref]
        ax_gt = fig.add_subplot(gs[ri, 0])
        sc = ax_gt.scatter(coords[:, 0], -coords[:, 1], c=gt,
                           s=point_size, cmap=CMAP_EXPR, vmin=vmin, vmax=vmax, linewidths=0)
        ax_gt.set_aspect("equal"); ax_gt.axis("off")
        ax_gt.set_ylabel(slide, fontsize=7, rotation=90, labelpad=2)
        if ri == 0:
            pass  # title already set above
        cb = plt.colorbar(sc, ax=ax_gt, shrink=0.65, pad=0.02, fraction=0.05)
        cb.ax.tick_params(labelsize=5)

        # ── Prediction columns ───────────────────────────────────────────────
        for ci, (name, d) in enumerate(method_data):
            ax = fig.add_subplot(gs[ri, ci + 1])
            if d is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.axis("off"); continue

            gi = gene_index(d, gene)
            if gi is None:
                ax.text(0.5, 0.5, "gene\nnot in\npanel", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")
                ax.axis("off"); continue

            mask = d["slide_ids"].astype(str) == slide
            if not mask.any():
                ax.text(0.5, 0.5, "slide\nmissing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")
                ax.axis("off"); continue

            coords_m = d["coords"][mask]
            pred     = d["pred"][mask, gi]
            tgt      = d["target"][mask, gi]

            sc = ax.scatter(coords_m[:, 0], -coords_m[:, 1], c=pred,
                            s=point_size, cmap=CMAP_EXPR, vmin=vmin, vmax=vmax, linewidths=0)
            ax.set_aspect("equal"); ax.axis("off")

            label = scc_str(pred, tgt)
            scc_val = spearmanr(pred, tgt)[0] if np.std(pred) > 1e-8 else float("nan")
            label_color = "#b22222" if (not np.isnan(scc_val) and scc_val < 0) else "black"
            ax.text(0.5, -0.04, label, transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, color=label_color)

            is_morph = name == "MorphoST"
            for spine in ["bottom", "top", "left", "right"]:
                ax.spines[spine].set_visible(is_morph)
                if is_morph:
                    ax.spines[spine].set_color("#E05800")
                    ax.spines[spine].set_linewidth(1.5)

            cb = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, fraction=0.05)
            cb.ax.tick_params(labelsize=5)

    fig.suptitle(f"Cross-organ LOOO: {organ.capitalize()} — {gene} (seed {seed})",
                 fontsize=10, fontweight="bold", y=0.97)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--organ",  default="breast")
    p.add_argument("--gene",   default="CD3E")
    p.add_argument("--slides", nargs="+", default=["NCBI785", "TENX95"],
                   help="Test slide IDs to display (default: NCBI785 TENX95 for breast CD3E)")
    p.add_argument("--seed",   type=int, default=1)
    p.add_argument("--point_size", type=float, default=PT_SIZE)
    p.add_argument("--output", default="MorphoST/paper/figures/figure3.pdf")
    args = p.parse_args()
    plot_figure3(args.organ, args.gene, args.slides, args.seed, args.output, args.point_size)


if __name__ == "__main__":
    main()
