"""Generate all four paper figures for MorphoST.

Figure 1 – Performance heatmap:   8 organs × 7 methods, Pearson color-coded.
Figure 2 – Per-gene PCC violin:   distribution of 50-gene PCCs per method on SKCM (skin).
Figure 3 – SKCM spatial maps:     GT vs worst/mid/best baseline vs MorphoST for 3 skin marker genes.
Figure 4 – Ablation spatial maps: C000→C001→C011→C111 on lung LOOO, CCR7 gene.

Usage:
  python MorphoST/plot_paper_figures.py --outdir MorphoST/paper/figures/
"""
import argparse, glob, json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── shared palette ────────────────────────────────────────────────────────────
MORPH_COLOR  = "#E05800"   # orange for MorphoST highlight
BASE_COLOR   = "#2c7bb6"   # blue for baselines
CMAP_EXPR    = "RdBu_r"
CMAP_HEAT    = "YlOrRd"
PT           = 6           # scatter point size

METHOD_COLORS = {
    "ST-Net":    "#7B9CCC",
    "HisToGene": "#5E9F5E",
    "Hist2ST":   "#C97B3E",
    "TRIPLEX":   "#9B6DB5",
    "Gene-DML":  "#CC4F4F",
    "HyperST":   "#57A0A0",
    "MorphoST":  MORPH_COLOR,
}

INTRA_METHODS = [
    ("ST-Net",    "results_corrected_hest_baselines", "stnet"),
    ("HisToGene", "results_corrected_hest_baselines", "histogene"),
    ("Hist2ST",   "results_corrected_hest_baselines", "hist2st"),
    ("TRIPLEX",   "results_corrected_hest_baselines", "triplex"),
    ("Gene-DML",  "results_corrected_hest_baselines", "genedml"),
    ("HyperST",   "results_corrected_hest_baselines", "hyperst"),
    ("MorphoST",  "MorphoST/results_corrected_hest_corr", "V3"),
]

ORGANS = [
    ("Breast",     ["IDC", "LYMPH_IDC"]),
    ("Prostate",   ["PRAD"]),
    ("Pancreas",   ["PAAD"]),
    ("Skin",       ["SKCM"]),
    ("Colorectal", ["COAD", "READ"]),
    ("Kidney",     ["CCRCC"]),
    ("Liver",      ["HCC"]),
    ("Lung",       ["LUNG"]),
]
SEEDS = [1, 2, 3]


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve(*parts):
    return os.path.join(ROOT, *parts)


def cohort_mean(result_dir, cohort, tag, seed, field="pearson_mean"):
    vals = [json.load(open(f))[field]
            for f in glob.glob(resolve(result_dir, f"{cohort}_{tag}_seed{seed}", "fold_*_results.json"))]
    return float(np.mean(vals)) if vals else None


def organ_mean(result_dir, tag, cohs, field="pearson_mean"):
    """Mean ± std over seeds of the mean over cohorts in an organ group."""
    ps = []
    for s in SEEDS:
        cv = [cohort_mean(result_dir, c, tag, s, field) for c in cohs]
        cv = [x for x in cv if x is not None]
        if len(cv) == len(cohs):
            ps.append(float(np.mean(cv)))
    return (float(np.mean(ps)), float(np.std(ps))) if ps else (None, None)


def load_npz(path):
    return np.load(path) if os.path.isfile(path) else None


def gene_idx(data, gene):
    genes = list(data["genes"].astype(str))
    return genes.index(gene) if gene in genes else None


def scc(pred, tgt):
    return float(spearmanr(pred, tgt)[0]) if np.std(pred) > 1e-8 else float("nan")


def pcc(pred, tgt):
    return float(pearsonr(pred, tgt)[0]) if np.std(pred) > 1e-8 else float("nan")


def scatter_panel(ax, coords, values, vmin, vmax, cmap=CMAP_EXPR, pt=PT):
    sc = ax.scatter(coords[:, 0], -coords[:, 1], c=values,
                    s=pt, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")
    return sc


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Performance Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def figure1(outdir):
    print("Building Figure 1: Performance heatmap …")
    method_names = [m for m, _, _ in INTRA_METHODS]
    organ_names  = [o for o, _ in ORGANS]

    mat = np.full((len(ORGANS), len(INTRA_METHODS)), np.nan)
    for mi, (mname, mdir, tag) in enumerate(INTRA_METHODS):
        for oi, (oname, cohs) in enumerate(ORGANS):
            mu, _ = organ_mean(mdir, tag, cohs)
            if mu is not None:
                mat[oi, mi] = mu

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, cmap=CMAP_HEAT, aspect="auto", vmin=0.0, vmax=0.75)

    # cell annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.55 else "black"
                bold  = (j == len(INTRA_METHODS) - 1)  # MorphoST column
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color=color,
                        fontweight="bold" if bold else "normal")

    # axes labels
    ax.set_xticks(range(len(INTRA_METHODS)))
    ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ORGANS)))
    ax.set_yticklabels(organ_names, fontsize=9)

    # highlight MorphoST column
    morph_col = len(INTRA_METHODS) - 1
    for spine in ["bottom", "top", "left", "right"]:
        ax.spines[spine].set_visible(True)
    rect = plt.Rectangle((morph_col - 0.5, -0.5), 1, len(ORGANS),
                          linewidth=2.5, edgecolor=MORPH_COLOR, facecolor="none")
    ax.add_patch(rect)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Pearson correlation", fontsize=9)
    ax.set_title("Intra-organ gene expression prediction accuracy across 8 organ types",
                 fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout()
    out = Path(outdir) / "fig1_heatmap.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Per-gene PCC Violin Plot (SKCM)
# ═══════════════════════════════════════════════════════════════════════════════

def figure2(outdir):
    print("Building Figure 2: Per-gene PCC violin …")
    method_names = [m for m, _, _ in INTRA_METHODS]
    all_pccs = {}
    for mname, mdir, tag in INTRA_METHODS:
        vals = []
        for f in glob.glob(resolve(mdir, f"SKCM_{tag}_seed*", "fold_*_results.json")):
            vals.extend(json.load(open(f)).get("pearson_per_gene", []))
        all_pccs[mname] = np.array(vals) if vals else np.array([])

    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(method_names))

    for i, mname in enumerate(method_names):
        v = all_pccs[mname]
        if len(v) == 0:
            continue
        color = METHOD_COLORS.get(mname, BASE_COLOR)
        bp = ax.violinplot(v, positions=[i], widths=0.7, showmedians=True,
                           showextrema=False)
        for part in bp["bodies"]:
            part.set_facecolor(color)
            part.set_alpha(0.75)
            part.set_edgecolor("white")
            part.set_linewidth(0.5)
        bp["cmedians"].set_color("white")
        bp["cmedians"].set_linewidth(2)

        # individual gene dots (jittered)
        rng = np.random.default_rng(42 + i)
        jitter = rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(i + jitter, v, s=6, color=color, alpha=0.35, linewidths=0)

        # mean marker
        ax.scatter(i, np.mean(v), marker="D", s=40, color=color,
                   edgecolors="black", linewidths=0.8, zorder=5)

    # highlight MorphoST
    morph_i = len(method_names) - 1
    ax.axvspan(morph_i - 0.45, morph_i + 0.45, alpha=0.08, color=MORPH_COLOR)

    ax.set_xticks(positions)
    ax.set_xticklabels(method_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Per-gene Pearson correlation", fontsize=10)
    ax.set_title("Per-gene prediction accuracy distribution — Skin (SKCM) intra-cohort\n"
                 "(50 genes × 2 folds × 3 seeds = 300 values per method)",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path(outdir) / "fig2_violin.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – SKCM Spatial Expression Maps (intra-cohort)
# ═══════════════════════════════════════════════════════════════════════════════

def figure3(outdir):
    print("Building Figure 3: SKCM spatial expression maps …")

    SHOW_GENES   = ["KRTDAP", "SERPINB5", "LYZ"]    # top marker genes for skin
    SHOW_METHODS = [                                  # GT + 3 representative baselines + MorphoST
        ("ST-Net",    "results_corrected_hest_baselines", "stnet"),
        ("Hist2ST",   "results_corrected_hest_baselines", "hist2st"),
        ("Gene-DML",  "results_corrected_hest_baselines", "genedml"),
        ("MorphoST",  "MorphoST/results_corrected_hest_corr", "V3"),
    ]
    SEED, FOLD, SLIDE = 1, 0, "TENX117"

    # load all method predictions
    preds = {}
    for mname, mdir, tag in SHOW_METHODS:
        path = resolve(mdir, f"SKCM_{tag}_seed{SEED}", f"fold_{FOLD}", "test_predictions.npz")
        d = load_npz(path)
        preds[mname] = d

    ref = next(d for d in preds.values() if d is not None)
    mask_slide = ref["slide_ids"].astype(str) == SLIDE
    coords_ref = ref["coords"][mask_slide]

    n_genes = len(SHOW_GENES)
    n_cols  = 1 + len(SHOW_METHODS)   # GT + methods
    col_labels = ["Ground\nTruth"] + [m for m, _, _ in SHOW_METHODS]

    fig = plt.figure(figsize=(n_cols * 2.1, n_genes * 2.3))
    gs  = gridspec.GridSpec(n_genes, n_cols, hspace=0.25, wspace=0.08,
                            left=0.04, right=0.97, top=0.90, bottom=0.04)

    # column headers
    for ci, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_title(label, fontsize=9,
                     fontweight="bold" if label not in ("Ground\nTruth",) else "normal",
                     color=MORPH_COLOR if label == "MorphoST" else "black", pad=3)

    for ri, gene in enumerate(SHOW_GENES):
        # ground truth
        gi_ref = gene_idx(ref, gene)
        if gi_ref is None:
            continue
        gt_vals  = ref["target"][mask_slide, gi_ref]
        vmin = float(np.percentile(gt_vals, 1))
        vmax = float(np.percentile(gt_vals, 99))

        ax_gt = fig.add_subplot(gs[ri, 0])
        sc = scatter_panel(ax_gt, coords_ref, gt_vals, vmin, vmax, pt=PT)
        if ri == 0:
            pass  # title already set
        ax_gt.set_ylabel(gene, fontsize=9, rotation=90, labelpad=3)

        # method panels
        for ci, (mname, mdir, tag) in enumerate(SHOW_METHODS):
            ax = fig.add_subplot(gs[ri, ci + 1])
            d  = preds.get(mname)
            if d is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
                ax.axis("off"); continue

            gi = gene_idx(d, gene)
            if gi is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
                ax.axis("off"); continue

            mask_m = d["slide_ids"].astype(str) == SLIDE
            pred_v = d["pred"][mask_m, gi]
            tgt_v  = d["target"][mask_m, gi]
            coords_m = d["coords"][mask_m]

            scatter_panel(ax, coords_m, pred_v, vmin, vmax, pt=PT)

            p_val = pcc(pred_v, tgt_v)
            s_val = scc(pred_v, tgt_v)
            label_str = f"PCC={p_val:.2f}"
            is_morph = (mname == "MorphoST")
            txt_color = MORPH_COLOR if is_morph else "black"
            ax.text(0.5, -0.03, label_str, transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color=txt_color,
                    fontweight="bold" if is_morph else "normal")

            # highlight MorphoST with border
            if is_morph:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(MORPH_COLOR)
                    spine.set_linewidth(2.0)

    fig.suptitle("Intra-cohort spatial gene expression prediction — Skin (SKCM), slide TENX117",
                 fontsize=11, fontweight="bold", y=0.97)

    out = Path(outdir) / "fig3_skcm_maps.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Ablation Spatial Maps (C000 → C001 → C011 → C111)
# ═══════════════════════════════════════════════════════════════════════════════

def figure4(outdir):
    print("Building Figure 4: Ablation spatial maps …")

    ORGAN      = "lung"
    GENE       = "CCR7"
    SHOW_SLIDES = ["TENX118", "TENX141"]
    SEED       = 1
    CONFIGS    = [
        ("C000", "No conditioning"),
        ("C001", "+ Slide cond."),
        ("C011", "+ Global cond."),
        ("C111", "+ Local cond.\n(Full model)"),
    ]

    # load predictions for each ablation config
    abl_data = {}
    for code, label in CONFIGS:
        path = resolve("MorphoST", "results_corrected_factorial",
                       f"LOOO_{code}_seed{SEED}", f"fold_{ORGAN}", "test_predictions.npz")
        abl_data[code] = load_npz(path)

    ref = next(d for d in abl_data.values() if d is not None)
    gi_ref = gene_idx(ref, GENE)
    all_slides = list(np.unique(ref["slide_ids"].astype(str)))
    slides = [s for s in SHOW_SLIDES if s in all_slides]

    n_rows = len(slides)
    n_cols = 1 + len(CONFIGS)   # GT + ablation configs

    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.4))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.1,
                            left=0.06, right=0.97, top=0.88, bottom=0.04)

    col_labels = ["Ground\nTruth"] + [f"{code}\n{desc}" for code, desc in CONFIGS]
    for ci, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, ci])
        is_full = "Full model" in label
        ax.set_title(label, fontsize=8.5,
                     fontweight="bold" if is_full else "normal",
                     color=MORPH_COLOR if is_full else "black", pad=3)

    for ri, slide in enumerate(slides):
        ref_d = abl_data.get("C111") or ref
        gi_r  = gene_idx(ref_d, GENE)
        mask_r = ref_d["slide_ids"].astype(str) == slide
        gt_vals = ref_d["target"][mask_r, gi_r]
        coords  = ref_d["coords"][mask_r]
        vmin = float(np.percentile(gt_vals, 1))
        vmax = float(np.percentile(gt_vals, 99))

        # GT column
        ax_gt = fig.add_subplot(gs[ri, 0])
        scatter_panel(ax_gt, coords, gt_vals, vmin, vmax, pt=PT)
        ax_gt.set_ylabel(slide, fontsize=8, rotation=90, labelpad=3)

        # ablation columns
        for ci, (code, _) in enumerate(CONFIGS):
            ax  = fig.add_subplot(gs[ri, ci + 1])
            d   = abl_data.get(code)
            is_full = (code == "C111")

            if d is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray"); ax.axis("off"); continue

            gi = gene_idx(d, GENE)
            if gi is None:
                ax.axis("off"); continue

            mask_m   = d["slide_ids"].astype(str) == slide
            pred_v   = d["pred"][mask_m, gi]
            tgt_v    = d["target"][mask_m, gi]
            coords_m = d["coords"][mask_m]

            scatter_panel(ax, coords_m, pred_v, vmin, vmax, pt=PT)

            s_val = scc(pred_v, tgt_v)
            p_val = pcc(pred_v, tgt_v)
            label_str = f"SCC={s_val:.3f}"
            txt_color = MORPH_COLOR if is_full else "black"
            ax.text(0.5, -0.03, label_str, transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color=txt_color,
                    fontweight="bold" if is_full else "normal")

            if is_full:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(MORPH_COLOR)
                    spine.set_linewidth(2.0)

    fig.suptitle(f"Ablation: effect of adding conditioning components — Lung LOOO, {GENE}",
                 fontsize=11, fontweight="bold", y=0.96)

    out = Path(outdir) / "fig4_ablation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="MorphoST/paper/figures")
    p.add_argument("--figs", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="Which figures to generate (default: 1 2 3 4)")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if 1 in args.figs: figure1(args.outdir)
    if 2 in args.figs: figure2(args.outdir)
    if 3 in args.figs: figure3(args.outdir)
    if 4 in args.figs: figure4(args.outdir)

    print("\nDone. Files in:", args.outdir)


if __name__ == "__main__":
    main()
