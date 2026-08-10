"""T8 — parameter count / compute overhead. Instantiate the STFlow Denoiser for each conditioner
variant (exactly as trained: UNI features, 50 genes, hidden 128, 4 layers) and count parameters,
to show FiLM adds negligible overhead over STFlow. Baseline counts are from STFlow (ICML'25) Tab.6
(feature-matched front-ends differ, but the order of magnitude is the comparison that matters).
Emits comparison_params.md."""
import argparse, os
import torch
from types import SimpleNamespace

import train_cross_organ as T
from stflow.model.denoiser import Denoiser

T.configure_organ_set("hest")  # sets N_ORGANS = 8

BASE = dict(
    n_genes=50, feature_dim=1024, feature_encoder="uni_v1_official",
    hidden_dim=128, pairwise_hidden_dim=128, n_layers=4, n_heads=4,
    dropout=0.2, attn_dropout=0.2, n_neighbors=8, norm="layer", activation="swiglu",
    backbone="spatial_transformer", n_proto=8, n_experts=4, moe_top_k=0,
    use_prototypes_in_router=False, lambda_bal=1e-2, lambda_smooth=1e-3,
    prior_sampler="zinb", n_sample_steps=5,
)

FILMS = ["none", "meta", "desc", "hybrid", "local", "moe"]


def build(film):
    cfg = SimpleNamespace(**BASE)
    cfg.film = film
    cfg.meta_categories = {"organ": T.N_ORGANS} if film == "meta" else None
    return Denoiser(cfg)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    counts = {}
    for f in FILMS:
        try:
            counts[f] = nparams(build(f))
        except Exception as e:
            counts[f] = None
            print(f"[warn] {f}: {e}")
    base = counts["none"]

    # STFlow Tab.6 baseline param counts (millions)
    BASELINE_M = [("ST-Net", 0.051), ("BLEEP", 0.670), ("HisToGene", 149.046),
                  ("TRIPLEX", 13.767), ("STFlow", 1.147)]

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_params.md")
    with open(out, "w") as fh:
        fh.write("### Model size / conditioning overhead (UNI, 50 genes, hidden 128, 4 layers)\n\n")
        fh.write("**Accepted baselines** (params from STFlow Tab.~6):\n\n")
        fh.write("| model | #params (M) |\n|---|---|\n")
        for name, m in BASELINE_M:
            fh.write(f"| {name} | {m:.3f} |\n")
        fh.write("\n**Our conditioner variants** (STFlow backbone + FiLM):\n\n")
        fh.write("| variant | #params (M) | Δ vs none (params) |\n|---|---|---|\n")
        for f in FILMS:
            c = counts[f]
            if c is None:
                fh.write(f"| {f} | — | — |\n"); continue
            d = c - base
            dstr = "—" if f == "none" else f"+{d:,}"
            fh.write(f"| {f} | {c/1e6:.3f} | {dstr} |\n")
        dloc = counts["local"] - base
        fh.write(f"\nThe recommended conditioner, \\textsc{{local}}, adds only {dloc:,} parameters "
                 f"(+{100*dloc/base:.1f}\\%) over the {base/1e6:.3f}M STFlow backbone, for a "
                 f"{counts['local']/1e6:.3f}M model --- 1--2 orders of magnitude smaller than "
                 f"TRIPLEX ({13.767:.1f}M) and HisToGene ({149.0:.0f}M). \\textsc{{meta}} is "
                 f"essentially free (+{counts['meta']-base:,}); only \\textsc{{moe}} inflates the "
                 f"model ({counts['moe']/1e6:.3f}M), consistent with its lack of benefit.\n")
    print(open(out).read())
    print("wrote comparison_params.md")


if __name__ == "__main__":
    main()
