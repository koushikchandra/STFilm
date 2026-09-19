"""Leakage-safe cross-organ splits for the built STImage-1K4M subset, in the same format
train_cross_organ.py consumes (sample_id,patches_path,expr_path CSVs + genes_<fold>.json).

Two regimes, mirroring cross_organ_splits8:
  LOOO   : leave-one-organ-out. test = all slides of organ O; train = every other organ.
           Panel = top-50 HVGs computed on the TRAINING organs only (never sees O) and
           restricted to genes present in every train+test slide (measurable, leakage-safe).
  POOLED : K-fold leave-slide-out, stratified by organ so every organ appears in train of
           every fold (tests the metadata benefit with all organs seen). Panel per fold from
           that fold's training slides only.

Panel size is fixed to 50 (MLPAttnEdgeAggregation hardcodes +50). HVG = top pooled log1p
variance over training spots, accumulated in a streaming fashion (memory-safe).
"""
import argparse, os, json, glob
import numpy as np
import pandas as pd
import scanpy as sc

N_GENES = 50


def list_slides(src_root):
    """{organ: [slide_id,...]} from dataset_stimage/<organ>/adata/*.h5ad"""
    out = {}
    for organ_dir in sorted(glob.glob(os.path.join(src_root, "*", "adata"))):
        organ = organ_dir.split(os.sep)[-2]
        sids = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(organ_dir, "*.h5ad"))]
        if sids:
            out[organ] = sorted(sids)
    return out


def var_names(src_root, organ, sid):
    a = sc.read_h5ad(os.path.join(src_root, organ, "adata", f"{sid}.h5ad"), backed="r")
    vn = list(map(str, a.var_names))
    a.file.close()
    return vn


def common_genes(src_root, slides):
    """Intersection of var_names over a list of (organ,sid)."""
    inter = None
    for organ, sid in slides:
        vn = set(var_names(src_root, organ, sid))
        inter = vn if inter is None else (inter & vn)
    return inter or set()


def hvg_panel(src_root, train_slides, candidate_genes, n=N_GENES):
    """Top-n genes by pooled variance of log1p counts over the training slides' spots.
    Streams slide-by-slide accumulating count/sum/sumsq per candidate gene."""
    genes = sorted(candidate_genes)
    gidx = {g: i for i, g in enumerate(genes)}
    s1 = np.zeros(len(genes)); s2 = np.zeros(len(genes)); n_spots = 0
    for organ, sid in train_slides:
        a = sc.read_h5ad(os.path.join(src_root, organ, "adata", f"{sid}.h5ad"))
        cols = [g for g in a.var_names if g in gidx]
        sub = a[:, cols]
        X = np.asarray(sub.X, dtype=np.float64)
        X = np.log1p(X)
        idx = np.array([gidx[g] for g in cols])
        s1[idx] += X.sum(0); s2[idx] += (X ** 2).sum(0); n_spots += X.shape[0]
    mean = s1 / max(1, n_spots)
    var = s2 / max(1, n_spots) - mean ** 2
    order = np.argsort(-var)
    return [genes[i] for i in order[:n]]


def write_split(regime_dir, fold, train_rows, test_rows, panel):
    sd = os.path.join(regime_dir, "splits"); os.makedirs(sd, exist_ok=True)
    def rows_to_df(rows):
        return pd.DataFrame([{
            "sample_id": sid,
            "patches_path": f"{organ}/patches/{sid}.h5",
            "expr_path": f"{organ}/adata/{sid}.h5ad",
        } for organ, sid in rows])
    rows_to_df(train_rows).to_csv(os.path.join(sd, f"train_{fold}.csv"), index=False)
    rows_to_df(test_rows).to_csv(os.path.join(sd, f"test_{fold}.csv"), index=False)
    with open(os.path.join(regime_dir, f"genes_{fold}.json"), "w") as f:
        json.dump({"genes": panel}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="dataset_stimage")
    ap.add_argument("--out", default="stimage_splits")
    ap.add_argument("--k_pooled", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    slides = list_slides(args.src_root)
    all_slides = [(o, s) for o, ss in slides.items() for s in ss]
    print(f"{len(all_slides)} slides across {len(slides)} organs: "
          + ", ".join(f"{o}={len(s)}" for o, s in slides.items()))

    # ---- LOOO ----
    loo_dir = os.path.join(args.out, "LOOO"); os.makedirs(loo_dir, exist_ok=True)
    for organ in slides:
        test_rows = [(organ, s) for s in slides[organ]]
        train_rows = [(o, s) for o, s in all_slides if o != organ]
        cand = common_genes(args.src_root, train_rows) & common_genes(args.src_root, test_rows)
        panel = hvg_panel(args.src_root, train_rows, cand)
        write_split(loo_dir, organ, train_rows, test_rows, panel)
        print(f"  LOOO {organ:9s}: train={len(train_rows)} test={len(test_rows)} "
              f"cand={len(cand)} panel={len(panel)}")

    # ---- INTRA (per-organ K-fold, mirrors HEST intra-cohort protocol) ----
    intra_dir = os.path.join(args.out, "INTRA"); os.makedirs(intra_dir, exist_ok=True)
    for organ, ss in slides.items():
        ss = list(ss); rng.shuffle(ss)
        for k in range(args.k_pooled):
            test_rows = [(organ, ss[i]) for i in range(k, len(ss), args.k_pooled)]
            train_rows = [(organ, ss[i]) for i in range(len(ss)) if i % args.k_pooled != k]
            cand = common_genes(args.src_root, train_rows) & common_genes(args.src_root, test_rows)
            panel = hvg_panel(args.src_root, train_rows, cand)
            fold_name = f"{organ}_{k}"
            write_split(intra_dir, fold_name, train_rows, test_rows, panel)
            print(f"  INTRA {fold_name:14s}: train={len(train_rows)} test={len(test_rows)} "
                  f"panel={len(panel)}")

    # ---- POOLED (leave-slide-out, organ-stratified K-fold) ----
    pool_dir = os.path.join(args.out, "POOLED"); os.makedirs(pool_dir, exist_ok=True)
    fold_of = {}
    for organ, ss in slides.items():
        ss = list(ss); rng.shuffle(ss)
        for i, s in enumerate(ss):
            fold_of[(organ, s)] = i % args.k_pooled
    for k in range(args.k_pooled):
        test_rows = [sl for sl in all_slides if fold_of[sl] == k]
        train_rows = [sl for sl in all_slides if fold_of[sl] != k]
        cand = common_genes(args.src_root, train_rows) & common_genes(args.src_root, test_rows)
        panel = hvg_panel(args.src_root, train_rows, cand)
        write_split(pool_dir, k, train_rows, test_rows, panel)
        org_in_test = sorted(set(o for o, _ in test_rows))
        print(f"  POOLED fold {k}: train={len(train_rows)} test={len(test_rows)} "
              f"panel={len(panel)} test_organs={len(org_in_test)}")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"organs": {o: len(s) for o, s in slides.items()},
                   "n_slides": len(all_slides), "k_pooled": args.k_pooled,
                   "n_genes": N_GENES}, f, indent=2)
    print(f"wrote splits to {args.out}")


if __name__ == "__main__":
    main()
