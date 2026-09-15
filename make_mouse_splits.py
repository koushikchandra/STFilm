"""Build cross-organ splits for the mouse HEST benchmark.

Produces mouse_cross_organ_splits/ with:
  INTRA/<organ>/splits/  — per-organ 5-fold (patient-stratified)
  LOOO/splits/           — leave-one-organ-out (7 folds)
  POOLED/splits/         — 5-fold pooled across all organs
  genes_<fold>.json      — top-50 HVG selected from training spots only (no leakage)

Usage:
  python make_mouse_splits.py --dataset_root dataset --out_root mouse_cross_organ_splits
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ORGANS = [
    "MOUSE_BRAIN", "MOUSE_LIVER", "MOUSE_SKIN", "MOUSE_HEART",
    "MOUSE_BOWEL", "MOUSE_KIDNEY", "MOUSE_MUSCLE",
]
N_GENES = 50
N_INTRA_FOLDS = 5
N_POOLED_FOLDS = 5
SEED = 42


def organ_short(organ):
    return organ.replace("MOUSE_", "").lower()


def get_sample_ids(dataset_root, organ):
    adata_dir = Path(dataset_root) / organ / "adata"
    return sorted(p.stem for p in adata_dir.glob("*.h5ad"))


def _strip_prefix(a):
    """Strip 'GRCm38_' genome-build prefix from var_names if present."""
    if len(a.var_names) > 0 and a.var_names[0].startswith("GRCm38_"):
        a.var_names = a.var_names.str.replace("GRCm38_", "", regex=False)
    return a


def _load_normed(path):
    """Load h5ad, strip prefix, normalize, log1p; return (var_names, X as dense float32)."""
    a = sc.read_h5ad(str(path))
    a = _strip_prefix(a)
    a.obs_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    X = a.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return list(a.var_names), X.astype(np.float32)


def top_hvg(sample_paths, n_genes=N_GENES):
    """Memory-efficient top-N HVG via online per-gene mean/variance (Welford).

    Processes one sample at a time; intersects gene sets across samples.
    """
    gene_set = None
    gene_sum  = {}
    gene_sum2 = {}
    gene_cnt  = {}

    for path in sample_paths:
        try:
            var_names, X = _load_normed(path)
        except Exception as e:
            print(f"  WARNING: {path}: {e}")
            continue

        if gene_set is None:
            gene_set = set(var_names)
        else:
            gene_set = gene_set & set(var_names)

        for gi, g in enumerate(var_names):
            col = X[:, gi]
            gene_sum[g]  = gene_sum.get(g, 0.0)  + float(col.sum())
            gene_sum2[g] = gene_sum2.get(g, 0.0) + float((col ** 2).sum())
            gene_cnt[g]  = gene_cnt.get(g, 0)    + len(col)

    if not gene_set:
        return []

    # Compute variance for genes in the intersection
    variances = {}
    for g in gene_set:
        n = gene_cnt[g]
        if n < 2:
            variances[g] = 0.0
            continue
        mean = gene_sum[g] / n
        variances[g] = max(0.0, gene_sum2[g] / n - mean ** 2)

    top = sorted(variances, key=variances.get, reverse=True)[:n_genes]
    return top


def split_patients(sample_ids, n_folds, seed):
    """Split sample_ids into n_folds stratified folds (each sample = one patient here)."""
    rng = random.Random(seed)
    ids = list(sample_ids)
    rng.shuffle(ids)
    folds = [ids[i::n_folds] for i in range(n_folds)]
    return folds


def write_split_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["sample_id", "patches_path", "expr_path"])
    df.to_csv(str(path), index=False)


def make_row(sid, organ):
    return {
        "sample_id":    sid,
        "patches_path": f"{organ}/patches/{sid}.h5",
        "expr_path":    f"{organ}/adata/{sid}.h5ad",
    }


def make_intra_splits(dataset_root, out_root):
    print("\n=== INTRA splits ===")
    for organ in ORGANS:
        ids = get_sample_ids(dataset_root, organ)
        folds = split_patients(ids, N_INTRA_FOLDS, SEED)
        organ_dir = Path(out_root) / "INTRA" / organ_short(organ)
        splits_dir = organ_dir / "splits"

        for fi in range(N_INTRA_FOLDS):
            test_ids  = folds[fi]
            train_ids = [s for j, fold in enumerate(folds) for s in fold if j != fi]

            write_split_csv(splits_dir / f"train_{fi}.csv", [make_row(s, organ) for s in train_ids])
            write_split_csv(splits_dir / f"test_{fi}.csv",  [make_row(s, organ) for s in test_ids])

            # gene panel from training spots (memory-efficient: one sample at a time)
            train_paths = [Path(dataset_root) / organ / "adata" / f"{s}.h5ad" for s in train_ids]
            genes = top_hvg(train_paths)
            with open(organ_dir / f"genes_{fi}.json", "w") as f:
                json.dump({"genes": genes, "organ": organ, "fold": fi}, f, indent=2)

        print(f"  {organ}: {len(ids)} slides, {N_INTRA_FOLDS} folds")


def make_looo_splits(dataset_root, out_root):
    print("\n=== LOOO splits ===")
    looo_dir  = Path(out_root) / "LOOO"
    splits_dir = looo_dir / "splits"

    for test_organ in ORGANS:
        short = organ_short(test_organ)
        test_ids  = get_sample_ids(dataset_root, test_organ)
        train_rows = []
        train_ids_all = {}
        for organ in ORGANS:
            if organ == test_organ:
                continue
            ids = get_sample_ids(dataset_root, organ)
            train_ids_all[organ] = ids
            train_rows.extend([make_row(s, organ) for s in ids])

        write_split_csv(splits_dir / f"train_{short}.csv", train_rows)
        write_split_csv(splits_dir / f"test_{short}.csv",
                        [make_row(s, test_organ) for s in test_ids])

        # gene panel: HVGs from all training organs pooled (memory-efficient)
        train_paths = [Path(dataset_root) / org / "adata" / f"{sid}.h5ad"
                       for org, ids in train_ids_all.items() for sid in ids]
        genes = top_hvg(train_paths)
        with open(looo_dir / f"genes_{short}.json", "w") as f:
            json.dump({"genes": genes, "test_organ": test_organ, "fold": short}, f, indent=2)

        print(f"  test={short}: {len(test_ids)} test, {len(train_rows)} train rows")


def make_pooled_splits(dataset_root, out_root):
    print("\n=== POOLED splits ===")
    pooled_dir = Path(out_root) / "POOLED"
    splits_dir = pooled_dir / "splits"

    # gather all samples with organ label
    all_samples = []
    for organ in ORGANS:
        for sid in get_sample_ids(dataset_root, organ):
            all_samples.append((sid, organ))

    rng = random.Random(SEED)
    rng.shuffle(all_samples)
    folds = [all_samples[i::N_POOLED_FOLDS] for i in range(N_POOLED_FOLDS)]

    for fi in range(N_POOLED_FOLDS):
        test_pairs  = folds[fi]
        train_pairs = [(s, o) for j, fold in enumerate(folds) for s, o in fold if j != fi]

        write_split_csv(splits_dir / f"train_{fi}.csv",
                        [make_row(s, o) for s, o in train_pairs])
        write_split_csv(splits_dir / f"test_{fi}.csv",
                        [make_row(s, o) for s, o in test_pairs])

        # gene panel from training spots (memory-efficient)
        train_paths = [Path(dataset_root) / org / "adata" / f"{sid}.h5ad"
                       for sid, org in train_pairs]
        genes = top_hvg(train_paths)
        with open(pooled_dir / f"genes_{fi}.json", "w") as f:
            json.dump({"genes": genes, "fold": fi}, f, indent=2)

        print(f"  fold {fi}: {len(train_pairs)} train, {len(test_pairs)} test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="dataset")
    p.add_argument("--out_root",     default="mouse_cross_organ_splits")
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(root, args.dataset_root)
    out_root     = os.path.join(root, args.out_root)

    make_intra_splits(dataset_root, out_root)
    make_looo_splits(dataset_root, out_root)
    make_pooled_splits(dataset_root, out_root)

    # Write manifest
    manifest = {
        "organs":   {organ_short(o): len(get_sample_ids(dataset_root, o)) for o in ORGANS},
        "n_slides": sum(len(get_sample_ids(dataset_root, o)) for o in ORGANS),
        "n_genes":  N_GENES,
        "k_pooled": N_POOLED_FOLDS,
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSplits written to {out_root}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
