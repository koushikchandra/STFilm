#!/usr/bin/env python3
"""
Cross-organ split generator for the HEST-1k benchmark (STFlow format).

Builds two evaluation protocols on top of the per-cohort HEST-bench splits so we
can pressure-test the metadata-FiLM generalization claim for STFlow:

  Regime B (leave-one-organ-out, LOOO)  -> tests cross-organ TRANSFER.
      For each organ group: test = all its cohorts, train = every other cohort.
      The held-out organ is UNSEEN, so its metadata embedding falls back to the
      null token at eval time (see FiLM_ideas.md). Tests per-layer FiLM + pooling.

  Regime A (pooled, leave-patient-out, all organs seen) -> tests the METADATA benefit.
      Train on all organs pooled; hold out whole patients within each organ so every
      organ is present in both train and test of every fold (organ embeddings trained).

Output CSVs keep STFlow's columns (sample_id, patches_path, expr_path) but rewrite
the paths to be relative to --hest_root (prefixed with the cohort dir) so a single
pooled run can load tiles/expr across cohorts. Run STFlow with source_dataroot=hest_root
and gene_list pointing at the emitted shared panel json.

Gene-panel leakage rule: the shared panel for a Regime-B fold is built ONLY from the
fold's TRAINING cohorts, so the held-out organ's HVGs never influence panel selection.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# The 10 HEST-bench cohorts (see STFlow benchmark.py:279), grouped to organ so that
# biologically near-duplicate cohorts never straddle the held-out boundary.
COHORT_TO_GROUP = {
    "CCRCC": "kidney",
    "COAD": "colorectal", "READ": "colorectal",   # both colorectal -> grouped
    "HCC": "liver",
    "IDC": "breast", "LYMPH_IDC": "breast",        # IDC + its lymph-node met -> grouped
    "LUNG": "lung",
    "PAAD": "pancreas",
    "PRAD": "prostate",
    "SKCM": "skin",
}
SPLIT_COLS = ["sample_id", "patches_path", "expr_path"]


def log(msg):
    print(f"[splits] {msg}", file=sys.stderr)


def discover_inventory(hest_root):
    """Read every cohort's existing splits/*.csv, dedup by sample_id, and return one
    DataFrame per cohort with paths rewritten relative to hest_root."""
    inv = {}
    for cohort in sorted(COHORT_TO_GROUP):
        cdir = os.path.join(hest_root, cohort)
        sdir = os.path.join(cdir, "splits")
        if not os.path.isdir(sdir):
            log(f"skip {cohort}: no splits/ dir at {sdir}")
            continue
        frames = []
        for fn in sorted(os.listdir(sdir)):
            if fn.endswith(".csv"):
                frames.append(pd.read_csv(os.path.join(sdir, fn)))
        if not frames:
            log(f"skip {cohort}: no csv files in splits/")
            continue
        df = pd.concat(frames, ignore_index=True)
        missing = [c for c in SPLIT_COLS if c not in df.columns]
        if missing:
            log(f"skip {cohort}: split csv missing columns {missing}")
            continue
        df = df.drop_duplicates(subset="sample_id").reset_index(drop=True)
        # rewrite paths so they resolve from hest_root, not the cohort dir
        df["patches_path"] = cohort + "/" + df["patches_path"].astype(str).str.lstrip("/")
        df["expr_path"] = cohort + "/" + df["expr_path"].astype(str).str.lstrip("/")
        df["cohort"] = cohort
        df["organ_group"] = COHORT_TO_GROUP[cohort]
        inv[cohort] = df
        log(f"{cohort:10s} ({COHORT_TO_GROUP[cohort]:11s}): {len(df)} samples")
    if not inv:
        raise SystemExit(f"No usable cohorts found under {hest_root}")
    return inv


def attach_patient(df, hest_meta, id_col, patient_col):
    """Add a 'patient' column for leakage-safe grouping. Falls back to sample_id
    (each sample its own group) when no metadata is supplied."""
    if hest_meta and os.path.isfile(hest_meta):
        meta = pd.read_csv(hest_meta)
        if id_col in meta.columns and patient_col in meta.columns:
            m = dict(zip(meta[id_col].astype(str), meta[patient_col].astype(str)))
            df["patient"] = df["sample_id"].astype(str).map(m)
            n_missing = df["patient"].isna().sum()
            if n_missing:
                log(f"WARN {n_missing} samples had no patient in metadata -> using sample_id")
                df["patient"] = df["patient"].fillna(df["sample_id"].astype(str))
            return df
        log(f"WARN metadata missing '{id_col}'/'{patient_col}' -> grouping by sample_id")
    else:
        log("WARN no --hest_meta -> grouping by sample_id (cannot dedup same-patient slides)")
    df["patient"] = df["sample_id"].astype(str)
    return df


def read_panel(hest_root, cohort):
    p = os.path.join(hest_root, cohort, "var_50genes.json")
    if not os.path.isfile(p):
        log(f"WARN {cohort}: no var_50genes.json")
        return []
    with open(p) as f:
        return list(json.load(f).get("genes", []))


def cohort_var_names(hest_root, df_cohort):
    """Genes measured in EVERY slide of a cohort (intersection over all h5ad reads).
    Used only to guarantee a panel gene is PRESENT (not to rank it) -> no leakage.
    Intersecting across all slides (not just the first) ensures a panel gene cannot
    be missing from any individual slide that the trainer will load. Optional."""
    try:
        import scanpy as sc
    except Exception:
        return None
    common = None
    for _, row in df_cohort.iterrows():
        ep = os.path.join(hest_root, row["expr_path"])
        if not os.path.isfile(ep):
            continue
        try:
            ad = sc.read_h5ad(ep)
        except Exception as e:
            log(f"WARN could not read var_names from {ep}: {e}")
            continue
        vn = set(map(str, ad.var_names))
        common = vn if common is None else (common & vn)
    return common


def build_panel(hest_root, inv, train_cohorts, mode, n_genes, check_availability,
                involved_cohorts):
    """Shared output gene panel built ONLY from train_cohorts (leakage-safe)."""
    panels = {c: read_panel(hest_root, c) for c in train_cohorts}

    if mode == "intersection":
        sets = [set(g) for g in panels.values() if g]
        genes = sorted(set.intersection(*sets)) if sets else []
    elif mode == "hvg":
        genes = _hvg_panel(hest_root, inv, train_cohorts, n_genes)
    else:  # union_topk (default): rank by how many training cohorts list the gene
        freq = Counter()
        for g in panels.values():
            freq.update(set(g))
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        genes = [g for g, _ in ranked]

    if check_availability:
        avail = []
        for c in involved_cohorts:  # train AND test cohorts must measure the gene
            vn = cohort_var_names(hest_root, inv[c])
            if vn is not None:
                avail.append(vn)
        if avail:
            common = set.intersection(*avail)
            before = len(genes)
            genes = [g for g in genes if g in common]
            log(f"  availability filter: {before} -> {len(genes)} genes present in all cohorts")

    if n_genes and len(genes) > n_genes:
        genes = genes[:n_genes]
    return genes


def _hvg_panel(hest_root, inv, train_cohorts, n_genes):
    """Expression-driven panel: genes measured in ALL training cohorts, ranked by
    pooled variance over training spots. Heavy; requires scanpy + downloaded h5ad."""
    try:
        import scanpy as sc
    except Exception:
        log("WARN scanpy unavailable -> hvg mode falls back to union_topk")
        return []
    var_name_sets, var_sums, n_spots = [], defaultdict(float), defaultdict(int)
    for c in train_cohorts:
        vn = None
        for _, row in inv[c].iterrows():
            ep = os.path.join(hest_root, row["expr_path"])
            if not os.path.isfile(ep):
                continue
            ad = sc.read_h5ad(ep)
            sc.pp.log1p(ad)
            names = np.array(list(map(str, ad.var_names)))
            vn = set(names) if vn is None else (vn & set(names))
            X = ad.X
            X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            v = X.var(axis=0)
            for g, val in zip(names, np.asarray(v).ravel()):
                var_sums[g] += float(val) * X.shape[0]
                n_spots[g] += X.shape[0]
        if vn is not None:
            var_name_sets.append(vn)
    if not var_name_sets:
        return []
    common = set.intersection(*var_name_sets)
    scored = sorted(((var_sums[g] / max(n_spots[g], 1), g) for g in common), reverse=True)
    return [g for _, g in scored[:n_genes]]


def write_split(out_dir, name, train_df, test_df, genes, meta):
    os.makedirs(os.path.join(out_dir, "splits"), exist_ok=True)
    train_df[SPLIT_COLS].to_csv(os.path.join(out_dir, "splits", f"train_{name}.csv"), index=False)
    test_df[SPLIT_COLS].to_csv(os.path.join(out_dir, "splits", f"test_{name}.csv"), index=False)
    with open(os.path.join(out_dir, f"genes_{name}.json"), "w") as f:
        json.dump({"genes": genes}, f, indent=2)
    meta.update({"n_train": int(len(train_df)), "n_test": int(len(test_df)),
                 "n_genes": len(genes)})
    return meta


def regime_B(hest_root, inv, out_root, mode, n_genes, check_availability):
    """Leave-one-organ-group-out."""
    out_dir = os.path.join(out_root, "LOOO")
    groups = sorted(set(COHORT_TO_GROUP[c] for c in inv))
    manifest = []
    for held in groups:
        test_cohorts = [c for c in inv if COHORT_TO_GROUP[c] == held]
        train_cohorts = [c for c in inv if COHORT_TO_GROUP[c] != held]
        if not train_cohorts:
            continue
        train_df = pd.concat([inv[c] for c in train_cohorts], ignore_index=True)
        test_df = pd.concat([inv[c] for c in test_cohorts], ignore_index=True)
        genes = build_panel(hest_root, inv, train_cohorts, mode, n_genes,
                            check_availability, train_cohorts + test_cohorts)
        m = write_split(out_dir, held, train_df, test_df, genes,
                        {"regime": "B_LOOO", "held_out_organ": held,
                         "test_cohorts": test_cohorts, "train_cohorts": train_cohorts})
        log(f"LOOO[{held:11s}] train={m['n_train']:4d} test={m['n_test']:4d} genes={m['n_genes']}")
        manifest.append(m)
    return out_dir, manifest


def regime_A(hest_root, inv, out_root, n_folds, mode, n_genes, check_availability, seed):
    """Pooled, leave-patient-out; every organ present in every fold."""
    out_dir = os.path.join(out_root, "POOLED")
    rng = np.random.default_rng(seed)
    full = pd.concat(inv.values(), ignore_index=True)
    # assign each patient (within its organ) to a fold so organs are balanced across folds
    fold_of = {}
    for organ, grp in full.groupby("organ_group"):
        patients = sorted(grp["patient"].unique())
        rng.shuffle(patients)
        for i, pt in enumerate(patients):
            fold_of[(organ, pt)] = i % n_folds
    full["fold"] = [fold_of[(o, p)] for o, p in zip(full["organ_group"], full["patient"])]

    all_cohorts = list(inv)
    genes = build_panel(hest_root, inv, all_cohorts, mode, n_genes,
                        check_availability, all_cohorts)
    manifest = []
    for i in range(n_folds):
        test_df = full[full["fold"] == i]
        train_df = full[full["fold"] != i]
        # guard against same-patient leakage across the boundary
        leaked = set(train_df["patient"]) & set(test_df["patient"])
        assert not leaked, f"patient leakage in fold {i}: {list(leaked)[:5]}"
        m = write_split(out_dir, str(i), train_df, test_df, genes,
                        {"regime": "A_pooled_leave_patient_out", "fold": i,
                         "organs": sorted(test_df["organ_group"].unique())})
        log(f"POOLED[fold {i}] train={m['n_train']:4d} test={m['n_test']:4d} "
            f"organs={len(m['organs'])} genes={m['n_genes']}")
        manifest.append(m)
    return out_dir, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hest_root", required=True,
                    help="Parent dir containing the per-cohort HEST-bench folders "
                         "(CCRCC/, COAD/, ... each with splits/ and var_50genes.json)")
    ap.add_argument("--out_root", required=True, help="Where to write LOOO/ and POOLED/")
    ap.add_argument("--regimes", default="AB", choices=["A", "B", "AB"],
                    help="A=pooled leave-patient-out, B=leave-one-organ-out (default both)")
    ap.add_argument("--n_folds", type=int, default=5, help="Regime A folds (default 5)")
    ap.add_argument("--n_genes", type=int, default=50, help="Shared panel size (default 50)")
    ap.add_argument("--gene_panel_mode", default="union_topk",
                    choices=["union_topk", "intersection", "hvg"],
                    help="union_topk (default, lightweight, leakage-safe), "
                         "intersection (genes HVG in all train cohorts), "
                         "hvg (expression-driven, needs scanpy + h5ad)")
    ap.add_argument("--check_availability", action="store_true",
                    help="Drop panel genes not present in every involved cohort's h5ad "
                         "(needs scanpy; prevents runtime gene-selection errors)")
    ap.add_argument("--hest_meta", default=None,
                    help="Optional HEST master metadata CSV for patient-level grouping")
    ap.add_argument("--meta_id_col", default="id")
    ap.add_argument("--meta_patient_col", default="patient")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    inv = discover_inventory(args.hest_root)
    for c in inv:
        inv[c] = attach_patient(inv[c], args.hest_meta, args.meta_id_col, args.meta_patient_col)
    os.makedirs(args.out_root, exist_ok=True)

    out = {}
    if "B" in args.regimes:
        d, m = regime_B(args.hest_root, inv, args.out_root, args.gene_panel_mode,
                        args.n_genes, args.check_availability)
        out["LOOO"] = {"dir": d, "folds": m}
    if "A" in args.regimes:
        d, m = regime_A(args.hest_root, inv, args.out_root, args.n_folds,
                        args.gene_panel_mode, args.n_genes, args.check_availability, args.seed)
        out["POOLED"] = {"dir": d, "folds": m}

    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump({"hest_root": args.hest_root, "config": vars(args), "outputs": out},
                  f, indent=2)
    log(f"done -> {os.path.join(args.out_root, 'manifest.json')}")


if __name__ == "__main__":
    main()
