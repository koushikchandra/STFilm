"""Feature-matched baselines under STFlow's PER-COHORT HEST protocol (within-organ k-fold), so we
have apples-to-apples columns for the MorphoST-V3 Table-1 comparison. Reuses baseline_spatial.py's
models UNCHANGED (histogene/hist2st/stnet/triplex/bleep + the new deepspace/mlpprobe); only the
data path is per-cohort (dataset/<COHORT>/splits + var_50genes.json). Cross-organ code untouched.

Usage:
  PYTHONPATH=STFlow python hest_baselines.py --model triplex --seed 1 --cohort all \
      --source_dataroot dataset --embed_dataroot embed_dataroot --save_root results_hest_baselines
"""
import os, json, glob, argparse
from operator import itemgetter
import numpy as np
import pandas as pd

import baseline_spatial as B
from MorphoST.evaluation import train_val_split
from stflow.utils import set_random_seed
from stflow.data.normalize_utils import get_normalize_method

HEST_COHORTS = ["CCRCC", "COAD", "READ", "HCC", "IDC", "LYMPH_IDC",
                "LUNG", "PAAD", "PRAD", "SKCM"]


def run_cohort(args, cohort, device, nm):
    gene_list = json.load(open(os.path.join(args.source_dataroot, cohort, args.gene_list)))["genes"]
    split_dir = os.path.join(args.source_dataroot, cohort, "splits")
    n_folds = len(glob.glob(os.path.join(split_dir, "train_*.csv")))
    save_dir = os.path.join(args.save_root, f"{cohort}_{args.model}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    fold_means = []
    for i in range(n_folds):
        out = os.path.join(save_dir, f"fold_{i}_results.json")
        if os.path.isfile(out):
            r = json.load(open(out)); fold_means.append(r["pearson_mean"])
            print(f"=== {cohort} {args.model} fold {i} SKIP ({r['pearson_mean']:.4f}) ==="); continue
        outer_train_df = pd.read_csv(os.path.join(split_dir, f"train_{i}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{i}.csv"))
        train_df, val_df = train_val_split(outer_train_df, args.seed + i, args.val_fraction)
        train_slides = B.load_slides(train_df, args, gene_list, nm, args.n_pos, args.k, cohort=cohort)
        val_slides = B.load_slides(val_df, args, gene_list, nm, args.n_pos, args.k, cohort=cohort)
        test_slides = B.load_slides(test_df, args, gene_list, nm, args.n_pos, args.k, cohort=cohort)
        fold_dir = os.path.join(save_dir, f"fold_{i}")
        if args.model == "bleep":
            res = B.bleep_train_fold(args, train_slides, val_slides, test_slides, gene_list,
                                     device, fold_dir)
        else:
            res = B.train_fold(args, train_slides, val_slides, test_slides, gene_list,
                               device, fold_dir)
        res["fold"] = i
        json.dump(res, open(out, "w"), sort_keys=True, indent=4)
        fold_means.append(res["pearson_mean"])
        print(f"=== {cohort} {args.model} fold {i} seed{args.seed}: pearson_mean={res['pearson_mean']:.4f} ===",
              flush=True)

    kfold = {"cohort": cohort, "pearson_mean": float(np.mean(fold_means)),
             "pearson_std": float(np.std(fold_means)), "mean_per_split": fold_means}
    json.dump(kfold, open(os.path.join(save_dir, "results_kfold.json"), "w"), sort_keys=True, indent=4)
    print(f"\n{cohort} {args.model} seed{args.seed}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"(per-fold {[round(x, 4) for x in fold_means]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["histogene", "hist2st", "stnet", "deepspace", "mlpprobe", "triplex", "bleep",
                            "mctogene", "hist", "histogpa", "hyperst", "genedml", "m2ost"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cohort", default="all")
    p.add_argument("--source_dataroot", default="dataset")
    p.add_argument("--embed_dataroot", default="embed_dataroot")
    p.add_argument("--feature_encoder", default="uni_v1_official")
    p.add_argument("--gene_list", default="var_50genes.json")
    p.add_argument("--save_root", default="results_hest_baselines")
    p.add_argument("--normalize_method", default="log1p")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--depth2", type=int, default=4)
    p.add_argument("--depth3", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_pos", type=int, default=128)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max_spots", type=int, default=4000)
    p.add_argument("--bleep_batch", type=int, default=512)
    p.add_argument("--k_retrieval", type=int, default=50)
    p.add_argument("--max_ref", type=int, default=30000)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--corr_weight", type=float, default=0.0)
    # recent baselines (feature-matched)
    p.add_argument("--alignment_beta", type=float, default=0.2)   # HyperST hyperbolic alignment weight
    p.add_argument("--entail_weight", type=float, default=0.4)    # HyperST entailment term
    p.add_argument("--aux_weight", type=float, default=0.25)      # Gene-DML per-stream metric weight
    args = p.parse_args()
    import torch
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    args.feature_dim = {"uni_v1_official": 1024, "gigapath": 1536, "resnet50_trunc": 1024}[args.feature_encoder]
    nm = get_normalize_method(args.normalize_method)
    cohorts = HEST_COHORTS if args.cohort == "all" else [args.cohort]
    for c in cohorts:
        run_cohort(args, c, device, nm)


if __name__ == "__main__":
    main()
