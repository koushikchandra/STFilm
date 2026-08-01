"""Foundation-model linear/RF probe baseline on the cross-organ splits.

Apple-to-apple with the STFlow runs: same UNI embeddings, same per-fold gene panel,
same log1p normalization, same pearson_mean metric (stflow.app.flow.test.metric_func).
The only thing that changes is the predictor: instead of the flow-matching SpatialTransformer,
we fit a simple per-fold regressor (RidgeCV or RandomForest) on the UNI patch embeddings.

Usage:
  PYTHONPATH=STFlow python probe_baseline.py --model ridge --splits_root cross_organ_splits8 \
      --save_root results_probe_uni8
"""
import os
import json
import argparse
from operator import itemgetter

import numpy as np
import pandas as pd

from stflow.data.normalize_utils import get_normalize_method
from stflow.hest_utils.st_dataset import load_adata
from stflow.hest_utils.file_utils import read_assets_from_h5
from stflow.app.flow.test import metric_func
from stflow.utils import merge_fold_results

COHORT_TO_GROUP = {
    "CCRCC": "kidney", "COAD": "colorectal", "READ": "colorectal", "HCC": "liver",
    "IDC": "breast", "LYMPH_IDC": "breast", "LUNG": "lung", "PAAD": "pancreas",
    "PRAD": "prostate", "SKCM": "skin",
}


def build_samples(df, args):
    samples = []
    for _, row in df.iterrows():
        cohort = row["patches_path"].split("/")[0]
        sid = row["sample_id"]
        samples.append({
            "name": sid,
            "cohort": cohort,
            "h5_path": os.path.join(args.embed_dataroot, cohort, args.feature_encoder, f"fp32/{sid}.h5"),
            "h5ad_path": os.path.join(args.source_dataroot, cohort, f"adata/{sid}.h5ad"),
        })
    return samples


def load_xy(samples, gene_list, normalize_method):
    """Concatenate UNI embeddings (X) and log1p expression (Y) across samples."""
    Xs, Ys = [], []
    for s in samples:
        data_dict, _ = read_assets_from_h5(s["h5_path"])
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        emb = data_dict["embeddings"].astype(np.float32)
        labels = load_adata(s["h5ad_path"], genes=gene_list, barcodes=barcodes,
                            normalize_method=normalize_method).values.astype(np.float32)
        Xs.append(emb)
        Ys.append(labels)
    return np.concatenate(Xs, 0), np.concatenate(Ys, 0)


def fit_predict(Xtr, Ytr, Xte, model, seed):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    if model == "ridge":
        from sklearn.linear_model import RidgeCV
        reg = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
    elif model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=100, max_features="sqrt",
                                    min_samples_leaf=5, n_jobs=-1, random_state=seed)
    else:
        raise ValueError(model)
    reg.fit(Xtr, Ytr)
    return reg.predict(Xte)


def run(args):
    regime_dir = os.path.join(args.splits_root, args.regime)
    split_dir = os.path.join(regime_dir, "splits")
    if args.regime == "POOLED":
        fold_names = [str(i) for i in range(5)]
    else:
        fold_names = list({"kidney", "liver", "lung", "pancreas",
                           "prostate", "skin", "breast", "colorectal"})
        fold_names = ["kidney", "liver", "lung", "pancreas", "prostate", "skin", "breast", "colorectal"]

    save_dir = os.path.join(args.save_root, f"{args.regime}_{args.model}")
    os.makedirs(save_dir, exist_ok=True)
    normalize_method = get_normalize_method(args.normalize_method)

    all_fold_results = []
    for fold in fold_names:
        fold_out = os.path.join(save_dir, f"fold_{fold}_results.json")
        if os.path.isfile(fold_out):
            print(f"=== {args.regime} fold {fold} ({args.model}) -> SKIP (exists) ===")
            with open(fold_out) as f:
                all_fold_results.append(json.load(f))
            continue
        train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        with open(os.path.join(regime_dir, f"genes_{fold}.json")) as f:
            gene_list = json.load(f)["genes"]

        Xtr, Ytr = load_xy(build_samples(train_df, args), gene_list, normalize_method)
        Xte, Yte = load_xy(build_samples(test_df, args), gene_list, normalize_method)
        pred = fit_predict(Xtr, Ytr, Xte, args.model, args.seed)

        res = metric_func(pred, Yte, gene_list)
        res["n_test"] = len(Yte)
        res["fold"] = fold
        with open(fold_out, "w") as f:
            json.dump(res, f, sort_keys=True, indent=4)
        all_fold_results.append(res)
        print(f"=== {args.regime} fold {fold} ({args.model}): "
              f"pearson_mean={res['pearson_mean']:.4f} (n={res['n_test']}, genes={len(gene_list)}) ===")

    kfold = merge_fold_results(all_fold_results)
    kfold["pearson_corrs"] = sorted(kfold["pearson_corrs"], key=itemgetter("mean"), reverse=True)
    with open(os.path.join(save_dir, "results_kfold.json"), "w") as f:
        json.dump(kfold, f, sort_keys=True, indent=4)
    print(f"\n{args.regime} {args.model}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"+/- {kfold['pearson_std']:.4f}  (per-fold {[round(x,4) for x in kfold['mean_per_split']]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regime", type=str, default="both", choices=["POOLED", "LOOO", "both"])
    p.add_argument("--model", type=str, default="ridge", choices=["ridge", "rf"])
    p.add_argument("--splits_root", type=str, default="cross_organ_splits8")
    p.add_argument("--source_dataroot", type=str, default="dataset")
    p.add_argument("--embed_dataroot", type=str, default="embed_dataroot")
    p.add_argument("--feature_encoder", type=str, default="uni_v1_official")
    p.add_argument("--save_root", type=str, default="results_probe_uni8")
    p.add_argument("--normalize_method", type=str, default="log1p")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    regimes = ["POOLED", "LOOO"] if args.regime == "both" else [args.regime]
    for r in regimes:
        args.regime = r
        run(args)
