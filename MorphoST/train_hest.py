"""Evaluate MorphoST V3 under STFlow's PER-COHORT protocol (HEST-1k), to build a Table-1-style
comparison directly against STFlow's reported numbers.

This is the *within-organ* setup from the STFlow paper (NOT our cross-organ LOOO/POOLED, which lives
untouched in train.py):
  - one k-fold cross-validation per cohort, using dataset/<COHORT>/splits/{train,test}_<i>.csv
  - the cohort's own top-50 HVG panel dataset/<COHORT>/var_50genes.json
  - Pearson on top-50 genes after log1p; best-on-test-fold early stopping (patience 20), 3 seeds
  - report per-cohort mean over its folds; aggregate mean/std over seeds is done by the aggregator.

MorphoST V3 config is kept identical to the cross-organ V3 runs (version V3, dim 256, 4 layers,
4 heads, k 8) so the two experiments describe the *same* model on two protocols.

Example:
  PYTHONPATH=../STFlow python train_hest.py --version V3 --seed 1 --cohort PRAD \
      --source_dataroot ../dataset --embed_dataroot ../embed_dataroot --save_root results_hest
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
import torch

from morphost import MorphoST, morphost_loss
from data import load_sample
from train import set_seed, metric_func, subsample  # reuse cross-organ helpers unchanged
from evaluation import expression_metrics, save_predictions, train_val_split

HEST_COHORTS = ["CCRCC", "COAD", "READ", "HCC", "IDC", "LYMPH_IDC",
                "LUNG", "PAAD", "PRAD", "SKCM"]


@torch.no_grad()
def evaluate(model, test_rows, args, gene_list, cohort, prediction_path=None):
    model.eval()
    preds_all, y_all, coords_all, slide_ids = [], [], [], []
    for _, row in test_rows.iterrows():
        feats, coords, expr = load_sample(row, args.feature_encoder, args.embed_dataroot,
                                          args.source_dataroot, gene_list, args.normalize_method,
                                          args.device, cohort=cohort)
        pred = model(feats, coords)
        preds_all.append(pred.cpu().numpy()); y_all.append(expr.cpu().numpy())
        coords_all.append(coords.cpu().numpy())
        slide_ids.extend([str(row["sample_id"])] * len(expr))
    pred = np.concatenate(preds_all); target = np.concatenate(y_all)
    coords = np.concatenate(coords_all); slide_ids = np.asarray(slide_ids)
    res = expression_metrics(pred, target, gene_list, slide_ids)
    if prediction_path:
        save_predictions(prediction_path, pred, target, coords, slide_ids, gene_list)
    return res


def train_fold(args, train_df, val_df, test_df, gene_list, cohort, fold_dir):
    model = MorphoST(feat_dim=1024, dim=args.dim, n_genes=len(gene_list), n_layers=args.n_layers,
                     n_heads=args.n_heads, k=args.k, dropout=args.dropout,
                     num_rbf=args.num_rbf,
                     attn_dropout=args.dropout, version=args.version,
                     morph_scope=args.morph_scope, cond_dropout=args.cond_dropout,
                     components=args.components, use_distance_bias=args.use_distance_bias,
                     coordinate_mode=args.coordinate_mode).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_rows = list(train_df.iterrows())
    best, best_state, bad = -1e9, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); np.random.shuffle(train_rows)
        tot = 0.0
        for _, row in train_rows:
            feats, coords, expr = load_sample(row, args.feature_encoder, args.embed_dataroot,
                                              args.source_dataroot, gene_list,
                                              args.normalize_method, args.device, cohort=cohort)
            feats, coords, expr = subsample(feats, coords, expr, args.max_spots)
            pred = model(feats, coords)
            loss = morphost_loss(pred, expr, args.corr_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step()
        if ep % args.eval_step == 0 or ep == args.epochs:
            res = evaluate(model, val_df, args, gene_list, cohort)
            score = res[args.selection_metric]
            if score > best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            print(f"  ep{ep:3d} loss {tot/len(train_rows):.4f}  val_PCC {res['pearson_mean']:.4f}"
                  f"  (best {best:.4f})", flush=True)
            if bad >= args.patience:
                print(f"  early stop @ ep{ep}"); break
    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "args": vars(args), "genes": gene_list},
               os.path.join(fold_dir, "best_model.pt"))
    return evaluate(model, test_df, args, gene_list, cohort,
                    os.path.join(fold_dir, "test_predictions.npz"))


def run_cohort(args, cohort):
    gene_list = json.load(open(os.path.join(args.source_dataroot, cohort, args.gene_list)))["genes"]
    split_dir = os.path.join(args.source_dataroot, cohort, "splits")
    n_folds = len(glob.glob(os.path.join(split_dir, "train_*.csv")))
    save_dir = os.path.join(args.save_root, f"{cohort}_{args.version}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    fold_means = []
    for i in range(n_folds):
        fout = os.path.join(save_dir, f"fold_{i}_results.json")
        if os.path.isfile(fout):
            r = json.load(open(fout)); fold_means.append(r["pearson_mean"])
            print(f"=== {cohort} fold {i} SKIP ({r['pearson_mean']:.4f}) ==="); continue
        print(f"\n=== {cohort} fold {i}/{n_folds} ({args.version}, seed{args.seed}) ===", flush=True)
        outer_train_df = pd.read_csv(os.path.join(split_dir, f"train_{i}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{i}.csv"))
        train_df, val_df = train_val_split(outer_train_df, args.seed + i, args.val_fraction)
        fold_dir = os.path.join(save_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        res = train_fold(args, train_df, val_df, test_df, gene_list, cohort, fold_dir)
        res["fold"] = i
        res["n_train_slides"] = len(train_df); res["n_val_slides"] = len(val_df)
        res["n_test_slides"] = len(test_df)
        json.dump(res, open(fout, "w")); fold_means.append(res["pearson_mean"])

    kfold = {"cohort": cohort, "pearson_mean": float(np.mean(fold_means)),
             "pearson_std": float(np.std(fold_means)), "mean_per_split": fold_means}
    json.dump(kfold, open(os.path.join(save_dir, "results_kfold.json"), "w"))
    print(f"\n{cohort} {args.version} seed{args.seed}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"(per-fold {[round(x, 4) for x in fold_means]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="V3", choices=["V0", "V1", "V2", "V3", "V4", "V5"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cohort", default="all", help="one HEST cohort or 'all'")
    p.add_argument("--source_dataroot", default="../dataset")
    p.add_argument("--embed_dataroot", default="../embed_dataroot")
    p.add_argument("--feature_encoder", default="uni_v1_official")
    p.add_argument("--gene_list", default="var_50genes.json")
    p.add_argument("--save_root", default="results_hest")
    p.add_argument("--normalize_method", default="log1p")
    p.add_argument("--device", default="cuda")
    # model (identical to cross-organ V3)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--num_rbf", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval_step", type=int, default=1)
    p.add_argument("--corr_weight", type=float, default=0.5)
    p.add_argument("--max_spots", type=int, default=3000)
    p.add_argument("--morph_scope", default="both", choices=["both", "global", "local"])
    p.add_argument("--cond_dropout", type=float, default=0.0)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--selection_metric", default="pearson_mean", choices=["pearson_mean", "spearman_mean"])
    p.add_argument("--components", default=None,
                   help="factorial LOCAL/GLOBAL/SLIDE bit mask, e.g. 101; overrides version components")
    p.add_argument("--use_distance_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coordinate_mode", default="distance", choices=["distance", "absolute"])
    args = p.parse_args()
    if not torch.cuda.is_available():
        args.device = "cpu"
    set_seed(args.seed)
    cohorts = HEST_COHORTS if args.cohort == "all" else [args.cohort]
    for c in cohorts:
        run_cohort(args, c)


if __name__ == "__main__":
    main()
