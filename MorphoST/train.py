"""Train/evaluate MorphoST on the leakage-safe cross-organ splits (HEST or STImage).

Mirrors train_cross_organ.py's fold structure, metric (nan-safe per-gene Pearson), and result
JSON layout so the existing aggregators/significance scripts work unchanged -- but the model is
MorphoST (no STFlow). One-seed HEST LOOO is the go/no-go test the plan calls for.

Example (V5, HEST LOOO, seed 1):
  PYTHONPATH=../STFlow python train.py --regime LOOO --version V5 --seed 1 \
      --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
      --embed_dataroot ../embed_dataroot --save_root results_morphost
"""
import os, json, argparse, random
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from morphost import MorphoST, morphost_loss
from data import load_sample, load_gene_list
from evaluation import expression_metrics, save_predictions, train_val_split

LOOO_FOLDS = ["kidney", "liver", "lung", "pancreas", "prostate", "skin", "breast", "colorectal"]
POOLED_FOLDS = ["0", "1", "2", "3", "4"]
STIMAGE_LOOO = ["breast", "kidney", "liver", "pancreas", "skin", "prostate", "brain", "heart"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def metric_func(preds, y, genes):
    """Per-gene Pearson averaged with nan-safety (constant target genes -> nan, excluded)."""
    pcc = []
    for g in range(y.shape[1]):
        if np.std(y[:, g]) < 1e-8 or np.std(preds[:, g]) < 1e-8:
            pcc.append(np.nan)
        else:
            pcc.append(pearsonr(preds[:, g], y[:, g])[0])
    pcc = np.array(pcc, dtype=float)
    return {"pearson_mean": float(np.nanmean(pcc)), "pearson_std": float(np.nanstd(pcc)),
            "pearson_per_gene": pcc.tolist()}


def subsample(feats, coords, expr, max_spots):
    N = feats.size(0)
    if max_spots and N > max_spots:
        idx = torch.randperm(N, device=feats.device)[:max_spots]
        return feats[idx], coords[idx], expr[idx]
    return feats, coords, expr


@torch.no_grad()
def evaluate(model, test_rows, args, gene_list, prediction_path=None):
    model.eval()
    preds_all, y_all, coords_all, slide_ids = [], [], [], []
    for _, row in test_rows.iterrows():
        feats, coords, expr = load_sample(row, args.feature_encoder, args.embed_dataroot,
                                          args.source_dataroot, gene_list, args.normalize_method,
                                          args.device)
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


def train_fold(args, train_df, val_df, test_df, gene_list, fold_dir):
    model = MorphoST(feat_dim=1024, dim=args.dim, n_genes=len(gene_list), n_layers=args.n_layers,
                     n_heads=args.n_heads, k=args.k, dropout=args.dropout,
                     num_rbf=args.num_rbf,
                     attn_dropout=args.dropout, version=args.version,
                     morph_scope=args.morph_scope, cond_dropout=args.cond_dropout,
                     components=args.components, use_distance_bias=args.use_distance_bias,
                     coordinate_mode=args.coordinate_mode).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # cache training samples on GPU-adjacent CPU once (reload per epoch to allow resampling)
    train_rows = list(train_df.iterrows())
    best, best_state = -1e9, None
    patience = args.patience; bad = 0
    for ep in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_rows)
        tot = 0.0
        for _, row in train_rows:
            feats, coords, expr = load_sample(row, args.feature_encoder, args.embed_dataroot,
                                              args.source_dataroot, gene_list,
                                              args.normalize_method, args.device)
            feats, coords, expr = subsample(feats, coords, expr, args.max_spots)
            pred = model(feats, coords)
            loss = morphost_loss(pred, expr, args.corr_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step()
        if ep % args.eval_step == 0 or ep == args.epochs:
            res = evaluate(model, val_df, args, gene_list)
            score = res[args.selection_metric]
            if score > best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            print(f"  ep{ep:3d} loss {tot/len(train_rows):.4f}  val_PCC {res['pearson_mean']:.4f}"
                  f"  (best {best:.4f})", flush=True)
            if bad >= patience:
                print(f"  early stop @ ep{ep}"); break
    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "args": vars(args), "genes": gene_list},
               os.path.join(fold_dir, "best_model.pt"))
    return evaluate(model, test_df, args, gene_list,
                    os.path.join(fold_dir, "test_predictions.npz"))


def run(args):
    set_seed(args.seed)
    if args.folds:
        folds = args.folds
    elif args.organ_set == "stimage":
        folds = STIMAGE_LOOO if args.regime == "LOOO" else POOLED_FOLDS
    else:
        folds = LOOO_FOLDS if args.regime == "LOOO" else POOLED_FOLDS
    model_tag = f"C{args.components}" if args.components else args.version
    tag = args.exp_code or f"{args.regime}_{model_tag}_seed{args.seed}"
    save_dir = os.path.join(args.save_root, tag); os.makedirs(save_dir, exist_ok=True)
    split_dir = os.path.join(args.splits_root, args.regime, "splits")

    all_res = []
    for fold in folds:
        fout = os.path.join(save_dir, f"fold_{fold}_results.json")
        if os.path.isfile(fout):
            all_res.append(json.load(open(fout))); print(f"=== fold {fold} SKIP ==="); continue
        print(f"\n=== {args.regime} fold {fold} ({args.version}, seed{args.seed}) ===", flush=True)
        outer_train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        train_df, val_df = train_val_split(outer_train_df, args.seed + sum(map(ord, str(fold))),
                                           args.val_fraction)
        gene_list = load_gene_list(args.splits_root, args.regime, fold)
        fold_dir = os.path.join(save_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        res = train_fold(args, train_df, val_df, test_df, gene_list, fold_dir)
        res["fold"] = fold
        res["n_train_slides"] = len(train_df); res["n_val_slides"] = len(val_df)
        res["n_test_slides"] = len(test_df)
        json.dump(res, open(fout, "w"))
        all_res.append(res)

    means = [r["pearson_mean"] for r in all_res]
    kfold = {"pearson_mean": float(np.mean(means)), "pearson_std": float(np.std(means)),
             "mean_per_split": means}
    json.dump(kfold, open(os.path.join(save_dir, "results_kfold.json"), "w"))
    print(f"\n{args.regime} {args.version}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"+/- {kfold['pearson_std']:.4f}  (per-fold {[round(x,4) for x in means]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=["LOOO", "POOLED"])
    p.add_argument("--version", default="V5", choices=["V0", "V1", "V2", "V3", "V4", "V5"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--organ_set", default="hest", choices=["hest", "stimage"])
    p.add_argument("--folds", nargs="+", default=None,
                   help="explicit fold names (overrides --organ_set defaults)")
    p.add_argument("--splits_root", default="../cross_organ_splits8")
    p.add_argument("--source_dataroot", default="../dataset")
    p.add_argument("--embed_dataroot", default="../embed_dataroot")
    p.add_argument("--feature_encoder", default="uni_v1_official")
    p.add_argument("--save_root", default="results_morphost")
    p.add_argument("--exp_code", default=None)
    p.add_argument("--normalize_method", default="log1p")
    p.add_argument("--device", default="cuda")
    # model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--num_rbf", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    # optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval_step", type=int, default=1)
    p.add_argument("--corr_weight", type=float, default=0.5)
    p.add_argument("--max_spots", type=int, default=3000)
    p.add_argument("--morph_scope", default="both", choices=["both", "global", "local"],
                   help="which morphology descriptor(s) feed the conditioner (V4/V5 only)")
    p.add_argument("--cond_dropout", type=float, default=0.0,
                   help="dropout on the conditioner (regularizes morph-conditioning)")
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--selection_metric", default="pearson_mean", choices=["pearson_mean", "spearman_mean"])
    p.add_argument("--components", default=None,
                   help="factorial LOCAL/GLOBAL/SLIDE bit mask, e.g. 101; overrides version components")
    p.add_argument("--use_distance_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coordinate_mode", default="distance", choices=["distance", "absolute"])
    args = p.parse_args()
    if not torch.cuda.is_available():
        args.device = "cpu"
    run(args)


if __name__ == "__main__":
    main()
