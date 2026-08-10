"""Conditioning-is-general experiment: add STFiLM-style descriptor FiLM (desc / local) ON TOP OF the
feature-matched baseline backbones (HisToGene, Hist2ST, TRIPLEX), to show the conditioning gain is
additive and not specific to STFlow. This file is purely additive: it imports the backbones from
baseline_spatial.py (unmodified) and writes to a SEPARATE results root (results_spatial_film_uni8),
so every prior file and result is untouched. The `none` reference for each backbone is its existing
run in results_spatial_uni8.

FiLM here mirrors STFiLM: a zero-initialized (identity at init) per-token shift/scale computed from
an inference-available UNI descriptor --- global slide mean (desc) or per-spot kNN mean (local) ---
injected right after the backbone's input projection. desc/local are permutation- and
SE(2)-invariant, matching the validity rule (available for unseen organs; never the target).

Usage:
  PYTHONPATH=STFlow python baseline_film.py --model triplex --film local --regime LOOO --seed 1 \
      --splits_root cross_organ_splits8 --save_root results_spatial_film_uni8 --device 0
"""
import os
import json
import glob
import argparse
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import baseline_spatial as B
from stflow.utils import set_random_seed, merge_fold_results
from stflow.data.normalize_utils import get_normalize_method


# ----------------------------- FiLM conditioner -----------------------------
class FiLMCond(nn.Module):
    """Descriptor-based FiLM: per-token (1+gamma), beta from an invariant UNI descriptor.
    Zero-initialized output layer -> identity at init, so the conditioned model starts exactly at
    its unconditioned backbone (training-stable, dormant conditioning)."""
    def __init__(self, fdim, dim, mode):
        super().__init__()
        self.mode = mode
        if mode != "none":
            self.proj = nn.Linear(fdim, dim)
            self.to_ss = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.to_ss.weight)
            nn.init.zeros_(self.to_ss.bias)

    def forward(self, x, feat, adj):  # x:[N,dim] hidden, feat:[N,fdim] UNI, adj:[N,N]
        if self.mode == "none":
            return x
        if self.mode == "desc":
            d = feat.mean(0, keepdim=True).expand_as(feat)       # global slide descriptor
        else:  # local
            mask = adj / adj.sum(1, keepdim=True).clamp(min=1)
            d = mask.mm(feat)                                    # per-spot kNN-mean descriptor
        gamma, beta = self.to_ss(F.gelu(self.proj(d))).chunk(2, dim=-1)
        return x * (1 + gamma) + beta


# ----------------------------- FiLM-augmented backbones -----------------------------
class HistoGeneFiLM(B.HistoGeneNet):
    def __init__(self, fdim, dim, depth, heads, n_genes, n_pos, dropout, film):
        super().__init__(fdim, dim, depth, heads, n_genes, n_pos, dropout)
        self.filmc = FiLMCond(fdim, dim, film)

    def forward(self, feat, gxy, adj=None):
        x = self.proj(feat) + self.x_embed(gxy[:, 0]) + self.y_embed(gxy[:, 1])
        x = self.filmc(x, feat, adj)
        x = self.drop(x)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(x[0])


class Hist2STFiLM(B.Hist2STNet):
    def __init__(self, fdim, dim, depth2, depth3, heads, n_genes, n_pos, dropout, film):
        super().__init__(fdim, dim, depth2, depth3, heads, n_genes, n_pos, dropout)
        self.filmc = FiLMCond(fdim, dim, film)

    def forward(self, feat, gxy, adj):
        g = self.proj(feat) + self.x_embed(gxy[:, 0]) + self.y_embed(gxy[:, 1])
        g = self.filmc(g, feat, adj)
        g = g[None]
        for b in self.attn:
            g = b(g)
        g = g[0]
        jk = []
        for gs in self.gs:
            g = gs(g, adj)
            jk.append(g[None])
        g = torch.cat(jk, 0)
        g = self.jk(g)[0].mean(0)
        return self.head(g)


class TriplexFiLM(B.TriplexNet):
    def __init__(self, fdim, dim, heads, depth, n_genes, dropout, film):
        super().__init__(fdim, dim, heads, depth, n_genes, dropout)
        self.filmc = FiLMCond(fdim, dim, film)

    def forward(self, feat, gxy, adj):
        mask = adj / adj.sum(1, keepdim=True).clamp(min=1)
        neigh_feat = mask.mm(feat)
        glob_feat = feat.mean(0, keepdim=True).expand_as(feat)
        s = self.spot(feat) + self.res_embed[0]
        s = self.filmc(s, feat, adj)                 # condition the spot-resolution stream
        nb = self.neigh(neigh_feat) + self.res_embed[1]
        gl = self.glob(glob_feat) + self.res_embed[2]
        x = torch.stack([s, nb, gl], dim=1)
        for b in self.blocks:
            x = b(x)
        return self.head(x[:, 0])


def build_model(args, n_genes):
    if args.model == "histogene":
        return HistoGeneFiLM(args.feature_dim, args.dim, args.depth, args.heads,
                             n_genes, args.n_pos, args.dropout, args.film)
    if args.model == "hist2st":
        return Hist2STFiLM(args.feature_dim, args.dim, args.depth2, args.depth3, args.heads,
                           n_genes, args.n_pos, args.dropout, args.film)
    if args.model == "triplex":
        return TriplexFiLM(args.feature_dim, args.dim, args.heads, args.depth,
                           n_genes, args.dropout, args.film)
    raise ValueError(f"unsupported backbone for FiLM: {args.model}")


# ----------------------------- train / eval (reuses baseline_spatial) -----------------------------
def train_fold(args, train_slides, test_slides, gene_list, device):
    model = build_model(args, len(gene_list)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_pearson, best_res, early = -1, None, 0
    order = list(range(len(train_slides)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(order)
        for i in order:
            s = train_slides[i]
            feat, gxy, adj, lab = s["feat"], s["gxy"], s["adj"], s["labels"]
            if feat.shape[0] > args.max_spots:
                sel = torch.randperm(feat.shape[0])[:args.max_spots]
                feat, gxy, lab = feat[sel], gxy[sel], lab[sel]
                adj = adj[sel][:, sel]
            feat = feat.to(device); gxy = gxy.to(device); adj = adj.to(device); lab = lab.to(device)
            pred = model(feat, gxy, adj)
            loss = F.mse_loss(pred, lab)
            opt.zero_grad(); loss.backward(); opt.step()
        res = B.evaluate(model, test_slides, gene_list, device)
        if res["pearson_mean"] > best_pearson:
            best_pearson, best_res, early = res["pearson_mean"], res, 0
        else:
            early += 1
            if early >= 20:
                break
    return best_res


def run(args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    args.feature_dim = {"uni_v1_official": 1024, "gigapath": 1536, "resnet50_trunc": 1024}[args.feature_encoder]
    regime_dir = os.path.join(args.splits_root, args.regime)
    split_dir = os.path.join(regime_dir, "splits")
    trains = glob.glob(os.path.join(split_dir, "train_*.csv"))
    fold_names = [os.path.basename(t)[len("train_"):-len(".csv")] for t in trains]
    fold_names.sort(key=lambda x: (int(x) if x.isdigit() else 1 << 30, x))

    tag = f"{args.regime}_{args.model}-{args.film}_seed{args.seed}"   # e.g. LOOO_triplex-local_seed1
    save_dir = os.path.join(args.save_root, tag)
    os.makedirs(save_dir, exist_ok=True)
    nm = get_normalize_method(args.normalize_method)

    all_res = []
    for fold in fold_names:
        out = os.path.join(save_dir, f"fold_{fold}_results.json")
        if os.path.isfile(out):
            print(f"=== {tag} fold {fold} -> SKIP ===")
            all_res.append(json.load(open(out))); continue
        train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        gene_list = json.load(open(os.path.join(regime_dir, f"genes_{fold}.json")))["genes"]
        train_slides = B.load_slides(train_df, args, gene_list, nm, args.n_pos, args.k)
        test_slides = B.load_slides(test_df, args, gene_list, nm, args.n_pos, args.k)
        res = train_fold(args, train_slides, test_slides, gene_list, device)
        res["fold"] = fold
        json.dump(res, open(out, "w"), sort_keys=True, indent=4)
        all_res.append(res)
        print(f"=== {tag} fold {fold}: pearson_mean={res['pearson_mean']:.4f} ===")

    kfold = merge_fold_results(all_res)
    kfold["pearson_corrs"] = sorted(kfold["pearson_corrs"], key=itemgetter("mean"), reverse=True)
    json.dump(kfold, open(os.path.join(save_dir, "results_kfold.json"), "w"), sort_keys=True, indent=4)
    print(f"\n{tag}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"(per-fold {[round(x,4) for x in kfold['mean_per_split']]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["histogene", "hist2st", "triplex"])
    p.add_argument("--film", required=True, choices=["none", "desc", "local"])
    p.add_argument("--regime", required=True, choices=["POOLED", "LOOO"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--splits_root", default="cross_organ_splits8")
    p.add_argument("--source_dataroot", default="dataset")
    p.add_argument("--embed_dataroot", default="embed_dataroot")
    p.add_argument("--feature_encoder", default="uni_v1_official")
    p.add_argument("--save_root", default="results_spatial_film_uni8")
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
    args = p.parse_args()
    run(args)
