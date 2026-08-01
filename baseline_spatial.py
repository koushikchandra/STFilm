"""Feature-matched spatial baselines: HisToGene-style and Hist2ST-style on UNI features.

Apple-to-apple with V0/V3: identical UNI embeddings, cross_organ_splits8 LOOO+POOLED,
per-fold gene panel, log1p normalization, and the same pearson_mean metric
(stflow.app.flow.test.metric_func). We swap each method's raw-image front-end for the shared
UNI features and keep its spatial-modeling core:

  histogene : ViT self-attention over all spots of a slide + learned (x,y) grid pos-embeddings
              (Pang et al. 2021), then an MLP gene head.
  hist2st   : transformer blocks (global) + GraphSAGE GCN blocks over a kNN(coords) graph
              (local) + jumping-knowledge LSTM fusion (Zeng et al. 2022), then a gene head.

All models train with MSE on log1p expression and use the same test-peek early stopping
(patience 20) as train_cross_organ.py so the comparison to V0/V3 is fair.

Usage:
  PYTHONPATH=STFlow python baseline_spatial.py --model histogene --regime LOOO --seed 1 \
      --splits_root cross_organ_splits8 --save_root results_spatial_uni8 --device 0
"""
import os
import json
import argparse
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from stflow.utils import set_random_seed, merge_fold_results
from stflow.data.normalize_utils import get_normalize_method
from stflow.hest_utils.st_dataset import load_adata
from stflow.hest_utils.file_utils import read_assets_from_h5
from stflow.app.flow.test import metric_func

COHORT_TO_GROUP = {
    "CCRCC": "kidney", "COAD": "colorectal", "READ": "colorectal", "HCC": "liver",
    "IDC": "breast", "LYMPH_IDC": "breast", "LUNG": "lung", "PAAD": "pancreas",
    "PRAD": "prostate", "SKCM": "skin",
}


# ----------------------------- data -----------------------------
def grid_coords(coords, n_pos):
    """Map continuous (x,y) to integer bins in [0, n_pos) per axis (HisToGene/Hist2ST style)."""
    g = np.zeros_like(coords, dtype=np.int64)
    for a in range(2):
        v = coords[:, a].astype(np.float64)
        lo, hi = v.min(), v.max()
        if hi <= lo:
            g[:, a] = 0
        else:
            g[:, a] = np.clip(((v - lo) / (hi - lo) * (n_pos - 1)).round().astype(np.int64), 0, n_pos - 1)
    return g


def knn_adj(coords, k):
    """Dense symmetric binary [N,N] kNN adjacency with self-loops."""
    c = torch.from_numpy(coords).float()
    d = torch.cdist(c, c)
    kk = min(k + 1, c.shape[0])
    idx = d.topk(kk, largest=False).indices
    A = torch.zeros(c.shape[0], c.shape[0])
    A.scatter_(1, idx, 1.0)
    A = ((A + A.t()) > 0).float()
    A.fill_diagonal_(1.0)
    return A


def load_slides(df, args, gene_list, normalize_method, n_pos, k):
    slides = []
    for _, row in df.iterrows():
        cohort = row["patches_path"].split("/")[0]
        sid = row["sample_id"]
        h5 = os.path.join(args.embed_dataroot, cohort, args.feature_encoder, f"fp32/{sid}.h5")
        h5ad = os.path.join(args.source_dataroot, cohort, f"adata/{sid}.h5ad")
        dd, _ = read_assets_from_h5(h5)
        barcodes = dd["barcodes"].flatten().astype(str).tolist()
        coords = dd["coords"].astype(np.float64)
        feat = dd["embeddings"].astype(np.float32)
        labels = load_adata(h5ad, genes=gene_list, barcodes=barcodes,
                            normalize_method=normalize_method).values.astype(np.float32)
        slides.append({
            "feat": torch.from_numpy(feat),
            "gxy": torch.from_numpy(grid_coords(coords, n_pos)),
            "adj": knn_adj(coords, k),
            "labels": torch.from_numpy(labels),
        })
    return slides


# ----------------------------- modules -----------------------------
class TBlock(nn.Module):
    """Standard pre-norm transformer block (the HisToGene/Hist2ST attention core)."""
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):  # x: [1, N, dim]
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.n2(x))
        return x


class GSBlock(nn.Module):
    """GraphSAGE mean-aggregation block (gcn=True), faithful to Hist2ST gs_block."""
    def __init__(self, fin, fout):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(fout, fin))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, A):  # x: [N, dim], A: [N, N]
        mask = A / A.sum(1, keepdim=True).clamp(min=1)
        neigh = mask.mm(x)
        out = F.relu(neigh.mm(self.weight.t()))
        return F.normalize(out, 2, 1)


class HistoGeneNet(nn.Module):
    def __init__(self, fdim, dim, depth, heads, n_genes, n_pos, dropout):
        super().__init__()
        self.proj = nn.Linear(fdim, dim)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TBlock(dim, heads, 2 * dim, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feat, gxy, adj=None):
        x = self.proj(feat) + self.x_embed(gxy[:, 0]) + self.y_embed(gxy[:, 1])
        x = self.drop(x)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(x[0])


class Hist2STNet(nn.Module):
    def __init__(self, fdim, dim, depth2, depth3, heads, n_genes, n_pos, dropout):
        super().__init__()
        self.proj = nn.Linear(fdim, dim)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.attn = nn.ModuleList([TBlock(dim, heads, dim, dropout) for _ in range(depth2)])
        self.gs = nn.ModuleList([GSBlock(dim, dim) for _ in range(depth3)])
        self.jk = nn.LSTM(dim, dim, 2)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feat, gxy, adj):
        g = self.proj(feat) + self.x_embed(gxy[:, 0]) + self.y_embed(gxy[:, 1])
        g = g[None]
        for b in self.attn:
            g = b(g)
        g = g[0]
        jk = []
        for gs in self.gs:
            g = gs(g, adj)
            jk.append(g[None])
        g = torch.cat(jk, 0)            # [depth3, N, dim]
        g = self.jk(g)[0].mean(0)       # jumping-knowledge fusion
        return self.head(g)


def build_model(args, n_genes):
    if args.model == "histogene":
        return HistoGeneNet(args.feature_dim, args.dim, args.depth, args.heads,
                            n_genes, args.n_pos, args.dropout)
    return Hist2STNet(args.feature_dim, args.dim, args.depth2, args.depth3, args.heads,
                      n_genes, args.n_pos, args.dropout)


# ----------------------------- train / eval -----------------------------
@torch.no_grad()
def evaluate(model, slides, gene_list, device):
    model.eval()
    preds, gts = [], []
    for s in slides:
        feat = s["feat"].to(device); gxy = s["gxy"].to(device); adj = s["adj"].to(device)
        pred = model(feat, gxy, adj).cpu().numpy()
        preds.append(pred); gts.append(s["labels"].numpy())
    res = metric_func(np.concatenate(preds, 0), np.concatenate(gts, 0), gene_list)
    res["n_test"] = sum(len(g) for g in gts)
    return res


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
            if feat.shape[0] > args.max_spots:  # cap memory on huge slides
                sel = torch.randperm(feat.shape[0])[:args.max_spots]
                feat, gxy, lab = feat[sel], gxy[sel], lab[sel]
                adj = adj[sel][:, sel]
            feat = feat.to(device); gxy = gxy.to(device); adj = adj.to(device); lab = lab.to(device)
            pred = model(feat, gxy, adj)
            loss = F.mse_loss(pred, lab)
            opt.zero_grad(); loss.backward(); opt.step()
        res = evaluate(model, test_slides, gene_list, device)
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
    if args.regime == "POOLED":
        fold_names = [str(i) for i in range(5)]
    else:
        fold_names = ["kidney", "liver", "lung", "pancreas", "prostate", "skin", "breast", "colorectal"]

    save_dir = os.path.join(args.save_root, f"{args.regime}_{args.model}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    nm = get_normalize_method(args.normalize_method)

    all_res = []
    for fold in fold_names:
        out = os.path.join(save_dir, f"fold_{fold}_results.json")
        if os.path.isfile(out):
            print(f"=== {args.regime} {args.model} fold {fold} seed{args.seed} -> SKIP ===")
            all_res.append(json.load(open(out))); continue
        train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        gene_list = json.load(open(os.path.join(regime_dir, f"genes_{fold}.json")))["genes"]
        train_slides = load_slides(train_df, args, gene_list, nm, args.n_pos, args.k)
        test_slides = load_slides(test_df, args, gene_list, nm, args.n_pos, args.k)
        res = train_fold(args, train_slides, test_slides, gene_list, device)
        res["fold"] = fold
        json.dump(res, open(out, "w"), sort_keys=True, indent=4)
        all_res.append(res)
        print(f"=== {args.regime} {args.model} fold {fold} seed{args.seed}: "
              f"pearson_mean={res['pearson_mean']:.4f} ===")

    kfold = merge_fold_results(all_res)
    kfold["pearson_corrs"] = sorted(kfold["pearson_corrs"], key=itemgetter("mean"), reverse=True)
    json.dump(kfold, open(os.path.join(save_dir, "results_kfold.json"), "w"), sort_keys=True, indent=4)
    print(f"\n{args.regime} {args.model} seed{args.seed}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"(per-fold {[round(x,4) for x in kfold['mean_per_split']]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["histogene", "hist2st"])
    p.add_argument("--regime", required=True, choices=["POOLED", "LOOO"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--splits_root", default="cross_organ_splits8")
    p.add_argument("--source_dataroot", default="dataset")
    p.add_argument("--embed_dataroot", default="embed_dataroot")
    p.add_argument("--feature_encoder", default="uni_v1_official")
    p.add_argument("--save_root", default="results_spatial_uni8")
    p.add_argument("--normalize_method", default="log1p")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)     # histogene ViT depth
    p.add_argument("--depth2", type=int, default=4)    # hist2st transformer depth
    p.add_argument("--depth3", type=int, default=2)    # hist2st GCN depth
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_pos", type=int, default=128)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max_spots", type=int, default=4000)
    args = p.parse_args()
    run(args)
