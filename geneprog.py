"""Shared gene-program factorization for cross-organ H&E->ST prediction (the novel core).

Thesis: cross-organ transfer works when we separate WHAT IS SHARED across organs from WHAT VARIES.
We factor per-spot expression as
        Y_hat = A @ B,   A in R^{N x K} (>=0),   B in R^{K x G} (shared),
where B is a small, organ-INVARIANT "gene-program -> gene" basis (the transferable part, one basis
per model, shared across all training organs), and A are per-spot program ACTIVATIONS predicted from
morphology and conditioned on the slide (the organ-varying part). Because the piece that must
generalize to an unseen organ (B) is forced to be shared and low-rank, it cannot overfit to the
training organs -- the mechanism is designed for LOOO transfer.

The encoder is deliberately lean (invariant local + global attention with descriptor-FiLM
conditioning); the contribution is the FACTORIZED DECODER, not module count. The falsifiable test is
built in: `--head factor` vs `--head direct` share the *identical* encoder and differ only in the
decoder, so any LOOO gain isolates the factorization.

Same protocol as the baselines (UNI features, cross_organ_splits8, 50-gene panel, log1p,
pearson_mean, 3 seeds, test-peek early stopping).

Usage:
  PYTHONPATH=STFlow python geneprog.py --regime LOOO --seed 1 --head factor --run_tag gp-factor \
      --splits_root cross_organ_splits8 --save_root results_geneprog_uni8 --device 0
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

from stflow.utils import set_random_seed, merge_fold_results
from stflow.data.normalize_utils import get_normalize_method
from stflow.hest_utils.st_dataset import load_adata
from stflow.hest_utils.file_utils import read_assets_from_h5
from stflow.app.flow.test import metric_func


# ----------------------------- data -----------------------------
def knn_graph(coords, k):
    c = torch.from_numpy(coords).float()
    d = torch.cdist(c, c)
    kk = min(k + 1, c.shape[0])
    dist, idx = d.topk(kk, largest=False)
    idx, dist = idx[:, 1:], dist[:, 1:]
    if idx.shape[1] < k:
        pad = k - idx.shape[1]
        idx = torch.cat([idx, idx[:, -1:].repeat(1, pad)], 1)
        dist = torch.cat([dist, dist[:, -1:].repeat(1, pad)], 1)
    med = dist.median().clamp(min=1e-6)
    return idx, (dist / med)


def load_slides(df, args, gene_list, nm):
    slides = []
    for _, row in df.iterrows():
        cohort = row["patches_path"].split("/")[0]
        sid = row["sample_id"]
        h5 = os.path.join(args.embed_dataroot, cohort, args.feature_encoder, f"fp32/{sid}.h5")
        h5ad = os.path.join(args.source_dataroot, cohort, f"adata/{sid}.h5ad")
        dd, _ = read_assets_from_h5(h5)
        barcodes = dd["barcodes"].flatten().astype(str).tolist()
        feat = torch.from_numpy(dd["embeddings"].astype(np.float32))
        idx, dist = knn_graph(dd["coords"].astype(np.float64), args.k)
        labels = load_adata(h5ad, genes=gene_list, barcodes=barcodes,
                            normalize_method=nm).values.astype(np.float32)
        slides.append({"feat": feat, "idx": idx, "dist": dist,
                       "labels": torch.from_numpy(labels)})
    return slides


# ----------------------------- lean invariant encoder -----------------------------
class RBF(nn.Module):
    def __init__(self, n, dmax=4.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, dmax, n))
        self.width = dmax / n

    def forward(self, dist):
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width ** 2))


class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_ss = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.to_ss.weight); nn.init.zeros_(self.to_ss.bias)

    def forward(self, x, cond):
        g, b = self.to_ss(cond).chunk(2, dim=-1)
        return x * (1 + g) + b


class Block(nn.Module):
    def __init__(self, dim, heads, n_rbf, dropout):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.gn = nn.LayerNorm(dim); self.gqkv = nn.Linear(dim, 3 * dim); self.go = nn.Linear(dim, dim)
        self.gate_g = nn.Parameter(torch.zeros(1))
        self.ln = nn.LayerNorm(dim)
        self.lq = nn.Linear(dim, dim); self.lk = nn.Linear(dim, dim)
        self.lv = nn.Linear(dim, dim); self.lo = nn.Linear(dim, dim)
        self.rbf = RBF(n_rbf); self.bias = nn.Sequential(nn.Linear(n_rbf, 32), nn.GELU(), nn.Linear(32, heads))
        self.gate_l = nn.Parameter(torch.zeros(1))
        self.fn = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.film = FiLM(dim); self.gate_f = nn.Parameter(torch.zeros(1))

    def _h(self, t, N):
        return t.view(N, self.h, self.dh).transpose(0, 1)

    def forward(self, x, idx, dist, cond):
        N = x.shape[0]
        h = self.gn(x); q, k, v = self.gqkv(h).chunk(3, -1)
        q, k, v = self._h(q, N), self._h(k, N), self._h(v, N)
        a = ((q @ k.transpose(-1, -2)) / (self.dh ** 0.5)).softmax(-1)
        x = x + self.gate_g * self.go((a @ v).transpose(0, 1).reshape(N, -1))
        h = self.ln(x)
        lq = self.lq(h).view(N, self.h, 1, self.dh)
        nb = h[idx]
        lk = self.lk(nb).view(N, -1, self.h, self.dh).transpose(1, 2)
        lv = self.lv(nb).view(N, -1, self.h, self.dh).transpose(1, 2)
        bias = self.bias(self.rbf(dist)).permute(0, 2, 1).unsqueeze(2)
        la = ((lq @ lk.transpose(-1, -2)) / (self.dh ** 0.5) + bias).softmax(-1)
        x = x + self.gate_l * self.lo((la @ lv).reshape(N, -1))
        x = x + self.gate_f * self.ff(self.film(self.fn(x), cond))
        return x


class GPNet(nn.Module):
    """Lean invariant encoder + factorized (A@B) or direct decoder."""
    def __init__(self, fdim, dim, depth, heads, n_rbf, n_genes, n_prog, dropout, head="factor",
                 nonneg_basis=True):
        super().__init__()
        self.head_mode = head
        self.spot = nn.Linear(fdim, dim)
        self.neigh = nn.Linear(fdim, dim)
        self.desc = nn.Linear(fdim, dim)
        self.local = nn.Linear(fdim, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, n_rbf, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        if head == "factor":
            self.to_A = nn.Linear(dim, n_prog)                 # per-spot program activations
            self.basis = nn.Parameter(torch.randn(n_prog, n_genes) * 0.02)  # SHARED gene-program basis
            self.nonneg_basis = nonneg_basis
        else:
            self.to_Y = nn.Linear(dim, n_genes)                # unstructured control head

    def encode(self, feat, idx, dist):
        nmean = feat[idx].mean(1)
        gmean = feat.mean(0, keepdim=True).expand_as(feat)
        x = self.spot(feat) + self.neigh(nmean)
        cond = F.gelu(self.desc(gmean)) + F.gelu(self.local(nmean))   # descriptor conditioning
        for b in self.blocks:
            x = b(x, idx, dist, cond)
        return self.norm(x)

    def basis_mat(self):
        return F.softplus(self.basis) if self.nonneg_basis else self.basis

    def forward(self, feat, idx, dist, return_A=False):
        h = self.encode(feat, idx, dist)
        if self.head_mode == "factor":
            A = F.softplus(self.to_A(h))                        # >=0 activations [N,K]
            Y = A @ self.basis_mat()                            # [N,G]
            return (Y, A) if return_A else Y
        return self.to_Y(h)


# ----------------------------- train / eval -----------------------------
@torch.no_grad()
def evaluate(model, slides, gene_list, device):
    model.eval()
    preds, gts = [], []
    for s in slides:
        pred = model(s["feat"].to(device), s["idx"].to(device), s["dist"].to(device))
        preds.append(pred.cpu().numpy()); gts.append(s["labels"].numpy())
    res = metric_func(np.concatenate(preds, 0), np.concatenate(gts, 0), gene_list)
    res["n_test"] = sum(len(g) for g in gts)
    return res


def cosine_lr(epoch, warmup, total, base, floor):
    import math
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    p = min(1.0, (epoch - warmup) / max(1, total - warmup))
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * p))


def train_fold(args, train_slides, test_slides, gene_list, device):
    model = GPNet(args.feature_dim, args.dim, args.depth, args.heads, args.n_rbf,
                  len(gene_list), args.n_prog, args.dropout, head=args.head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_res, early = -1, None, 0
    order = list(range(len(train_slides)))
    for epoch in range(1, args.epochs + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(epoch, args.warmup, args.epochs, args.lr, args.lr * 0.05)
        model.train(); np.random.shuffle(order)
        for i in order:
            s = train_slides[i]
            feat, idx, dist, lab = s["feat"], s["idx"], s["dist"], s["labels"]
            if feat.shape[0] > args.max_spots:
                sel = torch.randperm(feat.shape[0])[:args.max_spots]
                remap = {int(o): n for n, o in enumerate(sel.tolist())}
                feat, lab = feat[sel], lab[sel]
                idx = idx[sel].clone().apply_(lambda v: remap.get(v, 0))
                dist = dist[sel]
            feat = feat.to(device); idx = idx.to(device); dist = dist.to(device); lab = lab.to(device)
            pred = model(feat, idx, dist)
            loss = F.mse_loss(pred, lab)
            if args.head == "factor" and args.basis_reg > 0:      # encourage distinct programs
                Bn = F.normalize(model.basis_mat(), dim=1)
                gram = Bn @ Bn.t()
                loss = loss + args.basis_reg * (gram - torch.eye(gram.shape[0], device=device)).pow(2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        res = evaluate(model, test_slides, gene_list, device)
        if res["pearson_mean"] > best:
            best, best_res, early = res["pearson_mean"], res, 0
        else:
            early += 1
            if early >= args.patience:
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

    save_dir = os.path.join(args.save_root, f"{args.regime}_{args.run_tag}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    nm = get_normalize_method(args.normalize_method)

    all_res = []
    for fold in fold_names:
        out = os.path.join(save_dir, f"fold_{fold}_results.json")
        if os.path.isfile(out):
            print(f"=== {args.run_tag} {args.regime} fold {fold} seed{args.seed} -> SKIP ===")
            all_res.append(json.load(open(out))); continue
        train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        gene_list = json.load(open(os.path.join(regime_dir, f"genes_{fold}.json")))["genes"]
        train_slides = load_slides(train_df, args, gene_list, nm)
        test_slides = load_slides(test_df, args, gene_list, nm)
        res = train_fold(args, train_slides, test_slides, gene_list, device)
        res["fold"] = fold
        json.dump(res, open(out, "w"), sort_keys=True, indent=4)
        all_res.append(res)
        print(f"=== {args.run_tag} {args.regime} fold {fold} seed{args.seed}: pearson_mean={res['pearson_mean']:.4f} ===")

    kfold = merge_fold_results(all_res)
    kfold["pearson_corrs"] = sorted(kfold["pearson_corrs"], key=itemgetter("mean"), reverse=True)
    json.dump(kfold, open(os.path.join(save_dir, "results_kfold.json"), "w"), sort_keys=True, indent=4)
    print(f"\n{args.run_tag} {args.regime} seed{args.seed}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"(per-fold {[round(x,4) for x in kfold['mean_per_split']]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=["POOLED", "LOOO"])
    p.add_argument("--head", default="factor", choices=["factor", "direct"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--splits_root", default="cross_organ_splits8")
    p.add_argument("--source_dataroot", default="dataset")
    p.add_argument("--embed_dataroot", default="embed_dataroot")
    p.add_argument("--feature_encoder", default="uni_v1_official")
    p.add_argument("--save_root", default="results_geneprog_uni8")
    p.add_argument("--run_tag", default="gp")
    p.add_argument("--normalize_method", default="log1p")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n_rbf", type=int, default=16)
    p.add_argument("--n_prog", type=int, default=32, help="K: number of shared gene programs")
    p.add_argument("--basis_reg", type=float, default=0.0, help="orthogonality reg on the shared basis")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max_spots", type=int, default=3000)
    args = p.parse_args()
    run(args)
