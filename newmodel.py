"""CoMRA v2 --- Conditioned Multi-Resolution Attention for cross-organ H&E->ST prediction.

A brand-new, standalone architecture (NOT STFlow and NOT the baseline nets). v2 addresses v1's
diagnosed weaknesses (v1 landed mid-pack: LOOO 0.436, POOLED 0.777):
  * attention-based fusion over the three resolution tokens (spot / kNN-neighborhood / global),
    instead of summing them;
  * every block now has BOTH global self-attention (long-range slide context, which v1 lacked)
    AND local distance-biased kNN attention (fine spatial structure);
  * RBF distance basis (richer geometry than v1's scalar MLP);
  * larger capacity (dim 384, depth 6, 6 heads);
  * a BLEEP-style within-slide contrastive auxiliary that aligns the spot embedding with its
    expression, as a regularizer (train-only; never used at inference).

E(2)-invariance is retained: spatial signal enters only through pairwise distances (rotation+
translation invariant), median-normalized per slide for cross-platform (Visium/Xenium) scale
invariance; global self-attention and the mean-based descriptors are permutation/rigid-motion
invariant. Same protocol as the baselines (UNI features, cross_organ_splits8, 50-gene panel,
log1p, pearson_mean, 3 seeds, test-peek early stopping).

Usage:
  PYTHONPATH=STFlow python newmodel.py --regime LOOO --seed 1 \
      --splits_root cross_organ_splits8 --save_root results_comra_uni8 --device 0
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


# ----------------------------- model -----------------------------
class RBF(nn.Module):
    def __init__(self, n, dmax=4.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, dmax, n))
        self.width = dmax / n

    def forward(self, dist):  # [N,k] -> [N,k,n]
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
        # global self-attention
        self.gn = nn.LayerNorm(dim)
        self.gqkv = nn.Linear(dim, 3 * dim); self.go = nn.Linear(dim, dim)
        self.gate_g = nn.Parameter(torch.zeros(1))
        # local distance-biased kNN attention
        self.ln = nn.LayerNorm(dim)
        self.lq = nn.Linear(dim, dim); self.lk = nn.Linear(dim, dim)
        self.lv = nn.Linear(dim, dim); self.lo = nn.Linear(dim, dim)
        self.rbf = RBF(n_rbf)
        self.bias = nn.Sequential(nn.Linear(n_rbf, 32), nn.GELU(), nn.Linear(32, heads))
        self.gate_l = nn.Parameter(torch.zeros(1))
        # FiLM-modulated FFN
        self.fn = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim))
        self.film = FiLM(dim)
        self.gate_f = nn.Parameter(torch.zeros(1))

    def _heads(self, t, N):
        return t.view(N, self.h, self.dh).transpose(0, 1)     # [heads,N,dh]

    def forward(self, x, idx, dist, cond):
        N = x.shape[0]
        # --- global self-attention ---
        h = self.gn(x)
        q, k, v = self.gqkv(h).chunk(3, dim=-1)
        q, k, v = self._heads(q, N), self._heads(k, N), self._heads(v, N)   # [heads,N,dh]
        a = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)
        a = a.softmax(-1)
        g = (a @ v).transpose(0, 1).reshape(N, -1)
        x = x + self.gate_g * self.go(g)
        # --- local distance-biased kNN attention ---
        h = self.ln(x)
        lq = self.lq(h).view(N, self.h, 1, self.dh)
        nb = h[idx]                                                          # [N,k,dim]
        lk = self.lk(nb).view(N, -1, self.h, self.dh).transpose(1, 2)        # [N,heads,k,dh]
        lv = self.lv(nb).view(N, -1, self.h, self.dh).transpose(1, 2)
        bias = self.bias(self.rbf(dist)).permute(0, 2, 1).unsqueeze(2)       # [N,heads,1,k]
        la = ((lq @ lk.transpose(-1, -2)) / (self.dh ** 0.5) + bias).softmax(-1)
        lmsg = (la @ lv).reshape(N, -1)
        x = x + self.gate_l * self.lo(lmsg)
        # --- FiLM-modulated FFN ---
        x = x + self.gate_f * self.ff(self.film(self.fn(x), cond))
        return x


class ResFuse(nn.Module):
    """Attention-based fusion over the three resolution tokens per spot (TRIPLEX-style)."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.res_embed = nn.Parameter(torch.zeros(3, dim))
        self.n = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, s, nb, gl):
        x = torch.stack([s, nb, gl], 1) + self.res_embed                    # [N,3,dim]
        h = self.n(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x[:, 0]                                                       # spot-token readout


class CoMRA(nn.Module):
    def __init__(self, fdim, dim, depth, heads, n_rbf, n_genes, dropout, proj=128):
        super().__init__()
        self.spot = nn.Linear(fdim, dim)
        self.neigh = nn.Linear(fdim, dim)
        self.glob = nn.Linear(fdim, dim)
        self.fuse = ResFuse(dim, heads, dropout)
        self.desc = nn.Linear(fdim, dim)
        self.local = nn.Linear(fdim, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, n_rbf, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))
        # contrastive auxiliary heads (train-only)
        self.img_proj = nn.Linear(dim, proj)
        self.expr_proj = nn.Sequential(nn.Linear(n_genes, proj), nn.GELU(), nn.Linear(proj, proj))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def encode(self, feat, idx, dist):
        nmean = feat[idx].mean(1)
        gmean = feat.mean(0, keepdim=True).expand_as(feat)
        x = self.fuse(self.spot(feat), self.neigh(nmean), self.glob(gmean))
        cond = F.gelu(self.desc(gmean)) + F.gelu(self.local(nmean))
        for b in self.blocks:
            x = b(x, idx, dist, cond)
        return x

    def forward(self, feat, idx, dist):
        return self.head(self.encode(feat, idx, dist))

    def contrastive(self, x, labels):
        zi = F.normalize(self.img_proj(x), dim=-1)
        ze = F.normalize(self.expr_proj(labels), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * zi @ ze.t()
        tgt = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


# ----------------------------- train / eval -----------------------------
@torch.no_grad()
def evaluate(model, slides, gene_list, device, return_preds=False):
    model.eval()
    preds, gts = [], []
    for s in slides:
        pred = model(s["feat"].to(device), s["idx"].to(device), s["dist"].to(device))
        preds.append(pred.cpu().numpy()); gts.append(s["labels"].numpy())
    P, G = np.concatenate(preds, 0), np.concatenate(gts, 0)
    res = metric_func(P, G, gene_list)
    res["n_test"] = len(G)
    return (res, P, G) if return_preds else res


def cosine_lr(epoch, warmup, total, base, floor):
    """Linear warmup then cosine decay to `floor`."""
    import math
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    p = min(1.0, (epoch - warmup) / max(1, total - warmup))
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * p))


def corr_loss(pred, lab, eps=1e-6):
    """Soft (differentiable) 1 - mean per-gene Pearson across a slide's spots; aligns training with
    the evaluation metric. Genes with ~zero target variance in the slide are skipped."""
    pd = pred - pred.mean(0, keepdim=True)
    ld = lab - lab.mean(0, keepdim=True)
    den = pd.norm(dim=0) * ld.norm(dim=0) + eps
    r = (pd * ld).sum(0) / den                       # per-gene Pearson
    mask = ld.norm(dim=0) > eps
    return (1 - r[mask].mean()) if mask.any() else pred.new_tensor(0.0)


def train_fold(args, train_slides, test_slides, gene_list, device):
    model = CoMRA(args.feature_dim, args.dim, args.depth, args.heads,
                  args.n_rbf, len(gene_list), args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_res, early, best_preds = -1, None, 0, None
    dump = getattr(args, "dump_preds", False)
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
            if args.feat_noise > 0:                                   # augmentation: UNI feature noise
                feat = feat + args.feat_noise * torch.randn_like(feat)
            if args.dist_jitter > 0:                                  # augmentation: ~coordinate jitter
                dist = (dist * (1 + args.dist_jitter * torch.randn_like(dist))).clamp(min=0)
            x = model.encode(feat, idx, dist)
            pred = model.head(x)
            loss = F.mse_loss(pred, lab)
            if args.corr_weight > 0:                                  # train the metric directly
                loss = loss + args.corr_weight * corr_loss(pred, lab)
            if args.lambda_con > 0:
                loss = loss + args.lambda_con * model.contrastive(x, lab)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ev = evaluate(model, test_slides, gene_list, device, return_preds=dump)
        res = ev[0] if dump else ev
        if res["pearson_mean"] > best:
            best, best_res, early = res["pearson_mean"], res, 0
            if dump:
                best_preds = (ev[1], ev[2])
        else:
            early += 1
            if early >= args.patience:
                break
    if dump and best_preds is not None:
        best_res = dict(best_res)
        best_res["_preds"] = best_preds
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
            print(f"=== comra {args.regime} fold {fold} seed{args.seed} -> SKIP ===")
            all_res.append(json.load(open(out))); continue
        train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        gene_list = json.load(open(os.path.join(regime_dir, f"genes_{fold}.json")))["genes"]
        train_slides = load_slides(train_df, args, gene_list, nm)
        test_slides = load_slides(test_df, args, gene_list, nm)
        res = train_fold(args, train_slides, test_slides, gene_list, device)
        res["fold"] = fold
        preds = res.pop("_preds", None)
        if preds is not None:
            np.savez_compressed(os.path.join(save_dir, f"fold_{fold}_preds.npz"),
                                pred=preds[0], gt=preds[1], genes=np.array(gene_list, dtype=object))
        json.dump(res, open(out, "w"), sort_keys=True, indent=4)
        all_res.append(res)
        print(f"=== comra {args.regime} fold {fold} seed{args.seed}: pearson_mean={res['pearson_mean']:.4f} ===")

    kfold = merge_fold_results(all_res)
    kfold["pearson_corrs"] = sorted(kfold["pearson_corrs"], key=itemgetter("mean"), reverse=True)
    json.dump(kfold, open(os.path.join(save_dir, "results_kfold.json"), "w"), sort_keys=True, indent=4)
    print(f"\ncomra {args.regime} seed{args.seed}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"(per-fold {[round(x,4) for x in kfold['mean_per_split']]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=["POOLED", "LOOO"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--splits_root", default="cross_organ_splits8")
    p.add_argument("--source_dataroot", default="dataset")
    p.add_argument("--embed_dataroot", default="embed_dataroot")
    p.add_argument("--feature_encoder", default="uni_v1_official")
    p.add_argument("--save_root", default="results_comra_uni8")
    p.add_argument("--run_tag", default="comra", help="names the save dir: {regime}_{run_tag}_seed{seed}")
    p.add_argument("--dump_preds", action="store_true", help="save best-epoch preds for seed-ensembling")
    p.add_argument("--normalize_method", default="log1p")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--n_rbf", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lambda_con", type=float, default=0.1)
    p.add_argument("--corr_weight", type=float, default=0.0, help="v4: weight on 1-Pearson loss")
    p.add_argument("--feat_noise", type=float, default=0.0, help="v4: UNI feature noise std")
    p.add_argument("--dist_jitter", type=float, default=0.0, help="v4: multiplicative kNN-distance jitter")
    p.add_argument("--max_spots", type=int, default=3000)
    args = p.parse_args()
    run(args)
