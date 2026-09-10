"""Conditioning-is-general experiment: add STFiLM-style descriptor FiLM (desc / local) ON TOP OF
EVERY feature-matched baseline backbone, to test whether the conditioning gain is a general
property or specific to the STFlow flow-matching model. Covers all five baselines:

  regression backbones (full FiLM: none / desc / local):  histogene, hist2st, triplex, stnet
  retrieval backbone  (desc only, see caveat):             bleep

FiLM mirrors STFiLM exactly: a zero-initialized (identity at init) per-token shift/scale from an
inference-available UNI descriptor -- global slide mean (desc) or per-spot kNN mean (local) --
injected right after the backbone's input projection. Reuses backbones from the parent repo's
baseline_spatial.py (unmodified); writes to results_film_baselines/. The `none` reference is each
backbone re-run here (so the delta is same-code, same-seed, apples-to-apples).

BLEEP caveat: BLEEP predicts by retrieval over a pooled, cross-slide reference set, so it has no
per-spot hidden regression state and no per-slide graph in the pool. `local` FiLM is therefore not
well defined for it; we run only `desc` (a global image-descriptor modulation of the image head).

Usage:
  PYTHONPATH=../STFlow python film_baselines.py --model triplex --film local --regime LOOO --seed 1 \
      --splits_root ../cross_organ_splits8 --source_dataroot ../dataset --embed_dataroot ../embed_dataroot
"""
import os, sys, json, glob, argparse
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import baseline_spatial as B
from stflow.utils import set_random_seed
from stflow.data.normalize_utils import get_normalize_method
from MorphoST.evaluation import train_val_split   # nested-validation inner split (+ single-slide fallback)


# ----------------------------- FiLM conditioner -----------------------------
class FiLMCond(nn.Module):
    """Descriptor FiLM: per-token (1+gamma), beta from an invariant UNI descriptor; zero-init out
    layer -> identity at init, so the conditioned model starts exactly at its unconditioned base."""
    def __init__(self, fdim, dim, mode):
        super().__init__()
        self.mode = mode
        if mode != "none":
            self.proj = nn.Linear(fdim, dim)
            self.to_ss = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.to_ss.weight); nn.init.zeros_(self.to_ss.bias)

    def forward(self, x, feat, adj):
        if self.mode == "none":
            return x
        if self.mode == "desc" or adj is None:
            d = feat.mean(0, keepdim=True).expand_as(feat)          # global slide descriptor
        else:  # local
            mask = adj / adj.sum(1, keepdim=True).clamp(min=1)
            d = mask.mm(feat)                                       # per-spot kNN-mean descriptor
        gamma, beta = self.to_ss(F.gelu(self.proj(d))).chunk(2, dim=-1)
        return x * (1 + gamma) + beta


# ----------------------------- regression FiLM backbones -----------------------------
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
            g = gs(g, adj); jk.append(g[None])
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


class STNetFiLM(B.STNetNet):
    """ST-Net per-spot MLP + FiLM after the first hidden block. FiLM injects cross-spot descriptor
    info that the context-free MLP otherwise lacks (a genuine test of conditioning on this backbone)."""
    def __init__(self, fdim, dim, n_genes, dropout, film):
        super().__init__(fdim, dim, n_genes, dropout)
        self.filmc = FiLMCond(fdim, dim, film)

    def forward(self, feat, gxy=None, adj=None):
        n = self.net                                             # Lin,BN,GELU,Drop, Lin,BN,GELU,Drop, Lin
        x = n[3](n[2](n[1](n[0](feat))))
        x = self.filmc(x, feat, adj)
        x = n[7](n[6](n[5](n[4](x))))
        return n[8](x)


class DeepSpaCEFiLM(B.DeepSpaCENet):
    """DeepSpaCE per-spot MLP + FiLM after the first hidden block (mirrors STNetFiLM)."""
    def __init__(self, fdim, dim, n_genes, dropout, film):
        super().__init__(fdim, dim, n_genes, dropout)
        self.filmc = FiLMCond(fdim, dim, film)

    def forward(self, feat, gxy=None, adj=None):
        n = self.net                                             # Lin,ReLU,Drop, Lin,ReLU,Drop, Lin
        x = n[2](n[1](n[0](feat)))
        x = self.filmc(x, feat, adj)
        x = n[5](n[4](n[3](x)))
        return n[6](x)


class MLPProbeFiLM(B.MLPProbeNet):
    """Strong residual-MLP probe + FiLM after the input projection: injects the cross-spot
    descriptor context that this high-capacity but context-free backbone otherwise lacks."""
    def __init__(self, fdim, dim, n_genes, depth, dropout, film):
        super().__init__(fdim, dim, n_genes, depth, dropout)
        self.filmc = FiLMCond(fdim, dim, film)

    def forward(self, feat, gxy=None, adj=None):
        x = self.inp(feat)
        x = self.filmc(x, feat, adj)
        for b in self.blocks:
            x = x + b(x)
        return self.head(x)


# ----------------------------- BLEEP FiLM (desc only) -----------------------------
class BleepFiLM(B.BleepEncoder):
    def __init__(self, fdim, n_genes, dim, dropout, film):
        super().__init__(fdim, n_genes, dim, dropout)
        self.filmc = FiLMCond(fdim, dim, film)

    def embed_img(self, feat):
        h = self.filmc(self.img(feat), feat, None)               # desc: modulate image head
        return F.normalize(h, dim=-1)

    def forward(self, feat, expr):
        zi = F.normalize(self.filmc(self.img(feat), feat, None), dim=-1)
        ze = F.normalize(self.expr(expr), dim=-1)
        return zi, ze


# ----------------------------- build / train -----------------------------
def _load_hyperst():
    """Import the official HyperST hyperbolic-alignment core (baselines/HyperST/hyperst)."""
    hp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines", "HyperST")
    hp = os.path.abspath(hp)
    if hp not in sys.path: sys.path.insert(0, hp)
    from hyperst.modules.alignment import HHAlignment
    from hyperst.modules.encoder import ResMLPEncoder
    return HHAlignment, ResMLPEncoder


class HyperSTFiLM(nn.Module):
    """Feature-matched HyperST (CVPR 2026): the official Hierarchical Hyperbolic Alignment + ResMLP
    decoder, fed frozen UNI features instead of the paper's LoRA-tuned image encoder. The niche stream
    is the kNN-neighbour mean of UNI features (their "spot + neighbours" niche). The gene branch takes
    the log expression and drives ONLY the contrastive/entailment alignment loss during training; the
    prediction path (image_feats -> decoder) is gene-independent, so eval passes zeros and is
    leakage-free. `film` is accepted for interface parity but HyperST already conditions via the niche
    stream, so only `none` is meaningful (a warning is not raised)."""
    def __init__(self, fdim, emb, genes, dropout, alignment_beta, entail_weight):
        super().__init__()
        HHAlignment, ResMLPEncoder = _load_hyperst()
        self.alignment_beta = alignment_beta
        self.gene_dim = genes
        self.image_projector = nn.Linear(fdim, emb)
        self.niche_image_projector = nn.Linear(fdim, emb)
        self.gene_encoder = nn.Linear(genes, emb)
        self.niche_gene_encoder = nn.Linear(genes, emb)
        self.alignment = HHAlignment(image_dim=emb, gene_dim=emb, embed_dim=emb, mlp_ratio=2.0,
                                     image_dropout=dropout, gene_dropout=dropout,
                                     entail_weight=entail_weight, niche_project=True, predict_norm=False)
        self.gene_decoder = ResMLPEncoder(in_features=emb * 2, hidden_size=emb, mlp_ratio=2.0,
                                          drop=0., depth=2)
        self.fc = nn.Linear(emb, genes)
        self.align_loss = torch.zeros(())

    def forward(self, feat, gxy, adj, labels=None):
        w = adj / adj.sum(1, keepdim=True).clamp(min=1)
        image_emb = self.image_projector(feat)
        niche_image_emb = self.niche_image_projector(w @ feat)
        if self.alignment_beta > 0:
            if labels is not None:
                gene_in, niche_gene_in = labels, w @ labels
            else:  # eval: image_feats are gene-independent, so zeros give identical predictions
                gene_in = torch.zeros(feat.shape[0], self.gene_dim, device=feat.device)
                niche_gene_in = gene_in
            r = self.alignment(image_emb=image_emb, niche_image_emb=niche_image_emb,
                               gene_emb=self.gene_encoder(gene_in),
                               niche_gene_emb=self.niche_gene_encoder(niche_gene_in))
            self.align_loss = r["loss"] if labels is not None else torch.zeros((), device=feat.device)
            image_feats, niche_image_feats = r["emb"]["image_feats"], r["emb"]["niche_image_feats"]
        else:
            self.align_loss = torch.zeros((), device=feat.device)
            image_feats, niche_image_feats = image_emb, niche_image_emb
        return self.fc(self.gene_decoder(torch.cat([image_feats, niche_image_feats], dim=-1)))


def build_model(args, n_genes):
    m = args.model
    if m == "hyperst":   return HyperSTFiLM(args.feature_dim, args.dim, n_genes, args.dropout, args.alignment_beta, args.entail_weight)
    if m == "histogene": return HistoGeneFiLM(args.feature_dim, args.dim, args.depth, args.heads, n_genes, args.n_pos, args.dropout, args.film)
    if m == "hist2st":   return Hist2STFiLM(args.feature_dim, args.dim, args.depth2, args.depth3, args.heads, n_genes, args.n_pos, args.dropout, args.film)
    if m == "triplex":   return TriplexFiLM(args.feature_dim, args.dim, args.heads, args.depth, n_genes, args.dropout, args.film)
    if m == "stnet":     return STNetFiLM(args.feature_dim, args.dim, n_genes, args.dropout, args.film)
    if m == "deepspace": return DeepSpaCEFiLM(args.feature_dim, args.dim, n_genes, args.dropout, args.film)
    if m == "mlpprobe":  return MLPProbeFiLM(args.feature_dim, args.dim, n_genes, args.depth, args.dropout, args.film)
    raise ValueError(m)


def train_fold_reg(args, train_slides, val_slides, test_slides, gene_list, device):
    model = build_model(args, len(gene_list)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, early = -1e9, None, 0
    order = list(range(len(train_slides)))
    for epoch in range(1, args.epochs + 1):
        model.train(); np.random.shuffle(order)
        for i in order:
            s = train_slides[i]
            feat, gxy, adj, lab = s["feat"], s["gxy"], s["adj"], s["labels"]
            if feat.shape[0] > args.max_spots:
                sel = torch.randperm(feat.shape[0])[:args.max_spots]
                feat, gxy, lab, adj = feat[sel], gxy[sel], lab[sel], adj[sel][:, sel]
            feat, gxy, adj, lab = feat.to(device), gxy.to(device), adj.to(device), lab.to(device)
            if args.model == "hyperst":
                pred = model(feat, gxy, adj, lab)                       # gene branch drives alignment
                loss = F.mse_loss(pred, lab) + args.alignment_beta * model.align_loss
            else:
                pred = model(feat, gxy, adj)
                loss = F.mse_loss(pred, lab)
            opt.zero_grad(); loss.backward(); opt.step()
        score = B.evaluate(model, val_slides, gene_list, device)["pearson_mean"]   # select on VAL
        if best_state is None or (score == score and score > best):
            best, best_state, early = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            early += 1
            if early >= 20: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return B.evaluate(model, test_slides, gene_list, device)                       # score TEST once


def train_fold_bleep(args, train_slides, val_slides, test_slides, gene_list, device):
    """BLEEP contrastive train + retrieval eval, but with the FiLM'd image head."""
    model = BleepFiLM(args.feature_dim, len(gene_list), args.dim, args.dropout, args.film).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    all_feat, all_expr = B._bleep_pool(train_slides, device)
    M = all_feat.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    ref_sel = torch.randperm(M, generator=g)[:min(args.max_ref, M)]
    ref_feat, ref_expr = all_feat[ref_sel], all_expr[ref_sel]
    best, best_state, early = -1e9, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(M, generator=g)
        for i in range(0, M, args.bleep_batch):
            b = perm[i:i + args.bleep_batch]
            feat = all_feat[b].to(device); expr = all_expr[b].to(device)
            zi, ze = model(feat, expr)
            scale = model.logit_scale.exp().clamp(max=100)
            logits = scale * zi @ ze.t()
            tgt = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
            opt.zero_grad(); loss.backward(); opt.step()
        score = B.bleep_eval(model, ref_feat, ref_expr, val_slides, gene_list, device,
                             args.k_retrieval)["pearson_mean"]                      # select on VAL
        if best_state is None or (score == score and score > best):
            best, best_state, early = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            early += 1
            if early >= 20: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return B.bleep_eval(model, ref_feat, ref_expr, test_slides, gene_list, device,  # score TEST once
                        args.k_retrieval)


def run(args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    args.feature_dim = {"uni_v1_official": 1024, "gigapath": 1536, "resnet50_trunc": 1024}[args.feature_encoder]
    regime_dir = os.path.join(args.splits_root, args.regime)
    split_dir = os.path.join(regime_dir, "splits")
    trains = glob.glob(os.path.join(split_dir, "train_*.csv"))
    fold_names = [os.path.basename(t)[len("train_"):-len(".csv")] for t in trains]
    fold_names.sort(key=lambda x: (int(x) if x.isdigit() else 1 << 30, x))

    tag = f"{args.regime}_{args.model}-{args.film}_seed{args.seed}"
    save_dir = os.path.join(args.save_root, tag); os.makedirs(save_dir, exist_ok=True)
    nm = get_normalize_method(args.normalize_method)
    is_bleep = args.model == "bleep"

    all_res = []
    for fold in fold_names:
        out = os.path.join(save_dir, f"fold_{fold}_results.json")
        if os.path.isfile(out):
            print(f"=== {tag} fold {fold} SKIP ==="); all_res.append(json.load(open(out))); continue
        outer_train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        gene_list = json.load(open(os.path.join(regime_dir, f"genes_{fold}.json")))["genes"]
        seed_fold = args.seed + (int(fold) if str(fold).isdigit() else abs(hash(str(fold))) % 1000)
        inner_train_df, val_df = train_val_split(outer_train_df, seed_fold, args.val_fraction)
        train_slides = B.load_slides(inner_train_df, args, gene_list, nm, args.n_pos, args.k)
        val_slides = B.load_slides(val_df, args, gene_list, nm, args.n_pos, args.k)
        test_slides = B.load_slides(test_df, args, gene_list, nm, args.n_pos, args.k)
        fn = train_fold_bleep if is_bleep else train_fold_reg
        res = fn(args, train_slides, val_slides, test_slides, gene_list, device)   # nested validation
        res["fold"] = fold; res["n_val_slides"] = len(val_df)
        json.dump(res, open(out, "w"), sort_keys=True, indent=4); all_res.append(res)
        print(f"=== {tag} fold {fold}: pearson_mean={res['pearson_mean']:.4f} ===", flush=True)

    pm = [r["pearson_mean"] for r in all_res]
    kfold = {"pearson_mean": float(np.mean(pm)), "pearson_std": float(np.std(pm)), "mean_per_split": pm,
             "spearman_mean": float(np.nanmean([r.get("spearman_mean", float("nan")) for r in all_res])),
             "mse": float(np.nanmean([r.get("mse", float("nan")) for r in all_res])), "n_folds": len(all_res)}
    json.dump(kfold, open(os.path.join(save_dir, "results_kfold.json"), "w"), sort_keys=True, indent=4)
    print(f"\n{tag}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"(per-fold {[round(x,4) for x in pm]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["histogene", "hist2st", "triplex", "stnet", "deepspace", "mlpprobe", "bleep", "hyperst"])
    p.add_argument("--film", required=True, choices=["none", "desc", "local"])
    p.add_argument("--regime", required=True, choices=["POOLED", "LOOO"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--splits_root", default="../cross_organ_splits8")
    p.add_argument("--source_dataroot", default="../dataset")
    p.add_argument("--embed_dataroot", default="../embed_dataroot")
    p.add_argument("--feature_encoder", default="uni_v1_official")
    p.add_argument("--save_root", default="results_film_baselines")
    p.add_argument("--normalize_method", default="log1p")
    p.add_argument("--val_fraction", type=float, default=0.15)
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
    # hyperst (feature-matched HyperST, CVPR 2026)
    p.add_argument("--alignment_beta", type=float, default=0.2)   # weight of hyperbolic alignment loss
    p.add_argument("--entail_weight", type=float, default=0.4)    # entailment term inside alignment
    # bleep
    p.add_argument("--bleep_batch", type=int, default=512)
    p.add_argument("--k_retrieval", type=int, default=50)
    p.add_argument("--max_ref", type=int, default=8000)
    args = p.parse_args()
    if args.model == "bleep" and args.film == "local":
        raise SystemExit("bleep + local FiLM is not well-defined (pooled retrieval); use desc.")
    run(args)
