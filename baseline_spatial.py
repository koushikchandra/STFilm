"""Feature-matched spatial baselines on UNI features (BLEEP, Hist2ST, HisToGene, ST-Net, TRIPLEX).

Apple-to-apple with V0/V3: identical UNI embeddings, cross_organ_splits8 (or stimage_splits)
LOOO+POOLED, per-fold gene panel, log1p normalization, and the same pearson_mean metric
(stflow.app.flow.test.metric_func). We swap each method's raw-image front-end for the shared
UNI features and keep its spatial-modeling core:

  histogene : ViT self-attention over all spots of a slide + learned (x,y) grid pos-embeddings
              (Pang et al. 2021), then an MLP gene head.
  hist2st   : transformer blocks (global) + GraphSAGE GCN blocks over a kNN(coords) graph
              (local) + jumping-knowledge LSTM fusion (Zeng et al. 2022), then a gene head.
  stnet     : independent per-spot MLP regression, no spatial context (He et al. 2020); the
              original DenseNet-121 patch front-end is replaced by the shared UNI feature.
  triplex   : three-resolution fusion (Chung et al. 2024) — spot / neighbor (kNN-mean) /
              global (slide-mean) UNI streams fused by a small transformer, then a gene head.
  bleep     : bi-modal contrastive image/expression embedding (Xie et al. 2023); at inference
              each test spot retrieves its k nearest training spots in the joint image space
              and averages their expression (non-parametric retrieval, not regression).

All regression models train with MSE on log1p expression and use the same test-peek early
stopping (patience 20) as train_cross_organ.py so the comparison to V0/V3 is fair; bleep uses
the same early-stopping loop on its retrieval pearson_mean.

Usage:
  PYTHONPATH=STFlow python baseline_spatial.py --model histogene --regime LOOO --seed 1 \
      --splits_root cross_organ_splits8 --save_root results_spatial_uni8 --device 0
"""
import os
import glob
import json
import argparse
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import scanpy as sc
from stflow.utils import set_random_seed, merge_fold_results
from stflow.data.normalize_utils import get_normalize_method
from stflow.hest_utils.st_dataset import load_adata
from stflow.hest_utils.file_utils import read_assets_from_h5
from stflow.app.flow.test import metric_func
from MorphoST.evaluation import expression_metrics, save_predictions, train_val_split


def _load_adata(h5ad, genes, barcodes, normalize_method):
    """load_adata wrapper that strips GRCm38_ prefix from mouse MEND-series samples."""
    adata = sc.read_h5ad(h5ad)
    if len(adata.var_names) > 0 and adata.var_names[0].startswith("GRCm38_"):
        adata.var_names = adata.var_names.str.replace("GRCm38_", "", regex=False)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        adata = adata[:, genes]
    if normalize_method is not None:
        adata = normalize_method(adata)
    return adata.to_df()

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


def load_slides(df, args, gene_list, normalize_method, n_pos, k, cohort=None, cap=None):
    """cohort=None -> parse from patches_path (cross-organ CSVs, prefixed). Pass cohort explicitly
    for the STFlow-style per-cohort CSVs where patches_path has no cohort prefix.

    cap: if set, deterministically subsample each slide to <=cap spots BEFORE building the dense
    [N,N] kNN adjacency. Needed for models with O(N^2) attention (e.g. FEAST) where a large pooled
    slide's adjacency otherwise exhausts host/GPU memory. Seeded by slide id, so repeated loads (and
    none/desc/local FiLM variants) get identical spots."""
    slides = []
    for _, row in df.iterrows():
        coh = cohort if cohort is not None else row["patches_path"].split("/")[0]
        sid = row["sample_id"]
        h5 = os.path.join(args.embed_dataroot, coh, args.feature_encoder, f"fp32/{sid}.h5")
        h5ad = os.path.join(args.source_dataroot, coh, f"adata/{sid}.h5ad")
        dd, _ = read_assets_from_h5(h5)
        barcodes = dd["barcodes"].flatten().astype(str).tolist()
        coords = dd["coords"].astype(np.float64)
        feat = dd["embeddings"].astype(np.float32)
        labels = _load_adata(h5ad, genes=gene_list, barcodes=barcodes,
                             normalize_method=normalize_method).values.astype(np.float32)
        from MorphoST.evaluation import spot_role_index  # single-slide inner-val fallback (no-op otherwise)
        idx = spot_role_index(row, len(feat))
        if idx is not None:
            coords, feat, labels = coords[idx], feat[idx], labels[idx]
        if cap is not None and len(feat) > cap:
            g = np.random.default_rng(abs(hash(str(sid))) % (2**31))
            sel = np.sort(g.permutation(len(feat))[:cap])
            coords, feat, labels = coords[sel], feat[sel], labels[sel]
        slides.append({
            "slide_id": str(sid),
            "feat": torch.from_numpy(feat),
            "coords": torch.from_numpy(coords.astype(np.float32)),
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


class STNetNet(nn.Module):
    """ST-Net (He et al. 2020): independent per-spot regression with NO spatial context. The
    original front-end is a DenseNet-121 fine-tuned per patch; we swap it for the shared UNI
    feature and keep the per-spot MLP regression head."""
    def __init__(self, fdim, dim, n_genes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fdim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, n_genes))

    def forward(self, feat, gxy=None, adj=None):  # per-spot, no coords / graph
        return self.net(feat)


class DeepSpaCENet(nn.Module):
    """DeepSpaCE (Monjo et al. 2022): per-spot regression, NO spatial context. The original VGG16
    CNN front-end is replaced by the shared UNI feature; we keep the fully-connected ReLU/dropout
    regression head. A second context-free backbone (distinct from ST-Net's BN-GELU head)."""
    def __init__(self, fdim, dim, n_genes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fdim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, n_genes))

    def forward(self, feat, gxy=None, adj=None):  # per-spot, no coords / graph
        return self.net(feat)


class MLPProbeNet(nn.Module):
    """Strong per-spot foundation-feature probe: a deep pre-norm residual MLP over UNI features
    (the high-capacity, context-free regressor used as the HEST-bench probing baseline). Modern and
    competitive, yet processes each spot independently -> a strong test of whether FiLM helps even a
    *strong* context-poor backbone (not just weak ones)."""
    def __init__(self, fdim, dim, n_genes, depth, dropout):
        super().__init__()
        self.inp = nn.Linear(fdim, dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(2 * dim, dim))
            for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feat, gxy=None, adj=None):  # per-spot, no coords / graph
        x = self.inp(feat)
        for b in self.blocks:
            x = x + b(x)
        return self.head(x)


class TriplexNet(nn.Module):
    """TRIPLEX (Chung et al. 2024): fuse spot / neighbor / global resolutions. We build the
    three streams from shared UNI features — spot = per-spot feature, neighbor = kNN(coords)
    mean via the adjacency, global = slide-mean token — project each to a token, and let a
    small transformer attend over the 3 resolution tokens per spot; the spot token is read out."""
    def __init__(self, fdim, dim, heads, depth, n_genes, dropout):
        super().__init__()
        self.spot = nn.Linear(fdim, dim)
        self.neigh = nn.Linear(fdim, dim)
        self.glob = nn.Linear(fdim, dim)
        self.res_embed = nn.Parameter(torch.zeros(3, dim))  # per-resolution embedding
        self.blocks = nn.ModuleList([TBlock(dim, heads, 2 * dim, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feat, gxy, adj):
        mask = adj / adj.sum(1, keepdim=True).clamp(min=1)
        neigh_feat = mask.mm(feat)                                   # [N, fdim] local context
        glob_feat = feat.mean(0, keepdim=True).expand_as(feat)      # [N, fdim] global context
        s = self.spot(feat) + self.res_embed[0]
        nb = self.neigh(neigh_feat) + self.res_embed[1]
        gl = self.glob(glob_feat) + self.res_embed[2]
        x = torch.stack([s, nb, gl], dim=1)                         # [N, 3, dim] (batch=N, seq=3)
        for b in self.blocks:
            x = b(x)
        return self.head(x[:, 0])                                   # spot-token readout


class BleepEncoder(nn.Module):
    """BLEEP (Xie et al. 2023) bi-modal contrastive encoder: an image head over UNI features and
    an expression head over log1p counts, aligned by InfoNCE in a shared L2-normalized space."""
    def __init__(self, fdim, n_genes, dim, dropout):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(fdim, dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim, dim))
        self.expr = nn.Sequential(nn.Linear(n_genes, dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def embed_img(self, feat):
        return F.normalize(self.img(feat), dim=-1)

    def forward(self, feat, expr):
        zi = F.normalize(self.img(feat), dim=-1)
        ze = F.normalize(self.expr(expr), dim=-1)
        return zi, ze


class MCToGeneNet(nn.Module):
    """Feature-matched adaptation of MCToGene (Sun et al., CVPR 2026): high-order, many-to-many
    multi-cell interaction via induced-set (latent) attention, hierarchically coupled with pairwise
    graph aggregation. Captures the core mechanism under shared UNI features; not the official model."""
    def __init__(self, fdim, dim, depth, heads, n_genes, dropout, n_latents=16):
        super().__init__()
        self.proj = nn.Linear(fdim, dim)
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.pair = nn.ModuleList([GSBlock(dim, dim) for _ in range(depth)])
        self.to_lat = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) for _ in range(depth)])
        self.from_lat = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) for _ in range(depth)])
        self.ln = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feat, gxy, adj):
        x = self.proj(feat)
        for pair, tl, fl, ln in zip(self.pair, self.to_lat, self.from_lat, self.ln):
            lat = self.latents[None] + tl(self.latents[None], x[None], x[None], need_weights=False)[0]  # latents gather from cells
            mb = fl(x[None], lat, lat, need_weights=False)[0][0]     # cells read high-order (many-body) context
            pr = pair(x, adj)                                        # pairwise graph aggregation
            x = ln(x + mb + pr)                                      # hierarchical coupling
        return self.head(x)


class HiSTNet(nn.Module):
    """Feature-matched adaptation of HiST (Wu et al., 2026): hierarchical sparse WINDOW attention for
    local geometric correspondence, a coarse window-token level for multiscale context, and a slide
    calibration token for global conditioning. Core mechanism under shared UNI features; not official."""
    def __init__(self, fdim, dim, depth, heads, n_genes, n_pos, dropout, win=8):
        super().__init__()
        self.proj = nn.Linear(fdim, dim); self.win = win; self.stride = n_pos // win + 1
        self.q1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.wattn = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) for _ in range(depth)])
        self.q2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.cattn = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
                                               nn.Dropout(dropout), nn.Linear(2 * dim, dim)) for _ in range(depth)])
        self.calib = nn.Linear(dim, 2 * dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feat, gxy, adj=None):
        x = self.proj(feat)
        wid = (gxy[:, 0] // self.win) * self.stride + (gxy[:, 1] // self.win)   # spatial window id
        s = x.mean(0); gamma, beta = self.calib(s).chunk(2, -1); x = x * (1 + gamma) + beta  # slide calibration token
        wmask = wid[:, None] != wid[None, :]                                    # True -> different window (masked)
        for q1, wa, q2, ca, ff in zip(self.q1, self.wattn, self.q2, self.cattn, self.ff):
            h = q1(x)[None]
            x = x + wa(h, h, h, attn_mask=wmask, need_weights=False)[0][0]      # sparse window attention
            uniq, inv = torch.unique(wid, return_inverse=True)
            cnt = torch.bincount(inv).clamp(min=1).float()[:, None]
            wtok = torch.zeros(len(uniq), x.shape[1], device=x.device).index_add_(0, inv, q2(x)) / cnt
            wtok = wtok + ca(wtok[None], wtok[None], wtok[None], need_weights=False)[0][0]  # multiscale window attn
            x = x + wtok[inv] + ff(x)                                           # broadcast coarse context back
        return self.head(x)


class HistoGPANet(nn.Module):
    """Feature-matched adaptation of HistoGPA (Liu et al., 2026): a shared slide-level representation
    modulates local morphology (pathway 1) and conditions learned gene-prior embeddings that each spot
    retrieves via cross-attention (pathway 2); prediction is the affinity to the conditioned gene
    embeddings. Core mechanism under shared UNI features; not the official model."""
    def __init__(self, fdim, dim, depth, heads, n_genes, dropout):
        super().__init__()
        self.proj = nn.Linear(fdim, dim)
        self.local = nn.ModuleList([TBlock(dim, heads, 2 * dim, dropout) for _ in range(depth)])
        self.gene = nn.Parameter(torch.randn(n_genes, dim) * 0.02)             # learned gene prior
        self.slide_to_gene = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.slide_film = nn.Linear(dim, 2 * dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.scale = nn.Parameter(torch.ones(1)); self.bias = nn.Parameter(torch.zeros(n_genes))

    def forward(self, feat, gxy=None, adj=None):
        x = self.proj(feat)
        for b in self.local:
            x = b(x[None])[0]
        s = x.mean(0)
        gamma, beta = self.slide_film(s).chunk(2, -1); x = x * (1 + gamma) + beta   # pathway 1: modulate morphology
        g = self.gene + self.slide_to_gene(s)[None]                                 # pathway 2: condition gene prior
        x = x + self.cross(x[None], g, g, need_weights=False)[0][0]                 # retrieve context-adapted gene prior
        return self.scale * (x @ g[0].t()) + self.bias                              # gene-affinity prediction


def _load_hyperst():
    """Import the official HyperST hyperbolic-alignment core (baselines/HyperST/hyperst)."""
    import sys
    hp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "HyperST"))
    if hp not in sys.path: sys.path.insert(0, hp)
    from hyperst.modules.alignment import HHAlignment
    from hyperst.modules.encoder import ResMLPEncoder
    return HHAlignment, ResMLPEncoder


class HyperSTNet(nn.Module):
    """Feature-matched HyperST (Zhang et al., CVPR 2026): the official Hierarchical Hyperbolic
    Alignment + ResMLP decoder, fed frozen UNI features instead of the paper's LoRA-tuned image
    encoder. Niche = kNN-neighbour mean of UNI features (their "spot + neighbours" niche). The gene
    branch takes the log expression and drives ONLY the contrastive/entailment alignment loss during
    training; the prediction path (image_feats -> decoder) is gene-independent, so eval passes zeros
    and is leakage-free. `needs_labels`/`aux_loss` let the shared trainer add the alignment term."""
    needs_labels = True
    def __init__(self, fdim, emb, genes, dropout, alignment_beta=0.2, entail_weight=0.4):
        super().__init__()
        HHAlignment, ResMLPEncoder = _load_hyperst()
        self.alignment_beta = alignment_beta; self.gene_dim = genes
        self.image_projector = nn.Linear(fdim, emb)
        self.niche_image_projector = nn.Linear(fdim, emb)
        self.gene_encoder = nn.Linear(genes, emb)
        self.niche_gene_encoder = nn.Linear(genes, emb)
        self.alignment = HHAlignment(image_dim=emb, gene_dim=emb, embed_dim=emb, mlp_ratio=2.0,
                                     image_dropout=dropout, gene_dropout=dropout,
                                     entail_weight=entail_weight, niche_project=True, predict_norm=False)
        self.gene_decoder = ResMLPEncoder(in_features=emb * 2, hidden_size=emb, mlp_ratio=2.0, drop=0., depth=2)
        self.fc = nn.Linear(emb, genes)
        self.aux_loss = torch.zeros(())

    def forward(self, feat, gxy, adj, labels=None):
        w = adj / adj.sum(1, keepdim=True).clamp(min=1)
        image_emb = self.image_projector(feat)
        niche_image_emb = self.niche_image_projector(w @ feat)
        if self.alignment_beta > 0:
            if labels is not None:
                gene_in, niche_gene_in = labels, w @ labels
            else:
                gene_in = torch.zeros(feat.shape[0], self.gene_dim, device=feat.device); niche_gene_in = gene_in
            r = self.alignment(image_emb=image_emb, niche_image_emb=niche_image_emb,
                               gene_emb=self.gene_encoder(gene_in), niche_gene_emb=self.niche_gene_encoder(niche_gene_in))
            self.aux_loss = self.alignment_beta * r["loss"] if labels is not None else torch.zeros((), device=feat.device)
            image_feats, niche_image_feats = r["emb"]["image_feats"], r["emb"]["niche_image_feats"]
        else:
            self.aux_loss = torch.zeros((), device=feat.device)
            image_feats, niche_image_feats = image_emb, niche_image_emb
        return self.fc(self.gene_decoder(torch.cat([image_feats, niche_image_feats], dim=-1)))


class GeneDMLNet(nn.Module):
    """Feature-matched Gene-DML: spot / neighbour / global UNI streams fused by a small transformer
    with per-stream deep-metric supervision (the aux term), returning the fused prediction. Aux loss
    is added by the shared trainer via `needs_labels`/`aux_loss`."""
    needs_labels = True
    def __init__(self, fdim, dim, genes, heads, depth, dropout, aux_weight=0.25):
        super().__init__()
        self.aux_weight = aux_weight
        self.spot = nn.Linear(fdim, dim); self.neigh = nn.Linear(fdim, dim); self.glob = nn.Linear(fdim, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(layer, depth)
        self.type_emb = nn.Parameter(torch.zeros(3, dim)); nn.init.normal_(self.type_emb, std=.02)
        self.stream_heads = nn.ModuleList([nn.Linear(dim, genes) for _ in range(3)])
        self.mix = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, genes))
        self.aux_loss = torch.zeros(())

    def forward(self, feat, gxy, adj, labels=None):
        w = adj / adj.sum(1, keepdim=True).clamp(min=1)
        nfeat = w @ feat; gfeat = feat.mean(0, keepdim=True).expand_as(feat)
        toks = torch.stack([self.spot(feat), self.neigh(nfeat), self.glob(gfeat)], 1) + self.type_emb[None]
        toks = self.fuse(toks)
        pred = self.mix(toks.flatten(1))
        if labels is not None:
            aux = sum(F.mse_loss(h(toks[:, i]), labels) for i, h in enumerate(self.stream_heads)) / 3
            self.aux_loss = self.aux_weight * aux
        else:
            self.aux_loss = torch.zeros((), device=feat.device)
        return pred


def _load_m2ost():
    """Import M2OST's cross-scale Transformer core (baselines/M2OST/m2ost_core.py)."""
    import sys
    mp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "M2OST"))
    if mp not in sys.path: sys.path.insert(0, mp)
    from m2ost_core import Transformer
    return Transformer


class M2OSTNet(nn.Module):
    """Feature-matched M2OST (Wang et al., AAAI 2025): the official many-to-one cross-scale Transformer
    (intra-scale ITMM + cross-scale CTMM + channel-mixing CCMM) fed frozen UNI features. Since we cache
    single-magnification UNI features, the paper's 40x/10x/5x image pyramid is approximated by three
    *spatial* scales -- the spot itself (fine), its kNN neighbours (mid), and the slide-global mean
    (coarse) -- so M2OST's cross-scale mixing operates over spatial rather than magnification levels.
    A [cls] token per scale is read out, concatenated, and mapped to the gene panel."""
    def __init__(self, fdim, dim, genes, depth, heads, dropout, k):
        super().__init__()
        Transformer = _load_m2ost()
        self.k = k
        d3 = dim // 3; self.d3 = d3; eff = 3 * d3
        m2ost_heads = 6   # M2OST splits attention as heads//3 and chunks qkv by 3 -> heads must be a
                          # multiple of 3 with inner_dim divisible by 3 (the shared --heads=8 breaks this)
        self.proj_fine = nn.Linear(fdim, d3)
        self.proj_mid = nn.Linear(fdim, d3)
        self.proj_coarse = nn.Linear(fdim, d3)
        self.cls = nn.Parameter(torch.randn(1, 1, d3))
        self.pos_fine = nn.Parameter(torch.randn(1, 2, d3))
        self.pos_mid = nn.Parameter(torch.randn(1, k + 1, d3))
        self.pos_coarse = nn.Parameter(torch.randn(1, 2, d3))
        self.transformer = Transformer(eff, depth, m2ost_heads, 64, eff, dropout)
        self.head = nn.Linear(eff, genes)

    def forward(self, feat, gxy, adj):
        N = feat.shape[0]; kk = min(self.k, N)
        nbr = adj.topk(kk, dim=1).indices                      # [N, kk] nearest neighbours
        cls = self.cls.expand(N, -1, -1)
        fine = torch.cat([cls, self.proj_fine(feat).unsqueeze(1)], 1) + self.pos_fine[:, :2]
        mid = torch.cat([cls, self.proj_mid(feat[nbr])], 1) + self.pos_mid[:, :kk + 1]
        gmean = feat.mean(0, keepdim=True).expand(N, -1)
        coarse = torch.cat([cls, self.proj_coarse(gmean).unsqueeze(1)], 1) + self.pos_coarse[:, :2]
        x = self.transformer(fine, mid, coarse)
        return self.head(torch.cat([a[:, 0] for a in x], dim=-1))   # concat per-scale [cls]


def build_model(args, n_genes):
    if args.model == "m2ost":
        return M2OSTNet(args.feature_dim, args.dim, n_genes, args.depth, args.heads, args.dropout, args.k)
    if args.model == "hyperst":
        return HyperSTNet(args.feature_dim, args.dim, n_genes, args.dropout,
                          getattr(args, "alignment_beta", 0.2), getattr(args, "entail_weight", 0.4))
    if args.model == "genedml":
        return GeneDMLNet(args.feature_dim, args.dim, n_genes, args.heads, args.depth, args.dropout,
                          getattr(args, "aux_weight", 0.25))
    if args.model == "mctogene":
        return MCToGeneNet(args.feature_dim, args.dim, args.depth, args.heads, n_genes, args.dropout)
    if args.model == "hist":
        return HiSTNet(args.feature_dim, args.dim, args.depth, args.heads, n_genes, args.n_pos, args.dropout)
    if args.model == "histogpa":
        return HistoGPANet(args.feature_dim, args.dim, args.depth, args.heads, n_genes, args.dropout)
    if args.model == "histogene":
        return HistoGeneNet(args.feature_dim, args.dim, args.depth, args.heads,
                            n_genes, args.n_pos, args.dropout)
    if args.model == "hist2st":
        return Hist2STNet(args.feature_dim, args.dim, args.depth2, args.depth3, args.heads,
                          n_genes, args.n_pos, args.dropout)
    if args.model == "stnet":
        return STNetNet(args.feature_dim, args.dim, n_genes, args.dropout)
    if args.model == "deepspace":
        return DeepSpaCENet(args.feature_dim, args.dim, n_genes, args.dropout)
    if args.model == "mlpprobe":
        return MLPProbeNet(args.feature_dim, args.dim, n_genes, args.depth, args.dropout)
    if args.model == "triplex":
        return TriplexNet(args.feature_dim, args.dim, args.heads, args.depth, n_genes, args.dropout)
    raise ValueError(f"unknown model {args.model}")


# ----------------------------- train / eval -----------------------------
@torch.no_grad()
def evaluate(model, slides, gene_list, device, prediction_path=None):
    model.eval()
    preds, gts, coords, slide_ids = [], [], [], []
    for s in slides:
        feat = s["feat"].to(device); gxy = s["gxy"].to(device); adj = s["adj"].to(device)
        pred = model(feat, gxy, adj).cpu().numpy()
        preds.append(pred); gts.append(s["labels"].numpy()); coords.append(s["coords"].numpy())
        slide_ids.extend([s["slide_id"]] * len(pred))
    pred = np.concatenate(preds, 0); target = np.concatenate(gts, 0)
    coords = np.concatenate(coords, 0); slide_ids = np.asarray(slide_ids)
    res = expression_metrics(pred, target, gene_list, slide_ids)
    if prediction_path:
        save_predictions(prediction_path, pred, target, coords, slide_ids, gene_list)
    return res


def regression_loss(pred, target, corr_weight):
    mse = F.mse_loss(pred, target)
    if corr_weight == 0:
        return mse
    pd = pred - pred.mean(0, keepdim=True); yd = target - target.mean(0, keepdim=True)
    valid = yd.norm(dim=0) > 1e-6
    corr = (pd * yd).sum(0) / (pd.norm(dim=0) * yd.norm(dim=0) + 1e-6)
    return mse + corr_weight * (1 - corr[valid].mean()) if valid.any() else mse


def train_fold(args, train_slides, val_slides, test_slides, gene_list, device, fold_dir=None):
    model = build_model(args, len(gene_list)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_pearson, best_state, early = -1, None, 0
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
            if getattr(model, "needs_labels", False):
                pred = model(feat, gxy, adj, lab)           # model bakes its own weighted aux term
                loss = regression_loss(pred, lab, args.corr_weight) + model.aux_loss
            else:
                pred = model(feat, gxy, adj)
                loss = regression_loss(pred, lab, args.corr_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        res = evaluate(model, val_slides, gene_list, device)
        if res["pearson_mean"] > best_pearson:
            best_pearson = res["pearson_mean"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            early = 0
        else:
            early += 1
            if early >= 20:
                break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    if fold_dir:
        os.makedirs(fold_dir, exist_ok=True)
        torch.save({"model": best_state, "args": vars(args), "genes": gene_list},
                   os.path.join(fold_dir, "best_model.pt"))
    return evaluate(model, test_slides, gene_list, device,
                    os.path.join(fold_dir, "test_predictions.npz") if fold_dir else None)


# ----------------------------- BLEEP (contrastive + retrieval) -----------------------------
def _bleep_pool(slides, device):
    """Concatenate all slides' features + log1p labels into one retrieval pool."""
    feat = torch.cat([s["feat"] for s in slides], 0)
    expr = torch.cat([s["labels"] for s in slides], 0)
    return feat, expr


@torch.no_grad()
def bleep_eval(model, ref_feat, ref_expr, test_slides, gene_list, device, k, prediction_path=None):
    model.eval()
    zi_ref = model.embed_img(ref_feat.to(device))          # [M, dim]
    ref_expr = ref_expr.to(device)                          # [M, G] log1p true expression
    preds, gts, coords, slide_ids = [], [], [], []
    for s in test_slides:
        zi_q = model.embed_img(s["feat"].to(device))       # [N, dim]
        sim = zi_q @ zi_ref.t()                            # [N, M] cosine (both L2-normed)
        idx = sim.topk(min(k, zi_ref.shape[0]), dim=1).indices
        pred = ref_expr[idx].mean(1)                       # avg neighbour expression (imputation)
        pred = pred.cpu().numpy(); preds.append(pred); gts.append(s["labels"].numpy())
        coords.append(s["coords"].numpy()); slide_ids.extend([s["slide_id"]] * len(pred))
    pred = np.concatenate(preds, 0); target = np.concatenate(gts, 0)
    coords = np.concatenate(coords, 0); slide_ids = np.asarray(slide_ids)
    res = expression_metrics(pred, target, gene_list, slide_ids)
    if prediction_path:
        save_predictions(prediction_path, pred, target, coords, slide_ids, gene_list)
    return res


def bleep_train_fold(args, train_slides, val_slides, test_slides, gene_list, device, fold_dir=None):
    model = BleepEncoder(args.feature_dim, len(gene_list), args.dim, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    all_feat, all_expr = _bleep_pool(train_slides, device)          # [M, .]
    M = all_feat.shape[0]
    # fixed retrieval reference pool (subsample for memory/speed), faithful to BLEEP's reference set
    g = torch.Generator().manual_seed(args.seed)
    ref_sel = torch.randperm(M, generator=g)[:min(args.max_ref, M)]
    ref_feat, ref_expr = all_feat[ref_sel], all_expr[ref_sel]
    best_pearson, best_state, early = -1, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(M, generator=g)
        for i in range(0, M, args.bleep_batch):
            b = perm[i:i + args.bleep_batch]
            feat = all_feat[b].to(device); expr = all_expr[b].to(device)
            zi, ze = model(feat, expr)
            scale = model.logit_scale.exp().clamp(max=100)
            logits = scale * zi @ ze.t()                             # [B, B]
            tgt = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
            opt.zero_grad(); loss.backward(); opt.step()
        res = bleep_eval(model, ref_feat, ref_expr, val_slides, gene_list, device, args.k_retrieval)
        if res["pearson_mean"] > best_pearson:
            best_pearson = res["pearson_mean"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            early = 0
        else:
            early += 1
            if early >= 20:
                break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    if fold_dir:
        os.makedirs(fold_dir, exist_ok=True)
        torch.save({"model": best_state, "args": vars(args), "genes": gene_list},
                   os.path.join(fold_dir, "best_model.pt"))
    return bleep_eval(model, ref_feat, ref_expr, test_slides, gene_list, device, args.k_retrieval,
                      os.path.join(fold_dir, "test_predictions.npz") if fold_dir else None)


def run(args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    args.feature_dim = {"uni_v1_official": 1024, "gigapath": 1536, "resnet50_trunc": 1024}[args.feature_encoder]
    regime_dir = os.path.join(args.splits_root, args.regime)
    split_dir = os.path.join(regime_dir, "splits")
    # derive fold names from the split CSVs so this works for both HEST and STImage organ sets
    trains = glob.glob(os.path.join(split_dir, "train_*.csv"))
    fold_names = [os.path.basename(t)[len("train_"):-len(".csv")] for t in trains]
    fold_names.sort(key=lambda x: (int(x) if x.isdigit() else 1 << 30, x))

    save_dir = os.path.join(args.save_root, f"{args.regime}_{args.model}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    nm = get_normalize_method(args.normalize_method)

    all_res = []
    for fold in fold_names:
        out = os.path.join(save_dir, f"fold_{fold}_results.json")
        if os.path.isfile(out):
            print(f"=== {args.regime} {args.model} fold {fold} seed{args.seed} -> SKIP ===")
            all_res.append(json.load(open(out))); continue
        outer_train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        train_df, val_df = train_val_split(outer_train_df, args.seed + sum(map(ord, str(fold))),
                                           args.val_fraction)
        gene_list = json.load(open(os.path.join(regime_dir, f"genes_{fold}.json")))["genes"]
        train_slides = load_slides(train_df, args, gene_list, nm, args.n_pos, args.k)
        val_slides = load_slides(val_df, args, gene_list, nm, args.n_pos, args.k)
        test_slides = load_slides(test_df, args, gene_list, nm, args.n_pos, args.k)
        fold_dir = os.path.join(save_dir, f"fold_{fold}")
        if args.model == "bleep":
            res = bleep_train_fold(args, train_slides, val_slides, test_slides, gene_list, device, fold_dir)
        else:
            res = train_fold(args, train_slides, val_slides, test_slides, gene_list, device, fold_dir)
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
    p.add_argument("--model", required=True,
                   choices=["histogene", "hist2st", "stnet", "deepspace", "mlpprobe", "triplex", "bleep",
                            "mctogene", "hist", "histogpa", "genedml", "hyperst"])
    p.add_argument("--regime", required=True, choices=["POOLED", "LOOO", "INTRA"])
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
    # bleep-specific
    p.add_argument("--bleep_batch", type=int, default=512)   # InfoNCE contrastive batch
    p.add_argument("--k_retrieval", type=int, default=50)    # neighbours averaged at inference
    p.add_argument("--max_ref", type=int, default=30000)     # retrieval reference-pool cap
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--corr_weight", type=float, default=0.0)
    args = p.parse_args()
    run(args)
