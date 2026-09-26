"""ABLATION: MIST WITHOUT slide-level context (spatial local kNN + global attention only).

Identical to MIST (morphost.py) -- spatial kNN neighbours, same RBF distance bias, same config --
EXCEPT the slide-level stream (AttentionPool -> slide token) is removed from every block. Isolates
how much the slide token contributes to MIST. Reuses knn_graph, RBF, LocalKNNAttention,
GlobalAttention, morphost_loss from morphost.py (does not modify the paper model).
"""
import torch
import torch.nn as nn

from morphost import knn_graph, RBF, LocalKNNAttention, GlobalAttention, morphost_loss


class NoSlideMISTBlock(nn.Module):
    """MorphoBlock with the slide-level (AttentionPool -> slide_proj) stream removed."""
    def __init__(self, dim, n_heads, num_rbf, dropout, attn_dropout,
                 use_distance_bias=True, uniform_local=False):
        super().__init__()
        self.use_distance_bias = use_distance_bias
        self.uniform_local = uniform_local
        self.ln1 = nn.LayerNorm(dim)
        self.local_attn = LocalKNNAttention(dim, n_heads, num_rbf, attn_dropout)
        self.global_attn = GlobalAttention(dim, n_heads, attn_dropout)
        # ABLATION: slide-level context (AttentionPool -> slide token) REMOVED.
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, x, nbr_idx, rbf):
        h = self.ln1(x)
        local_out = self.local_attn(h, nbr_idx, rbf, self.use_distance_bias, self.uniform_local)
        x = x + local_out + self.global_attn(h)          # NO slide-token term
        x = x + self.ffn(self.ln2(x))
        return x


class NoSlideMIST(nn.Module):
    """MIST (spatial) with the slide-level stream removed: local kNN + global attention only."""
    def __init__(self, feat_dim=1024, dim=256, n_genes=50, n_layers=4, n_heads=4, k=8,
                 num_rbf=16, dropout=0.1, attn_dropout=0.1, use_distance_bias=True,
                 normalize_dist=False, uniform_local=False, **_):
        super().__init__()
        self.k = k
        self.normalize_dist = normalize_dist
        self.in_proj = nn.Linear(feat_dim, dim)
        self.rbf = RBF(num_rbf)
        self.blocks = nn.ModuleList([
            NoSlideMISTBlock(dim, n_heads, num_rbf, dropout, attn_dropout, use_distance_bias, uniform_local)
            for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feats, coords, return_hidden=False):
        nbr_idx, dist = knn_graph(coords, self.k)        # spatial kNN (unchanged from MIST)
        if self.normalize_dist:
            dist = dist / dist.median().clamp_min(1e-6)
        rbf = self.rbf(dist)
        x = self.in_proj(feats)
        for blk in self.blocks:
            x = blk(x, nbr_idx, rbf)
        if return_hidden:
            return x
        return self.head(x)
