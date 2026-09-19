"""ABLATION: stream-gated MIST for the context decomposition.

MIST's block sums three streams: LOCAL (spatial kNN attention), GLOBAL (full self-attention), and
SLIDE (AttentionPool -> slide token). This variant turns each stream on/off independently, and lets
the local stream use spatial-nearest OR random-k neighbours. Reuses the original MIST components
from morphost.py (and random_knn from morphost_random.py); does not modify the paper model.

Full L/G/S factorial (all 7 non-empty stream combinations) + a random-neighbour local variant:
  full          -> L=T G=T S=T        (= full MIST)
  global        -> L=F G=T S=F
  global_local  -> L=T G=T S=F  (spatial kNN local)
  global_slide  -> L=F G=T S=T
  local         -> L=T G=F S=F
  slide         -> L=F G=F S=T
  local_slide   -> L=T G=F S=T
  global_random -> L=T G=T S=F, local_random=T  (local attends to RANDOM k, not spatial kNN)
"""
import torch
import torch.nn as nn

from morphost import knn_graph, RBF, LocalKNNAttention, GlobalAttention, AttentionPool, morphost_loss
from morphost_random import random_knn


class ContextBlock(nn.Module):
    def __init__(self, dim, n_heads, num_rbf, dropout, attn_dropout,
                 use_local, use_global, use_slide, use_distance_bias=True):
        super().__init__()
        self.use_local, self.use_global, self.use_slide = use_local, use_global, use_slide
        self.use_distance_bias = use_distance_bias
        self.ln1 = nn.LayerNorm(dim)
        if use_local:
            self.local_attn = LocalKNNAttention(dim, n_heads, num_rbf, attn_dropout)
        if use_global:
            self.global_attn = GlobalAttention(dim, n_heads, attn_dropout)
        if use_slide:
            self.pool = AttentionPool(dim)
            self.slide_proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, x, nbr_idx, rbf):
        h = self.ln1(x)
        add = 0
        if self.use_local:
            add = add + self.local_attn(h, nbr_idx, rbf, self.use_distance_bias, False)
        if self.use_global:
            add = add + self.global_attn(h)
        if self.use_slide:
            add = add + self.slide_proj(self.pool(h)).unsqueeze(0)
        x = x + add
        x = x + self.ffn(self.ln2(x))
        return x


class ContextMIST(nn.Module):
    def __init__(self, feat_dim=1024, dim=256, n_genes=50, n_layers=4, n_heads=4, k=8,
                 num_rbf=16, dropout=0.1, attn_dropout=0.1, use_local=True, use_global=True,
                 use_slide=True, local_random=False, use_distance_bias=True,
                 normalize_dist=False, **_):
        super().__init__()
        assert use_local or use_global or use_slide, "at least one stream must be enabled"
        self.k = k
        self.normalize_dist = normalize_dist
        self.local_random = local_random
        self.use_local = use_local
        self.in_proj = nn.Linear(feat_dim, dim)
        self.rbf = RBF(num_rbf)
        self.blocks = nn.ModuleList([
            ContextBlock(dim, n_heads, num_rbf, dropout, attn_dropout,
                         use_local, use_global, use_slide, use_distance_bias)
            for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feats, coords, return_hidden=False):
        if self.use_local:
            nbr_idx, dist = (random_knn(coords, self.k) if self.local_random
                             else knn_graph(coords, self.k))
            if self.normalize_dist:
                dist = dist / dist.median().clamp_min(1e-6)
            rbf = self.rbf(dist)
        else:
            nbr_idx, rbf = None, None
        x = self.in_proj(feats)
        for blk in self.blocks:
            x = blk(x, nbr_idx, rbf)
        if return_hidden:
            return x
        return self.head(x)


# config name -> flags
CONFIGS = {
    # --- full L/G/S factorial (all 7 non-empty stream combos) ---
    "full":           dict(use_local=True,  use_global=True,  use_slide=True,  local_random=False),
    "global":         dict(use_local=False, use_global=True,  use_slide=False, local_random=False),
    "global_local":   dict(use_local=True,  use_global=True,  use_slide=False, local_random=False),
    "global_slide":   dict(use_local=False, use_global=True,  use_slide=True,  local_random=False),
    "local":          dict(use_local=True,  use_global=False, use_slide=False, local_random=False),
    "slide":          dict(use_local=False, use_global=False, use_slide=True,  local_random=False),
    "local_slide":    dict(use_local=True,  use_global=False, use_slide=True,  local_random=False),
    # --- local-mechanism variant: random neighbours instead of spatial kNN ---
    "global_random":  dict(use_local=True,  use_global=True,  use_slide=False, local_random=True),
}
