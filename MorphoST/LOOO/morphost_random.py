"""ABLATION: random-k neighbourhood control for MIST (the spatial model, morphost.py).

MIST defines each spot's local neighbourhood by SPATIAL kNN (the k nearest spots by coordinates).
RandomKMIST is identical in every other respect -- same MorphoBlock (local + global + slide),
same RBF distance bias, same config -- EXCEPT the local neighbour set is chosen UNIFORMLY AT RANDOM
instead of by spatial proximity. The randomly chosen neighbours' actual Euclidean distances still
feed the RBF bias, so the ONLY thing that differs from MIST is neighbour SELECTION.

Comparison: MIST (spatial top-k)  vs  RandomKMIST (random-k)
  * MIST >> Random  -> spatial neighbour selection carries real signal.
  * MIST  ~ Random  -> which spots are "local" is irrelevant; global + slide streams carry it.

Reuses RBF, MorphoBlock, morphost_loss from morphost.py (does not modify the paper model).
"""
import torch
import torch.nn as nn

from morphost import RBF, MorphoBlock, morphost_loss  # reuse identical components


def random_knn(coords: torch.Tensor, k: int):
    """k RANDOM neighbours per spot (excluding self). Selection ignores coordinates; the returned
    distances are the chosen neighbours' actual Euclidean distances (for the identical RBF bias)."""
    N = coords.size(0)
    k = min(k, N - 1)
    scores = torch.rand(N, N, device=coords.device)
    scores.fill_diagonal_(-1.0)                       # exclude self
    _, idx = torch.topk(scores, k, dim=1)             # k uniformly-random neighbours
    d = torch.cdist(coords, coords)
    dist = torch.gather(d, 1, idx)
    return idx, dist


class RandomKMIST(nn.Module):
    """MIST with RANDOM-k local neighbourhoods (spatial-selection ablation control)."""
    def __init__(self, feat_dim=1024, dim=256, n_genes=50, n_layers=4, n_heads=4, k=8,
                 num_rbf=16, dropout=0.1, attn_dropout=0.1, use_distance_bias=True,
                 normalize_dist=False, uniform_local=False, **_):
        super().__init__()
        self.k = k
        self.normalize_dist = normalize_dist
        self.in_proj = nn.Linear(feat_dim, dim)
        self.rbf = RBF(num_rbf)
        self.blocks = nn.ModuleList([
            MorphoBlock(dim, n_heads, num_rbf, dropout, attn_dropout, use_distance_bias, uniform_local)
            for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feats, coords, return_hidden=False):
        nbr_idx, dist = random_knn(coords, self.k)        # RANDOM neighbours (not nearest)
        if self.normalize_dist:
            dist = dist / dist.median().clamp_min(1e-6)
        rbf = self.rbf(dist)
        x = self.in_proj(feats)
        for blk in self.blocks:
            x = blk(x, nbr_idx, rbf)
        if return_hidden:
            return x
        return self.head(x)
