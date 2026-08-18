"""LoGST (V3): Local-Global Spatial Transformer for gene expression prediction.

Input:  feats [N, feat_dim]  — frozen patch embeddings (e.g. UNI, CONCH, UNI+CONCH concat)
        coords [N, 2]        — pixel-space (x, y) spot coordinates
Output: [N, n_genes]         — predicted log-normalised expression for every spot

Coordinate invariance: coords enter ONLY through pairwise distances (RBF-encoded).
Predictions are therefore invariant to translation, rotation, and reflection.

Architecture (one MorphoBlock repeated n_layers times):
  1. LocalKNNAttention  — distance-biased attention over k spatial neighbours
  2. GlobalAttention    — full self-attention over all N spots
  3. AttentionPool      — learned scalar scoring → slide summary token → added to global stream
  Residual fusion: x = x + local_out + global_out + slide_token
                   x = x + FFN(LayerNorm(x))
Output head: LayerNorm → Linear(dim → n_genes)

Loss: MSE + 0.5 * (1 - mean_gene_PCC)   [morphost_loss]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def knn_graph(coords: torch.Tensor, k: int):
    """Return (idx [N,k], dist [N,k]) of the k nearest neighbours, excluding self."""
    N = coords.size(0)
    d = torch.cdist(coords, coords)          # [N, N] — invariant to rigid motion
    d.fill_diagonal_(float("inf"))
    k = min(k, N - 1)
    dist, idx = torch.topk(d, k, dim=1, largest=False)
    return idx, dist


class RBF(nn.Module):
    """Scalar distance → R-dim radial basis: exp(-γ_r · d²), γ on a log grid."""
    def __init__(self, num: int = 16):
        super().__init__()
        self.register_buffer("gammas", torch.logspace(-3, 1, num))

    def forward(self, dist):                 # [...] → [..., num]
        return torch.exp(-self.gammas * dist.unsqueeze(-1) ** 2)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class LocalKNNAttention(nn.Module):
    """Distance-biased multi-head attention over each spot's k spatial neighbours."""
    def __init__(self, dim: int, n_heads: int, num_rbf: int = 16, dropout: float = 0.1):
        super().__init__()
        self.h = n_heads
        self.dh = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dist_bias = nn.Linear(num_rbf, n_heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, nbr_idx, rbf):     # x[N,D], nbr_idx[N,k], rbf[N,k,R]
        N, k = nbr_idx.shape
        q = self.q(x).view(N, self.h, self.dh)
        kf = self.k(x)[nbr_idx]             # [N, k, H, dh]
        vf = self.v(x)[nbr_idx]
        kf = kf.view(N, k, self.h, self.dh)
        vf = vf.view(N, k, self.h, self.dh)
        logits = torch.einsum("nhd,nkhd->nhk", q, kf) / math.sqrt(self.dh)
        logits = logits + self.dist_bias(rbf).permute(0, 2, 1)   # [N, H, k]
        a = self.drop(logits.softmax(dim=-1))
        out = torch.einsum("nhk,nkhd->nhd", a, vf).reshape(N, -1)
        return self.o(out)


class GlobalAttention(nn.Module):
    """Standard multi-head self-attention over all N spots (permutation-equivariant)."""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.h = n_heads
        self.dh = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                   # x[N, D] → [N, D]
        N = x.size(0)
        q, k, v = self.qkv(x).view(N, 3, self.h, self.dh).permute(1, 2, 0, 3)
        a = self.drop((q @ k.transpose(-2, -1) / math.sqrt(self.dh)).softmax(dim=-1))
        return self.o((a @ v).transpose(0, 1).reshape(N, -1))


class AttentionPool(nn.Module):
    """Slide-level summary token: learned scalar scoring → weighted sum → [D]."""
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                   # x[N, D] → [D]
        return (self.score(x).softmax(dim=0) * x).sum(dim=0)


# ---------------------------------------------------------------------------
# Transformer block (V3: local + global + slide token, no conditioning)
# ---------------------------------------------------------------------------

class MorphoBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, num_rbf: int,
                 dropout: float, attn_dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.local_attn = LocalKNNAttention(dim, n_heads, num_rbf, attn_dropout)
        self.global_attn = GlobalAttention(dim, n_heads, attn_dropout)
        self.pool = AttentionPool(dim)
        self.slide_proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, nbr_idx, rbf):
        h = self.ln1(x)
        local_out = self.local_attn(h, nbr_idx, rbf)           # [N, D]
        global_out = self.global_attn(h)                        # [N, D]
        slide_tok = self.slide_proj(self.pool(h))               # [D]
        x = x + local_out + global_out + slide_tok.unsqueeze(0)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MorphoST(nn.Module):
    """LoGST V3: coordinate-invariant local-global spatial transformer."""
    def __init__(
        self,
        feat_dim: int = 1024,
        dim: int = 256,
        n_genes: int = 50,
        n_layers: int = 4,
        n_heads: int = 4,
        k: int = 8,
        num_rbf: int = 16,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k
        self.in_proj = nn.Linear(feat_dim, dim)
        self.rbf = RBF(num_rbf)
        self.blocks = nn.ModuleList([
            MorphoBlock(dim, n_heads, num_rbf, dropout, attn_dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        feats:  [N, feat_dim]  patch embeddings
        coords: [N, 2]         spot coordinates (any unit, any orientation)
        returns [N, n_genes]
        """
        nbr_idx, dist = knn_graph(coords, self.k)
        rbf = self.rbf(dist)                 # [N, k, num_rbf]
        x = self.in_proj(feats)
        for blk in self.blocks:
            x = blk(x, nbr_idx, rbf)
        return self.head(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def morphost_loss(pred: torch.Tensor, target: torch.Tensor,
                  corr_weight: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """MSE + corr_weight * (1 - mean per-gene Pearson) over spots in one slide."""
    mse = F.mse_loss(pred, target)
    pd = pred - pred.mean(0, keepdim=True)
    gd = target - target.mean(0, keepdim=True)
    corr = (pd * gd).sum(0) / (pd.norm(dim=0) * gd.norm(dim=0) + eps)   # [G]
    valid = gd.norm(dim=0) > eps
    corr_loss = 1.0 - corr[valid].mean() if valid.any() else pred.new_tensor(0.0)
    return mse + corr_weight * corr_loss
