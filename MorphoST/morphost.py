"""MorphoST: Morphology-aware Local-Global Spatial Transformer for ST prediction.

Morphology-*aware* (fuses local kNN + global slide morphology context), NOT morphology-*conditioned*:
the explicit adaLN/FiLM conditioning stage (V4/V5) is available but found unnecessary -- V3 (no
conditioning) is the default model.

A from-scratch backbone, fully independent of STFlow (no flow matching, no ZINB sampling, no
frame averaging). E(2)-invariance is by construction: coordinates enter ONLY through pairwise
distances (rotation/translation invariant), never as absolute positions.

Staged ablation (set via `version` V0..V5, or the individual `use_*` flags):
  V0  UNI -> MLP                          image-only floor
  V1  + global self-attention            global modeling
  V2  + local kNN distance-biased attn   local spatial structure
  V3  + global slide token (attn pool)   local + global
  V4  + morphology conditioning (adaLN)  main adaptation idea
  V5  + adaptive local/global gate       full proposed model

Per-sample forward (one slide at a time): features [N,1024], coords [N,2] -> expression [N,G].
Batching is handled by the trainer looping over samples, so there is no padding/masking here.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ geometry
def knn_graph(coords: torch.Tensor, k: int):
    """Return (idx [N,k], dist [N,k]) of the k nearest neighbours (excluding self)."""
    N = coords.size(0)
    d = torch.cdist(coords, coords)                       # [N,N], invariant to rigid motion
    d.fill_diagonal_(float("inf"))
    k = min(k, N - 1)
    dist, idx = torch.topk(d, k, dim=1, largest=False)    # [N,k]
    return idx, dist


class RBF(nn.Module):
    """Distance -> radial-basis features exp(-gamma_m d^2), gammas fixed on a log grid."""
    def __init__(self, num=16, d_max=None):
        super().__init__()
        gammas = torch.logspace(-3, 1, num)               # 1e-3 .. 1e1
        self.register_buffer("gammas", gammas)

    def forward(self, dist):                              # dist [...], -> [..., num]
        return torch.exp(-self.gammas * dist.unsqueeze(-1) ** 2)


# ------------------------------------------------------------------ attention
class LocalKNNAttention(nn.Module):
    """Distance-biased attention over each spot's k spatial neighbours."""
    def __init__(self, dim, n_heads, num_rbf=16, attn_dropout=0.1):
        super().__init__()
        self.h, self.dh = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dist_bias = nn.Sequential(nn.Linear(num_rbf, n_heads))
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x, nbr_idx, rbf):                  # x[N,D], nbr_idx[N,k], rbf[N,k,R]
        N, k = nbr_idx.shape
        q = self.q(x).view(N, self.h, self.dh)           # [N,H,dh]
        kf = self.k(x).view(N, self.h, self.dh)
        vf = self.v(x).view(N, self.h, self.dh)
        kn = kf[nbr_idx]                                 # [N,k,H,dh]
        vn = vf[nbr_idx]
        logits = torch.einsum("nhd,nkhd->nhk", q, kn) / math.sqrt(self.dh)
        logits = logits + self.dist_bias(rbf).permute(0, 2, 1)   # [N,H,k]
        a = self.drop(logits.softmax(dim=-1))
        out = torch.einsum("nhk,nkhd->nhd", a, vn).reshape(N, -1)
        return self.o(out)


class GlobalAttention(nn.Module):
    """Standard full self-attention over all spots (permutation-equivariant, coord-free)."""
    def __init__(self, dim, n_heads, attn_dropout=0.1):
        super().__init__()
        self.h, self.dh = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x):                                # x[N,D]
        N = x.size(0)
        q, k, v = self.qkv(x).view(N, 3, self.h, self.dh).permute(1, 2, 0, 3)  # each [H,N,dh]
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        a = self.drop(a.softmax(dim=-1))
        out = (a @ v).transpose(0, 1).reshape(N, -1)
        return self.o(out)


class AttentionPool(nn.Module):
    """Slide token = attention-weighted pool over spots (content-based -> invariant)."""
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                                # x[N,D] -> [D]
        w = self.score(x).softmax(dim=0)                 # [N,1]
        return (w * x).sum(dim=0)


# ------------------------------------------------------------------ block
class MorphoBlock(nn.Module):
    def __init__(self, dim, n_heads, num_rbf, dropout, attn_dropout,
                 use_local, use_global, use_slide_token, use_morph_cond, use_gate):
        super().__init__()
        self.use_local = use_local
        self.use_global = use_global
        self.use_slide_token = use_slide_token
        self.use_morph_cond = use_morph_cond
        self.use_gate = use_gate

        self.ln1 = nn.LayerNorm(dim)
        if use_local:
            self.local_attn = LocalKNNAttention(dim, n_heads, num_rbf, attn_dropout)
        if use_global:
            self.global_attn = GlobalAttention(dim, n_heads, attn_dropout)
        if use_slide_token:
            self.pool = AttentionPool(dim)
            self.g_proj = nn.Linear(dim, dim)
        if use_gate and use_local and use_global:
            self.gate = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        # morphology-conditioned adaLN (zero-init -> identity at start)
        if use_morph_cond:
            self.ada = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, x, nbr_idx, rbf, cond):
        h = self.ln1(x)
        if self.use_morph_cond and cond is not None:
            gamma, beta = self.ada(cond).chunk(2, dim=-1)     # [N,D] each
            h = (1 + gamma) * h + beta
        h_local = self.local_attn(h, nbr_idx, rbf) if self.use_local else None
        h_glob = self.global_attn(h) if self.use_global else None
        if self.use_slide_token:
            g = self.g_proj(self.pool(h))                     # [D]
            if h_glob is not None:  h_glob = h_glob + g
            else:                   h_glob = g.expand_as(x)
        # combine local & global
        if self.use_gate and h_local is not None and h_glob is not None:
            g_tok = h_glob                                    # per-spot global context
            r = torch.sigmoid(self.gate(torch.cat([h_local, g_tok,
                                                   cond if cond is not None else h_local], -1)))
            attn_out = r * h_local + (1 - r) * h_glob
        else:
            parts = [p for p in (h_local, h_glob) if p is not None]
            attn_out = sum(parts) if parts else 0.0
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


# ------------------------------------------------------------------ full model
_VERSIONS = {   # (local, global, slide_token, morph_cond, gate)
    "V0": (False, False, False, False, False),
    "V1": (False, True,  False, False, False),
    "V2": (True,  True,  False, False, False),
    "V3": (True,  True,  True,  False, False),
    "V4": (True,  True,  True,  True,  False),
    "V5": (True,  True,  True,  True,  True),
}


class MorphoST(nn.Module):
    def __init__(self, feat_dim=1024, dim=256, n_genes=50, n_layers=4, n_heads=4,
                 k=8, num_rbf=16, dropout=0.1, attn_dropout=0.1, version="V5",
                 morph_scope="both", cond_dropout=0.0, components=None,
                 use_distance_bias=True, coordinate_mode="distance"):
        super().__init__()
        loc, glob, slide, morph, gate = _VERSIONS[version]
        if components is not None:
            if len(components) != 3 or any(c not in "01" for c in components):
                raise ValueError("components must be a three-bit LOCAL/GLOBAL/SLIDE code, e.g. 101")
            loc, glob, slide = (c == "1" for c in components)
            morph, gate = False, False
        if coordinate_mode not in ("distance", "absolute"):
            raise ValueError("coordinate_mode must be 'distance' or 'absolute'")
        self.k, self.use_morph, self.morph_scope = k, morph, morph_scope
        self.coordinate_mode = coordinate_mode
        self.in_proj = nn.Linear(feat_dim, dim)
        if coordinate_mode == "absolute":
            self.coord_proj = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.rbf = RBF(num_rbf)
        # morphology descriptors from RAW UNI features. morph_scope selects which are used:
        #   both  -> global slide mean + per-spot kNN mean  (original V4/V5)
        #   global-> slide mean only  (tests: is the local desc redundant with local kNN attn?)
        #   local -> per-spot kNN mean only
        if morph:
            if morph_scope in ("both", "global"):
                self.g_desc = nn.Linear(feat_dim, dim)
            if morph_scope in ("both", "local"):
                self.l_desc = nn.Linear(feat_dim, dim)
            self.cond_drop = nn.Dropout(cond_dropout)   # regularize the conditioner (overfit test)
        self.blocks = nn.ModuleList([
            MorphoBlock(dim, n_heads, num_rbf, dropout, attn_dropout,
                        loc, glob, slide, morph, gate) for _ in range(n_layers)])
        if not use_distance_bias:
            for block in self.blocks:
                if block.use_local:
                    nn.init.zeros_(block.local_attn.dist_bias[0].weight)
                    nn.init.zeros_(block.local_attn.dist_bias[0].bias)
                    for parameter in block.local_attn.dist_bias.parameters():
                        parameter.requires_grad_(False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

    def forward(self, feats, coords):                    # feats[N,1024], coords[N,2] -> [N,G]
        nbr_idx, dist = knn_graph(coords, self.k)
        rbf = self.rbf(dist)                             # [N,k,R]
        x = self.in_proj(feats)
        if self.coordinate_mode == "absolute":
            centered = coords - coords.mean(0, keepdim=True)
            scaled = centered / centered.std(0, keepdim=True).clamp_min(1e-6)
            x = x + self.coord_proj(scaled)
        cond = None
        if self.use_morph:
            terms = []
            if self.morph_scope in ("both", "global"):
                terms.append(self.g_desc(feats.mean(0, keepdim=True)))     # [1,D] per-slide
            if self.morph_scope in ("both", "local"):
                terms.append(self.l_desc(feats[nbr_idx].mean(1)))          # [N,D] per-spot
            cond = self.cond_drop(sum(t.expand(feats.size(0), -1) for t in terms))
        for blk in self.blocks:
            x = blk(x, nbr_idx, rbf, cond)
        return self.head(x)


# ------------------------------------------------------------------ loss
def morphost_loss(pred, target, corr_weight=0.5, eps=1e-6):
    """MSE + corr_weight * (1 - mean per-gene Pearson) over the spots in this sample."""
    mse = F.mse_loss(pred, target)
    pd = pred - pred.mean(0, keepdim=True)
    gd = target - target.mean(0, keepdim=True)
    denom = pd.norm(dim=0) * gd.norm(dim=0) + eps
    corr = (pd * gd).sum(0) / denom                      # [G]
    valid = gd.norm(dim=0) > eps
    corr_loss = 1 - corr[valid].mean() if valid.any() else pred.new_tensor(0.0)
    return mse + corr_weight * corr_loss
