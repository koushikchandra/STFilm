import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from .transformer import SpatialTransformer
from .config import ModelConfig


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MetadataEmbedder(nn.Module):
    """FiLM conditioner from categorical metadata (e.g. {"organ": 8, "platform": 2}).
    Index 0 of every field is reserved as a null token (unconditional / unseen category)
    for classifier-free guidance; it is initialized to zero so it is a true no-op.
    Pass meta as {field_name: LongTensor [B]} with 0 = null."""
    def __init__(self, num_categories: dict, hidden_size):
        super().__init__()
        self.embeds = nn.ModuleDict({
            name: nn.Embedding(n + 1, hidden_size) for name, n in num_categories.items()
        })
        for emb in self.embeds.values():
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()  # null token -> 0

    def forward(self, meta: dict):
        out = 0
        for name, emb in self.embeds.items():
            out = out + emb(meta[name])
        return out  # [B, hidden_size]


class AttentionPool(nn.Module):
    """Learned-query attention pooling: patch embeddings [B, n, feature_dim] -> K prototype
    tokens [B, K, d_model]. A richer, spatially-aware slide summary than a single masked
    mean-pool; feeds the hybrid cross-attention path. Padding spots are masked out."""
    def __init__(self, feature_dim, d_model, n_proto=8, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.query = nn.Parameter(torch.randn(n_proto, d_model) * 0.02)
        self.in_proj = nn.Linear(feature_dim, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches, pad_mask):
        # patches: [B, n, feature_dim]; pad_mask: [B, n] True = padding
        B, n, _ = patches.shape
        h, dh = self.n_heads, self.d_head
        x = self.in_proj(patches)                                       # [B, n, d_model]
        q = self.q(self.query).view(1, -1, h, dh).expand(B, -1, -1, -1)  # [B, K, h, dh]
        k = self.k(x).view(B, n, h, dh)
        v = self.v(x).view(B, n, h, dh)
        attn = einsum(q, k, 'b q h d, b n h d -> b h q n') / (dh ** 0.5)
        attn = attn.masked_fill(pad_mask[:, None, None, :], -1e9)
        attn = attn.softmax(dim=-1)
        out = einsum(attn, v, 'b h q n, b n h d -> b q h d').reshape(B, -1, h * dh)
        return self.norm(self.proj(out))                                # [B, K, d_model]


class Denoiser(nn.Module):
    def __init__(self, config) -> None:
        super(Denoiser, self).__init__()

        self.film_mode = getattr(config, "film", "context")

        self.backbone = SpatialTransformer(
            ModelConfig(
                n_genes=config.n_genes,
                d_input=config.feature_dim,
                d_model=config.hidden_dim,
                d_edge_model=config.pairwise_hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                n_neighbors=config.n_neighbors,
                act=config.activation,
                cross_attn=(self.film_mode == "hybrid"),
                use_moe=(self.film_mode == "moe"),
                n_experts=getattr(config, "n_experts", 4),
                moe_top_k=getattr(config, "moe_top_k", 0),
                use_prototypes_in_router=getattr(config, "use_prototypes_in_router", False),
            )
        )
        self.loss_func = nn.MSELoss()

        # MoE aux-loss weights (only used when film_mode == "moe"); aux stashed by inference().
        self.lambda_bal = getattr(config, "lambda_bal", 1e-2)
        self.lambda_smooth = getattr(config, "lambda_smooth", 1e-3)
        self._moe_aux = None

        self.fourier_proj = TimestepEmbedder(config.hidden_dim)
        self.image_transform = nn.Linear(config.feature_dim, config.hidden_dim)

        # FiLM ablation switch:
        #   "none"    -> cond=None, backbone runs the upstream un-modulated path (V0 baseline)
        #   "context" -> cond=img+time re-injected via adaLN every layer (V1)
        #   "meta"    -> "context" plus categorical-metadata embedding (V2)
        #   "desc"    -> "context" plus a continuous slide-level histology descriptor (V3):
        #               masked mean-pool of the patch embeddings, projected to hidden_size.
        #               Always available at inference (even for unseen organs) -> no null cliff.
        #   "hybrid"  -> "desc" FiLM/adaLN path PLUS a cross-attention path (V4): cells attend to
        #               K learned prototype tokens pooled from the slide's patch embeddings.
        #               Keeps FiLM; adds expressive per-cell conditioning. (self.film_mode set above.)

        # Optional categorical-metadata FiLM conditioner (organ/platform/subtype).
        meta_categories = getattr(config, "meta_categories", None)
        self.meta_embed = (MetadataEmbedder(meta_categories, config.hidden_dim)
                           if meta_categories else None)

        # V3/V4 continuous GLOBAL histology descriptor projection (masked slide mean).
        # Used by desc, hybrid, and local (local = global term + per-spot local term).
        self.desc_proj = (nn.Linear(config.feature_dim, config.hidden_dim)
                          if self.film_mode in ("desc", "hybrid", "local", "localg") else None)

        # V5 "local": hierarchical local+global FiLM. Adds a PER-SPOT descriptor = mean of
        # each spot's spatial-kNN neighbor embeddings (invariant scalars, distance-selected ->
        # SE(2)-invariant). Isolates "does per-spot conditioning help" vs the global-only desc.
        self.local_proj = (nn.Linear(config.feature_dim, config.hidden_dim)
                           if self.film_mode in ("local", "localg") else None)
        self.n_neighbors_local = getattr(config, "n_neighbors", 8)

        # "localg": gated fusion of the global (desc) and per-spot (local) descriptors. A per-spot,
        # per-channel gate learns how much of the local term to add on top of the global one, so it
        # strictly generalizes "local" (init so gate ~= 1 => localg == local at start).
        if self.film_mode == "localg":
            self.gate_proj = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 3.0)     # sigmoid(3) ~= 0.95 -> starts as local

        # Learned-query attention pool -> K prototype tokens.
        #   hybrid: prototypes feed the per-cell cross-attention path.
        #   moe:    prototypes (mean-pooled) feed the router, only if use_prototypes_in_router.
        build_pool = (self.film_mode == "hybrid") or (
            self.film_mode == "moe" and getattr(config, "use_prototypes_in_router", False))
        self.attn_pool = (AttentionPool(config.feature_dim, config.hidden_dim,
                                        n_proto=getattr(config, "n_proto", 8),
                                        n_heads=config.n_heads)
                          if build_pool else None)

    def _local_descriptor(self, raw_img, coords, pad_mask, k):
        # raw_img [B,n,F], coords [B,n,2], pad_mask [B,n] True=pad -> per-spot mean of kNN
        # neighbor embeddings [B,n,F]. Distance-selected + mean-pooled => SE(2)-invariant.
        B, n, Fd = raw_img.shape
        big = torch.finfo(coords.dtype).max
        dist = torch.cdist(coords, coords)                              # [B,n,n]
        dist = dist.masked_fill(pad_mask[:, None, :], big)             # cannot pick padding
        eye = torch.eye(n, device=dist.device, dtype=torch.bool)
        dist = dist.masked_fill(eye[None], big)                        # exclude self
        kk = min(k, n)
        idx = dist.topk(kk, dim=-1, largest=False).indices            # [B,n,kk]
        src = raw_img.unsqueeze(1).expand(B, n, n, Fd)                 # view, no copy
        nb = torch.gather(src, 2, idx.unsqueeze(-1).expand(B, n, kk, Fd))
        return nb.mean(dim=2)                                          # [B,n,F]

    def inference(self, noisy_exp, img_features, coords, t_steps, meta=None, predict=False):
        # noisy_exp: [B, n_cells, n_genes]
        # img_features: [B, n_cells, n_features]
        # coords: [B, n_cells, 2]
        # t_steps: [B]
        # meta: {field: LongTensor [B]} or None

        raw_img = img_features                       # keep raw embeddings for the V3 descriptor
        img_features = self.image_transform(img_features)
        time_emb = self.fourier_proj(t_steps)[:, None].expand(noisy_exp.shape[0], noisy_exp.shape[1], -1)
        features = img_features + time_emb           # token init (unchanged from upstream)

        cond_tokens = None
        if self.film_mode == "none":
            cond = None                              # upstream baseline: no per-layer modulation
        else:
            cond = features                          # per-layer FiLM conditioner
            if self.meta_embed is not None and meta is not None:
                cond = cond + self.meta_embed(meta)[:, None]   # broadcast [B,hid]->[B,n_cells,hid]
            desc_term = None
            if self.desc_proj is not None:
                valid = (raw_img.sum(-1) != 0).float()         # [B,n_cells] 1=real, 0=pad
                denom = valid.sum(1, keepdim=True).clamp(min=1)
                desc = (raw_img * valid[..., None]).sum(1) / denom   # masked mean -> [B,feature_dim]
                desc_term = self.desc_proj(desc)[:, None]      # [B,1,hidden] broadcast slide descriptor
                cond = cond + desc_term
            if self.local_proj is not None:                    # local: per-spot neighborhood descriptor
                pad_mask_l = raw_img.sum(-1) == 0              # [B, n_cells] True = padding
                local = self._local_descriptor(raw_img, coords, pad_mask_l, self.n_neighbors_local)
                local_term = self.local_proj(local)            # PER-SPOT [B,n_cells,hidden]
                if self.film_mode == "localg":                 # gated fusion: learn how much local to add
                    gate = torch.sigmoid(self.gate_proj(
                        torch.cat([desc_term.expand_as(local_term), local_term], dim=-1)))
                    cond = cond + gate * local_term
                else:
                    cond = cond + local_term
            if self.attn_pool is not None:                     # hybrid: prototype tokens for cross-attn
                pad_mask = raw_img.sum(-1) == 0                 # [B, n_cells] True = padding
                cond_tokens = self.attn_pool(raw_img, pad_mask)  # [B, K, hidden]

        prediction, aux = self.backbone(
            gene_exp=noisy_exp,
            features=features,
            coords=coords,
            cond=cond,
            cond_tokens=cond_tokens,
        )
        self._moe_aux = aux  # (load_balance, smoothness) or None; consumed by forward()
        return prediction

    def forward(self, exp, img_features, coords, labels, t_steps, meta=None):
        prediction = self.inference(exp, img_features, coords, t_steps, meta=meta)
        pad_mask = img_features.sum(-1) == 0
        loss = self.loss_func(prediction[~pad_mask], labels[~pad_mask])
        if self._moe_aux is not None:  # MoE regularizers, only active for film=moe
            bal, smooth = self._moe_aux
            loss = loss + self.lambda_bal * bal + self.lambda_smooth * smooth
        return prediction, loss