import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, SwiGLUPacked
from torch_geometric.utils import to_dense_batch
from einops import rearrange, einsum

from .fa import FrameAveraging


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]


def modulate(x, shift, scale):
    # FiLM / adaLN affine on the scalar token stream; x,shift,scale all [N_tokens, d_model]
    return x * (1 + scale) + shift


class CrossAttention(nn.Module):
    """Per-cell multi-head cross-attention from the invariant scalar token stream to a small
    set of per-cell conditioning tokens (e.g. slide prototype tokens). Equivariance-safe: it
    only reads/writes the invariant scalar stream and never touches coords/frames/edges."""
    def __init__(self, d_model, n_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, ctx):
        # x: [N, d_model] one query per cell; ctx: [N, K, d_model] per-cell conditioning tokens
        N, K, _ = ctx.shape
        h, dh = self.n_heads, self.d_head
        q = self.q(self.norm_q(x)).view(N, h, dh)               # [N, h, dh]
        kv = self.kv(self.norm_kv(ctx)).view(N, K, 2, h, dh)
        k, v = kv[:, :, 0], kv[:, :, 1]                          # [N, K, h, dh]
        attn = einsum(q, k, 'n h d, n k h d -> n h k') / (dh ** 0.5)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = einsum(attn, v, 'n h k, n k h d -> n h d').reshape(N, h * dh)
        return self.proj_drop(self.proj(out))


class MoEMlp(nn.Module):
    """Morphology-routed mixture-of-experts on the invariant scalar token stream (film=moe).

    Replaces the single MLP of the block: a router assigns each cell a distribution over E
    small expert MLPs, and the output is the (optionally top-k sparse) weighted combination.
    Equivariance-safe: it reads/writes only the invariant scalar stream and routes on scalar
    features (never coords / gene_exp). Returns the routing probs (for the spatial-smoothness
    loss) and a Switch-style load-balance aux loss.
    """
    def __init__(self, d_model, n_experts=4, mlp_ratio=4.0, top_k=0,
                 activation="gelu", proj_drop=0.):
        super(MoEMlp, self).__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        hidden = int(d_model * mlp_ratio)
        if activation == "swiglu":
            self.experts = nn.ModuleList([
                SwiGLUPacked(in_features=d_model, hidden_features=hidden,
                             drop=proj_drop, norm_layer=nn.LayerNorm)
                for _ in range(n_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                Mlp(in_features=d_model, hidden_features=hidden,
                    act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm)
                for _ in range(n_experts)
            ])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, h, router_ctx=None):
        # h: [N, d_model] scalar tokens; router_ctx: optional [N, d_model] extra routing signal
        r = h if router_ctx is None else h + router_ctx
        logits = self.router(r)                                   # [N, E]
        if self.top_k and self.top_k < self.n_experts:
            topv, topi = logits.topk(self.top_k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, topi, topv)
            logits = masked
        probs = logits.softmax(dim=-1)                            # [N, E], 0 on non-top-k

        out = 0
        for e, expert in enumerate(self.experts):
            out = out + probs[:, e:e + 1] * expert(h)

        # Switch-Transformer load balance: E * sum_e frac_e * prob_e (frac = hard dispatch)
        hard = torch.zeros_like(probs)
        hard.scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
        frac = hard.mean(dim=0)                                   # [E], detached (no grad path)
        prob = probs.mean(dim=0)                                  # [E]
        aux_balance = self.n_experts * (frac.detach() * prob).sum()
        return out, probs, aux_balance


class GeneUpdate(nn.Module):
    def __init__(
            self,
            d_model,
            n_genes,
            proj_drop=0.,
            non_negative=True,   # accepted to match TransformerBlock's call (upstream passes it)
        ):
        super(GeneUpdate, self).__init__()
        self.non_negative = non_negative

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )
    
    def forward(self, features):
        update = self.output(features) 
        return update


class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(
            self,
            d_model,
            d_edge_model,
            n_genes,
            n_heads=1,
            proj_drop=0.,
            attn_drop=0.,
            activation='gelu',
        ):
        super(MLPAttnEdgeAggregation, self).__init__(dim=2)
        
        self.d_head, self.d_edge_head, self.n_heads = d_model // n_heads, d_edge_model // n_heads, n_heads

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )

        if activation == "swiglu":
            self.mlp_attn = SwiGLUPacked(
                in_features=self.d_head*2+self.d_edge_head+n_genes, hidden_features=d_model,
                out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = SwiGLUPacked(
                in_features=self.dim+1, hidden_features=d_edge_model, 
                out_features=d_edge_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = SwiGLUPacked(
                in_features=d_model+d_edge_model, hidden_features=d_model, 
                out_features=d_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp_attn = Mlp(
                in_features=self.d_head*2+self.d_edge_head+n_genes, hidden_features=d_model,
                out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = Mlp(
                in_features=self.dim+1, hidden_features=d_edge_model, out_features=d_edge_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = Mlp(
                in_features=d_model+d_edge_model, hidden_features=d_model, out_features=d_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None):
        # gene_exp: [N, N_genes], token_embs: [N, -1], geo_token_embs: [N, 3]
        # neighbor_indices: [N, N_neighbor], neighbor_masks: [N, N_neighbor]
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = map(lambda x: rearrange(x, 'n (h d) -> n h d', h=n_heads), (q_s, k_s, v_s))

        """build pairwise representation with FA"""
        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)  # [N, N_neighbor, 2]
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)  # [N, N_neighbor, 1]

        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)  # [N*8, N_neighbors, 3]
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)  # [N, 8, N_neighbors, d_model]

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)  # [N, N_neighbors, d_edge_model]

        """gene expression features"""
        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)  # [N, N_neighbor, N_genes]
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)  # [N, N_neighbor, n_heads, N_genes+1]

        """attention map"""
        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        message = torch.cat([q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand], dim=-1)
        
        attn_map = self.mlp_attn(message).squeeze(-1)
        if neighbor_masks is not None:
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))  # [N, n_heads, N_neighbor]

        """context aggregation"""
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)  # [N, n_heads, N_neighbor, D]
        scalar_context = einsum(attn_map, v_s_neighs, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        edge_context = einsum(attn_map, frame_feats, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        return self.W_output(torch.cat([scalar_context, edge_context], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(            
            self,
            d_model,
            d_edge_model,
            n_genes,
            n_heads=1,
            activation="gelu",
            attn_drop=0.,
            proj_drop=0.,
            gene_exp_non_negative=True,
            mlp_ratio=4.0,
            cross_attn=False,
            use_moe=False,
            n_experts=4,
            moe_top_k=0,
            use_prototypes_in_router=False,
        ):
        super(TransformerBlock, self).__init__()

        self.attn = MLPAttnEdgeAggregation(
            d_model=d_model, d_edge_model=d_edge_model, n_genes=n_genes, n_heads=n_heads,
            proj_drop=proj_drop, attn_drop=attn_drop, activation=activation
        )

        if activation == "swiglu":
            self.mlp = SwiGLUPacked(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp = Mlp(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        
        self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop, non_negative=gene_exp_non_negative)

        # adaLN-Zero FiLM conditioning (zero-initialized in SpatialTransformer).
        # Equivariance-safe: we only gate the geometric attention's OUTPUT (its input is
        # left untouched) and apply full shift/scale on the scalar MLP path.
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model, bias=True),
        )

        # HYBRID (opt-in): cross-attention conditioning on top of FiLM/adaLN. Only built when
        # cross_attn=True (film=hybrid); otherwise this block is byte-identical to the FiLM-only
        # version, so none/context/meta/desc runs are unchanged. Its own adaLN-Zero gate
        # (self.adaLN_xattn) starts at zero so the block is still identity at init.
        self.use_cross_attn = cross_attn
        if cross_attn:
            self.cross_attn = CrossAttention(
                d_model, n_heads=n_heads, attn_drop=attn_drop, proj_drop=proj_drop
            )
            self.adaLN_xattn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=True),
            )

        # MoE (opt-in, film=moe): morphology-routed experts replace the single MLP on the
        # invariant scalar stream. gate_mlp (adaLN-Zero) still multiplies the output, so the
        # path is identity at init and none/context/meta/desc/hybrid are byte-identical.
        self.use_moe = use_moe
        if use_moe:
            self.moe = MoEMlp(d_model, n_experts=n_experts, mlp_ratio=mlp_ratio,
                              top_k=moe_top_k, activation=activation, proj_drop=proj_drop)
            # optional router context from slide prototype tokens (mean-pooled), kept invariant
            self.router_ctx_proj = (nn.Linear(d_model, d_model)
                                    if use_prototypes_in_router else None)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, c=None, cond_tokens=None):
        if c is None:
            # fall back to the original (un-modulated) block
            token_embs = token_embs + self.attn(gene_exp, token_embs, coords, neighbor_indices)
            token_embs = token_embs + self.mlp(token_embs)
            return self.gene_updater(token_embs), token_embs, None

        gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(4, dim=-1)

        # attention path: input untouched (preserves SE(2)-equivariance), gated output
        context_token_embs = self.attn(gene_exp, token_embs, coords, neighbor_indices)
        token_embs = token_embs + gate_msa * context_token_embs

        # HYBRID cross-attention path: gated (adaLN-Zero), on the invariant scalar stream only
        if self.use_cross_attn and cond_tokens is not None:
            gate_xattn = self.adaLN_xattn(c)
            token_embs = token_embs + gate_xattn * self.cross_attn(token_embs, cond_tokens)

        # MLP path: full FiLM on the scalar token stream
        h = modulate(self.norm_mlp(token_embs), shift_mlp, scale_mlp)
        aux = None
        if self.use_moe:
            router_ctx = None
            if self.router_ctx_proj is not None and cond_tokens is not None:
                router_ctx = self.router_ctx_proj(cond_tokens.mean(dim=1))  # [N, d_model]
            mlp_out, probs, aux_balance = self.moe(h, router_ctx)
            token_embs = token_embs + gate_mlp * mlp_out
            # spatial-smoothness of routing over the kNN graph (invariant scalar penalty)
            p_nb = probs[neighbor_indices]                          # [N, n_neighbors, E]
            smooth = ((probs.unsqueeze(1) - p_nb) ** 2).sum(dim=-1).mean()
            aux = (aux_balance, smooth)
        else:
            token_embs = token_embs + gate_mlp * self.mlp(h)

        gene_exp = self.gene_updater(token_embs)
        return gene_exp, token_embs, aux


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()

        self.n_neighbors = config.n_neighbors
        self.cross_attn = getattr(config, "cross_attn", False)
        self.use_moe = getattr(config, "use_moe", False)

        self.blks = nn.ModuleList([
            TransformerBlock(config.d_model, config.d_edge_model,
                          n_genes=config.n_genes, n_heads=config.n_heads,
                              activation=config.act, attn_drop=config.attn_dropout,
                              proj_drop=config.dropout,
                              cross_attn=self.cross_attn,
                              use_moe=self.use_moe,
                              n_experts=getattr(config, "n_experts", 4),
                              moe_top_k=getattr(config, "moe_top_k", 0),
                              use_prototypes_in_router=getattr(config, "use_prototypes_in_router", False),
                            ) \
                for i in range(config.n_layers)
        ])

        # adaLN-Zero: start every block as identity so training begins from a
        # well-behaved point and learns to open the FiLM (and cross-attn) gates.
        for blk in self.blks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
            if getattr(blk, "use_cross_attn", False):
                nn.init.zeros_(blk.adaLN_xattn[-1].weight)
                nn.init.zeros_(blk.adaLN_xattn[-1].bias)

    def _build_graph(self, coords, batch_idx, n_neighbors, exclude_self=True):
        # coords: [N, 2], batch_idx: [N], n_neighbors: int
        exclude_self_mask = torch.eye(coords.shape[0], dtype=torch.bool, device=coords.device)  # 1: diagonal elements
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)  # [N, N], True if the token is in the same batch

        # calculate relative distance
        rel_pos = rearrange(coords, 'n d -> n 1 d') - rearrange(coords, 'n d -> 1 n d')
        rel_dist = rel_pos.norm(dim = -1).detach()  # [N, N]
        if exclude_self:
            rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)
        else:
            rel_dist.masked_fill_(~batch_mask, 1e9)

        dist_values, nearest_indices = rel_dist.topk(n_neighbors, dim = -1, largest = False)
        return nearest_indices

    def forward(self, gene_exp, features, coords, cond=None, cond_tokens=None):
        # gene_exp: [B, N_cells, N_genes], features: [B, N_cells, -1], coords: [B, N_cells, 2]
        # cond: [B, N_cells, d_model] FiLM conditioner (img+time(+meta)); None -> no modulation
        # cond_tokens: [B, K, d_model] per-slide conditioning tokens for hybrid cross-attn; or None
        B, N_cells, N_genes = gene_exp.shape[0], gene_exp.shape[1], gene_exp.shape[-1]
        device = features.device

        pad_mask = features.sum(dim=-1) == 0  # [B, N_cells], True if the token is padding
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)[~pad_mask]

        features = features[~pad_mask]  # [-1, 1024]
        coords = coords[~pad_mask]  # [-1, 3]
        gene_exp = gene_exp[~pad_mask]  # [-1, N_genes]
        if cond is not None:
            cond = cond[~pad_mask]  # [-1, d_model], flattened identically to features
        if cond_tokens is not None:
            cond_tokens = cond_tokens[batch_idx]  # [-1, K, d_model], gathered per surviving cell

        nearest_indices = self._build_graph(
            coords, batch_idx, min(self.n_neighbors, N_cells), exclude_self=True
        )

        # forward pass
        all_gene_exp = []
        bal_total, smooth_total, any_aux = 0.0, 0.0, False
        for blk in self.blks:
            gene_exp, features, aux = blk(gene_exp, features, coords, nearest_indices, cond, cond_tokens)
            if aux is not None:
                any_aux = True
                bal_total = bal_total + aux[0]
                smooth_total = smooth_total + aux[1]
            all_gene_exp.append(gene_exp)
        gene_exp = torch.stack(all_gene_exp, dim=0).mean(dim=0)  # [B, N_cells, N_genes]

        # average the gene expression among the neighbors
        gene_exp, _ = to_dense_batch(gene_exp, batch=batch_idx, fill_value=0, max_num_nodes=N_cells)  # [B, N_cells, N_genes]
        aux_out = (bal_total, smooth_total) if any_aux else None  # MoE load-balance + smoothness
        return gene_exp, aux_out
