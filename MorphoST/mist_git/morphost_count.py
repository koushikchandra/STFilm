"""Count-likelihood head for MorphoST: Negative Binomial (NB) and Zero-Inflated NB (ZINB).

Motivation: the default model predicts log1p expression under a Gaussian (MSE) loss. Spatial
transcriptomics is over-dispersed count data with heavy dropout, so a count likelihood may model
the low-signal / high-dropout cohorts better.

Design note: standard scVI-style NB decoders scale the mean by the *observed* library size. In
H&E->ST prediction the counts are the target and are NOT observed at test time, so we cannot use
observed library size. We therefore use a log-link NB GLM: mu = exp(rate) directly (rate from the
backbone), with a per-gene dispersion theta. ZINB adds a per-gene-per-spot dropout logit.

CountMorphoST(feats, coords)                 -> predicted expression in log1p space (for metrics)
CountMorphoST(feats, coords, return_params=True) -> (mu, theta, pi_logits or None) for the loss
count_loss(params, counts, kind)             -> mean NB / ZINB negative log-likelihood
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from morphost import MorphoST

_EPS = 1e-8


class CountMorphoST(nn.Module):
    def __init__(self, feat_dim, dim, n_genes, zinb=False, **kw):
        super().__init__()
        self.zinb = zinb
        self.n_genes = n_genes
        # reuse the full MorphoST backbone; its own head is unused (return_hidden=True)
        self.backbone = MorphoST(feat_dim=feat_dim, dim=dim, n_genes=n_genes, **kw)
        self.norm = nn.LayerNorm(dim)
        self.rate_head = nn.Linear(dim, n_genes)                 # log-mean per gene per spot
        self.theta = nn.Parameter(torch.zeros(n_genes))          # log-dispersion per gene (global)
        self.drop_head = nn.Linear(dim, n_genes) if zinb else None  # dropout logit (ZINB)

    def _params(self, feats, coords):
        h = self.norm(self.backbone(feats, coords, return_hidden=True))
        mu = torch.exp(self.rate_head(h).clamp(max=12.0))        # [N,G] positive mean
        theta = torch.exp(self.theta).clamp(min=1e-4, max=1e4)   # [G] dispersion
        pi_logits = self.drop_head(h) if self.zinb else None     # [N,G]
        return mu, theta, pi_logits

    def forward(self, feats, coords, return_params=False):
        mu, theta, pi_logits = self._params(feats, coords)
        if return_params:
            return mu, theta, pi_logits
        # predicted expression for metrics: expected value, then log1p (comparable to log1p targets)
        if self.zinb:
            expected = (1.0 - torch.sigmoid(pi_logits)) * mu
        else:
            expected = mu
        return torch.log1p(expected)


def _nb_ll(y, mu, theta):
    """Elementwise NB log-likelihood. theta broadcast over spots. (scVI parameterisation)"""
    log_theta_mu = torch.log(theta + mu + _EPS)
    return (
        theta * (torch.log(theta + _EPS) - log_theta_mu)
        + y * (torch.log(mu + _EPS) - log_theta_mu)
        + torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
    )


def count_loss(params, counts, kind="nb"):
    """Mean negative log-likelihood over spots x genes."""
    mu, theta, pi_logits = params
    theta = theta.unsqueeze(0)                       # [1,G] -> broadcast
    y = counts
    if kind == "nb":
        ll = _nb_ll(y, mu, theta)
    elif kind == "zinb":
        # scVI-style numerically stable ZINB
        sp = F.softplus(-pi_logits)                  # -log sigmoid(-pi) = softplus(pi)? use logits
        log_sig = -F.softplus(pi_logits)             # log(1-pi_dropout)
        log_dropout = -F.softplus(-pi_logits)        # log(pi_dropout)
        nb_ll_pos = _nb_ll(y, mu, theta)
        # y == 0 branch: log( pi + (1-pi) * NB(0) )
        nb_zero = _nb_ll(torch.zeros_like(y), mu, theta)   # NB log-prob at 0
        zero_case = torch.logaddexp(log_dropout, log_sig + nb_zero)
        pos_case = log_sig + nb_ll_pos
        ll = torch.where(y < 0.5, zero_case, pos_case)
    else:
        raise ValueError(kind)
    return -ll.mean()
