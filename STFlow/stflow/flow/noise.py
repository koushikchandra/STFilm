import torch


class PriorSampler:
    def __init__(self, prior_sample_type, **kwargs):
        self.prior_sample_type = prior_sample_type

        if prior_sample_type == "gaussian":
            self.prior_sampler = gaussian_prior
        elif prior_sample_type == "zero":
            self.prior_sampler = all_zeros
        elif prior_sample_type == "zinb":
            # https://github.com/scverse/scvi-tools/blob/main/src/scvi/distributions/_negative_binomial.py#L433
            tc, lg = kwargs.get("total_count", None), kwargs.get("logits", None)
            zi = kwargs.get("zi_logits", None)  # real number
            try:
                from scvi.distributions import ZeroInflatedNegativeBinomial
                prior_sampler = ZeroInflatedNegativeBinomial(total_count=tc, logits=lg, zi_logits=zi)
                self.prior_sampler = lambda shape: prior_sampler.sample(shape).squeeze(-1)
            except ImportError:
                # pure-torch fallback: scvi's ZINB is a Gamma-Poisson NB with a Bernoulli
                # zero-inflation gate; torch.distributions.NegativeBinomial samples identically.
                self.prior_sampler = lambda shape: zinb_prior(shape, tc, lg, zi)
        else:
            raise ValueError("Invalid prior sample type")

    def sample(self, shape):
        return self.prior_sampler(shape)


def gaussian_prior(shape):
    return torch.randn(shape)


def all_zeros(shape):
    return torch.zeros(shape)


def zinb_prior(shape, total_count, logits, zi_logits):
    """Pure-torch ZeroInflatedNegativeBinomial sampler matching scvi's parameterization
    (total_count, logits) + Bernoulli zero-inflation gate sigmoid(zi_logits)."""
    tc = total_count if torch.is_tensor(total_count) else torch.tensor([float(total_count)])
    lg = logits if torch.is_tensor(logits) else torch.tensor([float(logits)])
    nb = torch.distributions.NegativeBinomial(total_count=tc.float(), logits=lg.float())
    x = nb.sample(torch.Size(shape)).squeeze(-1)  # params have shape [1] -> trailing dim
    if zi_logits is not None:
        p_zero = torch.sigmoid(torch.as_tensor(float(zi_logits)))
        x = x.masked_fill(torch.rand_like(x) < p_zero, 0.0)
    return x.float()
