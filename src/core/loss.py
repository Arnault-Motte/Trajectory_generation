import torch
import torch.distributions as distrib
import torch.nn.functional as F


# %%
# MSE loss for reconstruction
def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """
    Basic MSE loss
    """
    return F.mse_loss(x_recon, x, reduction="sum")

def reconstruction_loss2(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    MSE loss with optional masking.
    """
    squared_error = (x_recon - x) ** 2  # shape: (batch, seq_len, features)

    if mask is not None:
        # Expand mask to shape (batch, seq_len, features)
        mask = mask.unsqueeze(1)           # shape: (batch, 1, features)
        mask = mask.expand_as(x)           # shape: (batch, seq_len, features)

        masked_error = squared_error * mask
        return masked_error.sum()
    else:
        return squared_error.sum()


def negative_log_likelihood(
    x: torch.Tensor,
    recon: torch.Tensor,
    scale: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes the negative log likelihood for the given scalar scale.
    """
    mu = recon
    dist = distrib.Normal(mu, scale)
    log_likelihood = dist.log_prob(x)

    if mask is not None:
        mask = mask.unsqueeze(1) 
        mask = mask.expand_as(x)
        # print("not masked ",-log_likelihood.sum(dim=[i for i in range(1, len(x.size()))]).mean())
        
        
        masked_log_prob = log_likelihood * mask
        # print(log_likelihood.shape)
        # print("masked ",-masked_log_prob.sum(dim=[i for i in range(1, len(x.size()))]).mean())
        # print("masked ",-masked_log_prob)
        # print(-log_likelihood)
        # input("continue")
        return -masked_log_prob.sum(dim=[i for i in range(1, len(x.size()))])

    # print("not masked ",-log_likelihood.sum(dim=[i for i in range(1, len(x.size()))]).mean())
    # print(-log_likelihood)
    # input("continue")
    return -log_likelihood.sum(dim=[i for i in range(1, len(x.size()))])


# Normal KL loss for gaussian prior
def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL loss for the given parameters and the standard normal
    law.
    """
    var = torch.exp(logvar)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
    return kl_divergence


def create_mixture(
    mu: torch.Tensor, log_var: torch.Tensor, vamp_weight: torch.Tensor
) -> distrib.MixtureSameFamily:
    """
    Creates a mixture of gaussian, using the log_var an mu tensor in entry.
    Each batch dim of mu and log_var must represent a component of the GMM.
    The Weights contol the importance of each component.
    """

    n_components = mu.size(0)
    if torch.isnan(mu).any():
        print("NaN detected in mu")
    if torch.isnan(log_var).any():
        print("NaN detected in log_var")

    # print("loss vamp ", vamp_weight.shape)
    # print("pseud_in ", mu.shape )

    # print("w ", vamp_weight.shape)
    # print("mu ", mu.shape)
    dist = distrib.MixtureSameFamily(
        distrib.Categorical(logits=vamp_weight),
        component_distribution=distrib.Independent(
            distrib.Normal(mu, (log_var / 2).exp()), 1
        ),
    )
    return dist


def create_distrib_posterior(
    mu: torch.Tensor, log_var: torch.Tensor
) -> distrib.Distribution:
    """
    Returns the gaussian posterior
    """
    return distrib.Independent(distrib.Normal(mu, (log_var / 2).exp()), 1)


def vamp_prior_kl_loss(
    z: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    pseudo_mu: torch.Tensor,
    pseudo_log_var: torch.Tensor,
    vamp_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the kl loss for the vamp_prior implementation
    """
    # print("Z ", z.shape)
    prior = create_mixture(pseudo_mu, pseudo_log_var, vamp_weight)
    posterior = create_distrib_posterior(mu, log_var)
    log_prior = prior.log_prob(z)
    log_posterior = posterior.log_prob(z)
    # print("log prob ",log_prior.shape)
    return log_posterior - log_prior


# Vamp prior loss
def VAE_vamp_prior_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pseudo_mu: torch.Tensor,
    pseudo_log_var: torch.Tensor,
    scale: torch.Tensor = None,
    vamp_weight: torch.Tensor = None,
    beta: float = 1,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    ELBO for the VampPrior implementations
    """

    recon_loss = negative_log_likelihood(x, x_recon, scale,mask)
    # Compute KL divergence
    batch_size = x.size(0)
    kl_loss = vamp_prior_kl_loss(
        z, mu, logvar, pseudo_mu, pseudo_log_var, vamp_weight
    )
    return recon_loss.mean() + beta * kl_loss.mean()


def CVAE_vamp_prior_loss_label_weights(
    x: torch.Tensor,
    weights: torch.Tensor,
    x_recon: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pseudo_mu: torch.Tensor,
    pseudo_log_var: torch.Tensor,
    scale: torch.Tensor = None,
    vamp_weight: torch.Tensor = None,
    beta: float = 1,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    ELBO for the VampPrior implementations
    """

    recon_loss = negative_log_likelihood(x, x_recon, scale, mask)
    # Compute KL divergence
    batch_size = x.size(0)
    kl_loss = vamp_prior_kl_loss(
        z, mu, logvar, pseudo_mu, pseudo_log_var, vamp_weight
    )
    # print("recon_loss shape", recon_loss.shape)
    # print("kl_loss shape", kl_loss.shape)
    loss = weights * recon_loss + beta * kl_loss
    return loss.mean()
