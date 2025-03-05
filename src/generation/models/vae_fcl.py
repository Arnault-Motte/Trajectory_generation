# %%
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

sys.executable


# %%
def get_data_loader(data: np.ndarray, batch_size: int) -> DataLoader:
    data2 = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data2)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %%
# Fully connected layer encoder
class VAE_encoder_FCL(nn.Module):
    def __init__(
        self, input_dim : int, hidden_dim_1 : int, hidden_dim_2: int, latent_dim:int, dropout_prob:float=0.0
    ) -> None:
        super(VAE_encoder_FCL, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim_2, latent_dim),
        )
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.layers(x)
        mu = self.mu(res)
        log_var = self.log_var(res)
        return mu, log_var


# %%
# same as encoder but reversed
# defining the decoder
class VAE_decoder_FCL(nn.Module):
    def __init__(
        self, output_dim:int, hidden_dim_1:int, hidden_dim_2:int, latent_dim:int, dropout_prob:float=0.0
    ) -> None:
        super(VAE_decoder_FCL, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim_1, output_dim),
        )

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        result = self.layers(z)
        return result


# %%
# Loss function
def reconstruction_loss(x:torch.Tensor, x_recon:torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x, reduction="sum")
def gaussian_likelihood_learned_variance(x:torch.Tensor, x_reco:torch.Tensor, scale:torch.Tensor=None) -> torch.Tensor:
    mean = x_reco
    stddev = torch.Tensor([1e-2]).to(x.device)
    if scale:
        stddev = torch.exp(scale).to(x.device)
    dist = torch.distributions.Normal(mean, stddev)
    log_pxz = dist.log_prob(x)
    dims = [i for i in range(1, len(x.size()))]
    log_like =  log_pxz.sum(dims)  #  negative log-likelihood
    return -log_like  # Minimize negative log-likelihood*

def kl_loss( mu: torch.Tensor, logvar: torch.Tensor)->torch.Tensor:
        var = torch.exp(logvar)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
        return kl_divergence

def vae_fcl_loss(x:torch.Tensor, x_recon:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor, scale:torch.Tensor=None, beta:float=1)->float:
    recon_loss = gaussian_likelihood_learned_variance(x, x_recon, scale)
    # Compute KL divergence
    batch_size = x.size(0)
    kl = kl_loss(mu,logvar)/batch_size
    #return recon_loss.mean() + beta*kl
    return reconstruction_loss(x,x_recon) + beta*kl




# %%
# Define the AE model
class VAE_FCL(nn.Module):
    def __init__(
        self, hidden_dim_1:int, hidden_dim_2:int, latent_dim:int, seq_len:int, num_channels:int
    ) -> None:
        super(VAE_FCL, self).__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        # Encoder: Maps input to latent space (mean and log variance)
        self.encoder = VAE_encoder_FCL(
            num_channels * seq_len, hidden_dim_1, hidden_dim_2, latent_dim
        )

        # Decoder: Maps latent space back to input space
        self.decoder = VAE_decoder_FCL(
            num_channels * seq_len, hidden_dim_1, hidden_dim_2, latent_dim
        )

        self.trained = False

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        mu,log_var =  self.encoder(x)
        return mu,log_var

    def decode(self, z:torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    #Reparametrization trick
    def reparametrize(self, mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    #forward pass
    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        x = x.view(x.size(0), -1)
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon.view(x.size(0), self.seq_len, self.num_channels),mu,log_var

    def fit(self, x: np.ndarray, epochs:int=1000, lr:float=1e-3, batch_size:int=500)-> None:
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataloader = get_data_loader(x, batch_size)

        for epoch in range(epochs):
            total_loss = 0.0
            likehood = 0.0
            kl_div = 0.0
            total_mse = 0.0
            for x_batch in dataloader:

                x_batch = x_batch[0].to(next(self.parameters()).device)
                cur_size = x_batch.size(0) 
                x_recon,mu,logvar = self(x_batch)
                x_recon = x_recon.view(
                    x_recon.size(0), self.seq_len, self.num_channels
                )

                loss = vae_fcl_loss(x_batch,x_recon,mu,logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                kl = kl_loss(mu,logvar)
                like = gaussian_likelihood_learned_variance(x_batch,x_recon)
                mse = reconstruction_loss(x_batch,x_recon)
                likehood += like.mean()
                kl_div += kl/cur_size
                total_mse += mse

                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f},MSE: {total_mse:.4f}, Likehood: {likehood:.4f}, KL: {kl_div:.4f} ")

        self.trained = True

    def reproduce(self, x:torch.Tensor) ->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        if not self.trained:
            raise Exception("Model not trained yet")
        self.eval()
        return self.forward(x)

    #Reproduce n_batch batches through the model. The data is loaded from the data loader
    def reproduce_data(self,data:np.ndarray, batch_size : int,n_batch:int) -> torch.Tensor:
        if not self.trained:
            raise Exception("Model not trained yet")
        data1 = get_data_loader(data, batch_size)
        i = 0
        batches_reconstructed = []
        for batch in data1:
            if i == n_batch:
                break
            b = batch[0].to(self.encoder.layers[0].weight.device)
            x_recon,_,_ = self.reproduce(b)
            batches_reconstructed.append(x_recon)
            i += 1
        reproduced_data = torch.cat(batches_reconstructed,dim=0)
        return reproduced_data



