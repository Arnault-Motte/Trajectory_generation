# %%
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np

sys.executable


# %%
def get_data_loader(
    data: np.ndarray, batch_size: int, train_split: float = 0.8
) -> DataLoader:
    data2 = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data2)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


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
def reconstruction_loss(x:torch.Tensor, x_recon:torch.Tensor) -> float:
    return F.mse_loss(x_recon, x, reduction="sum")


def kl_loss( mu: torch.Tensor, logvar: torch.Tensor)->torch.Tensor:
        var = torch.exp(logvar)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
        return kl_divergence

def vae_fcl_loss(x:torch.Tensor, x_recon:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor, scale:torch.Tensor=None, beta:float=1)->torch.Tensor:
    recon_loss = reconstruction_loss(x,x_recon)
    # Compute KL divergence
    batch_size = x.size(0)
    kl = kl_loss(mu,logvar)/batch_size
    #return recon_loss.mean() + beta*kl
    return  recon_loss/batch_size + beta*kl




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
        self.latent_dim = latent_dim

        self.trained = False

    def encode(self, x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
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
        dataloader,test_data = get_data_loader(x, batch_size)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
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
                mse = reconstruction_loss(x_batch,x_recon)
                kl_div += kl/cur_size
                total_mse += mse/cur_size

                total_loss += loss.item()
            val_loss, val_kl, val_recons = self.compute_val_loss(
                test_data
            )
            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}]\n Train set, Loss: {total_loss:.4f},MSE: {total_mse:.4f}, KL: {kl_div:.4f} "
                )
                print(
                    f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f} "
                )
            if epoch == epochs - 1:
                print(
                    f"Epoch [{epoch + 1}/{epochs}]\n Train set, Loss: {total_loss:.4f},MSE: {total_mse:.4f}, KL: {kl_div:.4f} "
                )
                print(
                    f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f} "
                )
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
        data1,_ = get_data_loader(data, batch_size,train_split=0.8)
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
    
    def compute_val_loss(
        self, val_data: DataLoader
    ) -> tuple[float, float, float]:
        total_loss = 0.0
        kl_div = 0.0
        total_mse = 0.0
        self.eval()
        for x_batch in val_data:
            x_batch = x_batch[0].to(next(self.parameters()).device)
            x_recon, mu, log_var = self(x_batch)
            loss = vae_fcl_loss(x_batch, x_recon, mu, log_var)
            total_loss += loss.item()
            kl = kl_loss(mu, log_var)
            mse = reconstruction_loss(x_batch, x_recon)
            cur_size = x_batch.size(0)
            total_mse += mse/cur_size

            kl_div += kl / cur_size
        return total_loss, kl_div, total_mse

    def sample(self, num_samples: int, batch_size: int = 500) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        samples = []
        for _ in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - len(samples))
            z = torch.randn(
                current_batch_size, self.latent_dim
            ).to(device)
            generated_data = self.decode(z)
            samples.append(generated_data)
        samples = torch.cat(samples)
        return samples



