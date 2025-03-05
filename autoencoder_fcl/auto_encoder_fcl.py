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
class AE_encoder_FCL(nn.Module):
    def __init__(
        self, input_dim : int, hidden_dim_1 : int, hidden_dim_2: int, latent_dim:int, dropout_prob:float=0.0
    ) -> None:
        super(AE_encoder_FCL, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim_2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.layers(x)
        return result


# %%
# same as encoder but reversed
# defining the decoder
class AE_decoder_FCL(nn.Module):
    def __init__(
        self, output_dim:int, hidden_dim_1:int, hidden_dim_2:int, latent_dim:int, dropout_prob:float=0
    ) -> None:
        super(AE_decoder_FCL, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim_1, output_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        result = self.layers(x)
        return result


# %%
# Loss function
def reconstruction_loss(x:torch.Tensor, x_recon:torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x, reduction="sum")


# %%
# Define the AE model
class AE_FCL(nn.Module):
    def __init__(
        self, hidden_dim_1:int, hidden_dim_2:int, latent_dim:int, seq_len:int, num_channels:int
    ) -> None:
        super(AE_FCL, self).__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        # Encoder: Maps input to latent space (mean and log variance)
        self.encoder = AE_encoder_FCL(
            num_channels * seq_len, hidden_dim_1, hidden_dim_2, latent_dim
        )

        # Decoder: Maps latent space back to input space
        self.decoder = AE_decoder_FCL(
            num_channels * seq_len, hidden_dim_1, hidden_dim_2, latent_dim
        )

        self.trained = False

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z:torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x_recon = self.encode(x)
        x_recon = self.decode(x_recon)
        return x_recon.view(x.size(0), self.seq_len, self.num_channels)

    def fit(self, x: np.ndarray, epochs:int=1000, lr:float=1e-3, batch_size:int=500) -> None:
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataloader = get_data_loader(x, batch_size)

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch in dataloader:
                x_batch = x_batch[0].to(next(self.parameters()).device)
                x_recon = self(x_batch)
                x_recon = x_recon.view(
                    x_recon.size(0), self.seq_len, self.num_channels
                )
                loss = reconstruction_loss(x_batch, x_recon)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

        self.trained = True

    def reproduce(self, x:torch.Tensor) -> torch.Tensor:
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
            x_recon = self.reproduce(b)
            batches_reconstructed.append(x_recon)
            i += 1
        reproduced_data = torch.cat(batches_reconstructed,dim=0)
        return reproduced_data



