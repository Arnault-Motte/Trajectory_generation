import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np


def get_data_loader(
    data: np.ndarray, batch_size: int, train_split: float = 0.8
) -> tuple[DataLoader, DataLoader]:
    data2 = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data2)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# %%
# Loss function
def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x, reduction="sum")


def negative_log_likehood(
    x: torch.Tensor, recon: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    mu = recon
    dist = distrib.Normal(mu, scale)
    log_likelihood = dist.log_prob(x)  # Calculate log-likelihood for each pixel
    return -log_likelihood.sum(
        dim=[i for i in range(1, len(x.size()))]
    )  # Return negative log-likelihood (as loss)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
    return kl_divergence


def other_kl_loss(
    z: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    n = mu.size(1)
    device = mu.device
    prior_mu = torch.zeros((1, n)).to(device)
    prior_log_var = torch.zeros((1, n)).to(device)
    prior = distrib.Independent(
        distrib.Normal(prior_mu, (prior_log_var / 2).exp()), 1
    )
    posterior = distrib.Independent(distrib.Normal(mu, (log_var / 2).exp()), 1)
    log_prior = prior.log_prob(z)
    log_posterior = posterior.log_prob(z)

    return log_posterior - log_prior


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    scale: torch.Tensor = None,
    beta: float = 1,
) -> torch.Tensor:
    recon_loss = negative_log_likehood(x, x_recon, scale)
    # Compute KL divergence
    batch_size = x.size(0)
    kl = kl_loss(mu, logvar) / batch_size
    return recon_loss.mean() + beta * kl


def manul_recons(
    x: torch.Tensor, x_recon: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    dim = x.size(1) * x.size(2)
    batch = x.size(0)
    temp_scale = torch.exp(scale) ** 2
    test = (
        -1
        / 2
        * dim
        * batch
        * torch.log(2 * torch.Tensor([torch.pi]).to(x.device))
        - 1 / 2 * dim * batch * torch.log(temp_scale)
        - 1 / (2 * temp_scale.item()) * F.mse_loss(x, x_recon, reduction="sum")
    )
    return test


# Basic TCN blocks
class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        active: bool = True,
    ) -> None:
        super(TCNBlock, self).__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
            )
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.left_padding = (kernel_size - 1) * dilation
        self.active = active
        self.init_weights()

    def init_weights(self) -> None:
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0), "constant", 0)
        x = self.conv(x)
        x = self.relu(x) if self.active else x
        x = self.dropout(x)
        return x


class TCN_residual_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        last: bool = False,
    ) -> None:
        super(TCN_residual_block, self).__init__()
        self.tcn1 = TCNBlock(
            in_channels, out_channels, kernel_size, stride, dilation, dropout
        )
        active = True if not last else False
        self.tcn2 = TCNBlock(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            dropout,
            active=active,
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )  # we can't do it for the first block
        # self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.tcn1(x)
        out = self.tcn2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out = out + residual
        # out = self.relu(out)
        return out


class TCN(nn.Module):
    def __init__(
        self,
        initial_channels: int,
        latent_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilatation: int,
        dropout: float,
        nb_blocks: int,
    ):
        super(TCN, self).__init__()
        self.first_block = TCN_residual_block(
            initial_channels,
            latent_channels,
            kernel_size,
            stride,
            dilatation**0,
            dropout,
        )
        self.layers = [
            TCN_residual_block(
                latent_channels,
                latent_channels,
                kernel_size,
                stride,
                dilatation ** (index + 1),
                dropout,
            )
            for index in range(nb_blocks - 2)
        ]
        self.blocks = nn.Sequential(*self.layers)
        self.last_block = TCN_residual_block(
            latent_channels,
            out_channels,
            kernel_size,
            stride,
            dilatation ** (nb_blocks - 1),
            dropout,
            last=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_block(x)
        x = self.blocks(x)
        x = self.last_block(x)
        return x


## Encoder decoder and model
class TCN_encoder(nn.Module):
    def __init__(
        self,
        inital_channels: int,
        latent_channels: int,
        latent_dim: int,
        kernel_size: int,
        stride: int,
        dilatation: int,
        dropout: float,
        nb_blocks: int,
        avgpoolnum: int,
        seq_len: int,
    ):
        super(TCN_encoder, self).__init__()
        self.tcn = TCN(
            inital_channels,
            latent_channels,
            latent_channels,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
        )
        self.avgpool = nn.AvgPool1d(avgpoolnum)
        self.dense_dim = latent_channels * seq_len // avgpoolnum
        self.mu_layer = nn.Linear(self.dense_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.dense_dim, latent_dim)
        self.hardtan = nn.Hardtanh(min_val=-6.0, max_val=2.0)
        self.flaten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tcn(x)
        x = self.avgpool(x)
        x = self.flaten(x)
        mu = self.mu_layer(x)
        log_var = self.logvar_layer(x)
        log_var = self.hardtan(log_var)
        return mu, log_var


class TCN_decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        kernel_size: int,
        stride: int,
        dilatation: int,
        dropout: float,
        nb_blocks: int,
        upsamplenum: int,
        seq_len: int,
    ):
        super(TCN_decoder, self).__init__()
        self.seq_len = seq_len
        self.first_dense = nn.Linear(
            latent_dim, in_channels * (seq_len // upsamplenum)
        )
        self.upsample = nn.Upsample(scale_factor=upsamplenum)
        self.tcn = TCN(
            in_channels,
            latent_dim,
            out_channels,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
        )
        self.sampling_factor = upsamplenum
        self.dense_log_std = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_std = self.dense_log_std(x)
        x = self.first_dense(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x = self.upsample(x)
        x = self.tcn(x)
        return x, log_std


class VAE_TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        kernel_size: int,
        stride: int,
        dilatation: int,
        dropout: float,
        nb_blocks: int,
        avgpoolnum: int,
        upsamplenum: int,
        seq_len: int = 200,
    ):
        super(VAE_TCN, self).__init__()
        self.encoder = TCN_encoder(
            in_channels,
            out_channels,
            latent_dim,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
            avgpoolnum,
            seq_len,
        )
        self.decoder = TCN_decoder(
            latent_dim,
            in_channels,
            latent_dim,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
            upsamplenum,
            seq_len,
        )
        self.seq_len = seq_len
        self.trained = False
        self.in_channels = in_channels
        self.log_std = nn.Parameter(torch.Tensor([1]), requires_grad=True)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        return mu, log_var

    def reparametrize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x, log_std = self.decoder(z)
        return x, log_std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)  # 500 3 200
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_reconstructed, log_std = self.decode(z)
        return x_reconstructed, mu, log_var, z, log_std

    def fit(
        self,
        x: np.ndarray,
        epochs: int = 1000,
        lr: float = 1e-3,
        batch_size: int = 500,
        step_size: int = 200,
        gamma: float = 0.5,
    ) -> None:
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataloader_train, data_loader_val = get_data_loader(x, batch_size)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(epochs):
            total_loss = 0.0
            kl_div = 0.0
            total_mse = 0.0
            total_manual_recons = 0.0
            kl_2 = 0.0
            for x_batch in dataloader_train:
                self.train()
                x_batch = x_batch[0].to(next(self.parameters()).device)
                x_recon, mu, log_var, z, _ = self(x_batch)
                # print(self.log_std)
                loss = vae_loss(
                    x_batch.permute(0, 2, 1),
                    x_recon,
                    mu,
                    log_var,
                    scale=self.log_std,
                )
                # loss = reconstruction_loss(x_batch.permute(0, 2, 1), x_recon)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                kl = kl_loss(mu, log_var)
                mse = reconstruction_loss(x_batch.permute(0, 2, 1), x_recon)
                cur_size = x_batch.size(0)
                total_mse += mse / cur_size
                kl_2 += other_kl_loss(z, mu, log_var).mean()
                kl_div += kl / cur_size
                recons_true = (
                    manul_recons(
                        x_batch.permute(0, 2, 1), x_recon, self.log_std
                    ).item()
                    / cur_size
                )
                total_manual_recons += recons_true
                # if (loss<0):
                #     print(mu[0],log_var[0],self.log_std.item())
                #     input("Press Enter to continue...")

            scheduler.step()
            # validation

            val_loss, val_kl, val_recons = self.compute_val_loss(
                data_loader_val
            )
            total_loss = total_loss / len(dataloader_train)
            total_mse = total_mse / len(dataloader_train)
            kl_div = kl_div / len(dataloader_train)
            kl_2 = kl_2 / len(dataloader_train)
            val_loss = val_loss / len(data_loader_val)
            val_kl = val_kl / len(data_loader_val)
            val_recons = val_recons / len(data_loader_val)
            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}]\n Train set, Loss: {total_loss:.4f},MSE: {total_mse:.4f}, KL: {kl_div:.4f}, Log_like: {total_loss - kl_div:.4f}, deviation {self.log_std.item():.4f}, true_kl = {kl_2:.4f}, manual_recons = {total_manual_recons:.4f} "
                )
                print(
                    f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f}, Log_like: {val_loss - val_kl:.4f} "
                )
            if epoch == epochs - 1:
                print(
                    f"Epoch [{epoch + 1}/{epochs}]\n Train set, Loss: {total_loss:.4f},MSE: {total_mse:.4f}, KL: {kl_div:.4f}, Log_like: {total_loss - kl_div:.4f} "
                )
                print(
                    f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f}, Log_like: {val_loss - val_kl:.4f} "
                )

        self.trained = True

    def reproduce(self, x: torch.Tensor) -> torch.Tensor:
        if not self.trained:
            raise Exception("Model not trained yet")
        self.eval()
        return self.forward(x)[0].permute(0, 2, 1)

    def compute_val_loss(
        self, val_data: DataLoader
    ) -> tuple[float, float, float]:
        total_loss = 0.0
        kl_div = 0.0
        total_mse = 0.0
        self.eval()
        for x_batch in val_data:
            x_batch = x_batch[0].to(next(self.parameters()).device)
            x_recon, mu, log_var, _, _ = self(x_batch)
            loss = vae_loss(
                x_batch.permute(0, 2, 1), x_recon, mu, log_var, self.log_std
            )
            total_loss += loss.item()
            kl = kl_loss(mu, log_var)
            mse = reconstruction_loss(x_batch.permute(0, 2, 1), x_recon)
            cur_size = x_batch.size(0)
            total_mse += mse.item() / cur_size
            kl_div += kl / cur_size
        return total_loss, kl_div, total_mse

    def reproduce_data(
        self, data: np.ndarray, batch_size: int, n_batch: int
    ) -> tuple[torch.Tensor, DataLoader]:
        if not self.trained:
            raise Exception("Model not trained yet")
        data1, _ = get_data_loader(data, batch_size, train_split=0.8)
        i = 0
        batches_reconstructed = []
        for batch in data1:
            if i == n_batch:
                break
            device = next(self.parameters()).device
            b = batch[0].to(device)
            x_recon = self.reproduce(b)
            batches_reconstructed.append(x_recon)
            i += 1
        reproduced_data = torch.cat(batches_reconstructed, dim=0)
        # print(reproduced_data)

        return reproduced_data, data1

    def sample_from_prior(self, num_samples: int = 1) -> torch.Tensor:
        return torch.randn(num_samples, self.encoder.mu_layer.out_features)

    def sample(self, num_samples: int, batch_size: int = 500) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        samples = []
        for _ in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - len(samples))
            z = self.sample_from_prior(current_batch_size).to(device)
            generated_data = self.decode(z)
            samples.append(generated_data)
        final_samples = torch.cat(samples)
        return final_samples

    def save_model(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    def load_model(self, weight_file_path: str) -> None:
        self.load_state_dict(torch.load(weight_file_path))
        self.trained = True
