import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split

from data_orly.src.core.early_stop import Early_stopping

import numpy as np

from traffic.core import tqdm


# Creates a train and test dataloader from a numpy array.
def get_data_loader(
    data: np.ndarray, batch_size: int, train_split: float = 0.8, num_worker=4
) -> tuple[DataLoader, DataLoader]:
    data2 = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data2)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_worker
    )
    test_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, num_workers=num_worker
    )
    return train_loader, test_loader


# %%
# MSE loss for reconstruction
def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x, reduction="sum")


def negative_log_likehood(
    x: torch.Tensor, recon: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    mu = recon
    dist = distrib.Normal(mu, scale)
    log_likelihood = dist.log_prob(x)
    return -log_likelihood.sum(dim=[i for i in range(1, len(x.size()))])


# Normal KL loss for gaussian prior
def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
    return kl_divergence


def create_mixture(
    mu: torch.Tensor, log_var: torch.Tensor, vamp_weight: torch.Tensor
) -> distrib.MixtureSameFamily:
    n_components = mu.size(0)
    if torch.isnan(mu).any():
        print("NaN detected in mu")
    if torch.isnan(log_var).any():
        print("NaN detected in log_var")
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
    return distrib.Independent(distrib.Normal(mu, (log_var / 2).exp()), 1)


def vamp_prior_kl_loss(
    z: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    pseudo_mu: torch.Tensor,
    pseudo_log_var: torch.Tensor,
    vamp_weight: torch.Tensor,
) -> torch.Tensor:
    prior = create_mixture(pseudo_mu, pseudo_log_var, vamp_weight)
    posterior = create_distrib_posterior(mu, log_var)
    log_prior = prior.log_prob(z)
    log_posterior = posterior.log_prob(z)
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
) -> torch.Tensor:
    recon_loss = negative_log_likehood(x, x_recon, scale)
    # Compute KL divergence
    batch_size = x.size(0)
    kl_loss = vamp_prior_kl_loss(
        z, mu, logvar, pseudo_mu, pseudo_log_var, vamp_weight
    )
    return recon_loss.mean() + beta * kl_loss.mean()


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
    ):
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


# Residual TCN block
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
    ):
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


# TCN formed of n_block blocks
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


## Encoder
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
        self.flaten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.avgpool(x)
        x = self.flaten(x)
        mu = self.mu_layer(x)
        log_var = self.logvar_layer(x)
        return mu, log_var


## Decoder
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_dense(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x = self.upsample(x)
        x = self.tcn(x)
        return x


# VampPrior Pseudo inputs generator
class Pseudo_inputs_generator(nn.Module):
    def __init__(
        self,
        number_of_channels: int,
        number_of_points: int,
        pseudo_inputs_num: int,
        dropout: float,
    ) -> None:
        super(Pseudo_inputs_generator, self).__init__()
        self.number_of_channels = number_of_channels
        self.number_of_points = number_of_points
        self.pseudo_inputs_num = pseudo_inputs_num
        self.first_layer = nn.Linear(pseudo_inputs_num, pseudo_inputs_num)
        self.relu = nn.ReLU()
        self.second_layer = nn.Linear(
            pseudo_inputs_num, number_of_channels * number_of_points
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self) -> torch.Tensor:
        x = torch.eye(self.pseudo_inputs_num, self.pseudo_inputs_num).to(
            next(self.parameters()).device
        )
        pseudo_inputs = self.first_layer(x)
        pseudo_inputs = self.relu(pseudo_inputs)
        pseudo_inputs = self.dropout(pseudo_inputs)
        pseudo_inputs = self.second_layer(pseudo_inputs)
        pseudo_inputs = self.dropout(pseudo_inputs)
        return pseudo_inputs.view(
            -1, self.number_of_channels, self.number_of_points
        )


# Model
class VAE_TCN_Vamp(nn.Module):
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
        pseudo_input_num: int = 800,
        early_stopping: bool = False,
        patience: int = 100,
        min_delta: float = -100.0,
        num_workers: int = 4,
        temp_name: str = "best_model.pth",
        init_std:float = 1.0,
    ):
        super(VAE_TCN_Vamp, self).__init__()

        self.temp_name = temp_name
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
        self.pseudo_inputs_layer = Pseudo_inputs_generator(
            in_channels, seq_len, pseudo_input_num, dropout=dropout
        )
        self.seq_len = seq_len
        self.trained = False
        self.in_channels = in_channels
        self.log_std = nn.Parameter(torch.Tensor([init_std]), requires_grad=True)

        self.prior_weights = nn.Parameter(
            torch.ones((1, pseudo_input_num)), requires_grad=True
        )
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.num_workers = num_workers

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        return mu, log_var

    def pseudo_inputs_latent(self) -> tuple[torch.Tensor, torch.Tensor]:
        pseudo_inputs = self.pseudo_inputs_layer.forward()
        mu, log_var = self.encode(pseudo_inputs)
        return mu, log_var

    def reparametrize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mu_pseudo, log_var_pseudo = self.pseudo_inputs_latent()
        x = x.permute(0, 2, 1)  # 500 3 200
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var).to(next(self.parameters()).device)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z, mu, log_var, mu_pseudo, log_var_pseudo

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
        dataloader_train, data_loader_val = get_data_loader(
            x, batch_size, num_worker=self.num_workers
        )
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        early_stopping = (
            Early_stopping(
                patience=self.patience,
                min_delta=self.min_delta,
                best_model_path="data_orly/src/generation/models/saved_temp/"
                + self.temp_name,
            )
            if self.early_stopping
            else None
        )

        for epoch in tqdm(range(epochs),bar_format="{desc}: {percentage:3.0f}% | ETA: {remaining}"):
            total_loss = 0.0
            kl_div = 0.0
            total_mse = 0.0
            for x_batch in dataloader_train:
                self.train()
                x_batch = x_batch[0].to(next(self.parameters()).device)
                x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self(
                    x_batch
                )
                loss = VAE_vamp_prior_loss(
                    x_batch.permute(0, 2, 1),
                    x_recon,
                    z,
                    mu,
                    log_var,
                    pseudo_mu,
                    pseudo_log_var,
                    scale=self.log_std,
                    vamp_weight=self.prior_weights,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                kl = vamp_prior_kl_loss(
                    z,
                    mu,
                    log_var,
                    pseudo_mu,
                    pseudo_log_var,
                    vamp_weight=self.prior_weights,
                )
                mse = reconstruction_loss(x_batch.permute(0, 2, 1), x_recon)
                cur_size = x_batch.size(0)
                total_mse += mse.item() / cur_size
                kl_div += kl.mean().item()
            scheduler.step()
            # validation
            val_loss, val_kl, val_recons = self.compute_val_loss(
                data_loader_val
            )

            total_loss = total_loss / len(dataloader_train)
            total_mse = total_mse / len(dataloader_train)
            kl_div = kl_div / len(dataloader_train)
            val_loss = val_loss / len(data_loader_val)
            val_kl = val_kl / len(data_loader_val)
            val_recons = val_recons / len(data_loader_val)
            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}]\n Train set, Loss: {total_loss:.4f},MSE: {total_mse:.4f}, KL: {kl_div:.4f}, Log_like: {total_loss - kl_div:.4f},scale: {self.log_std.item():.4f} "
                )
                print(
                    f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f}, Log_like: {val_loss - val_kl:.4f},scale: {self.log_std.item():.4f} "
                )
            if epoch == epochs - 1:
                print(
                    f"Epoch [{epoch + 1}/{epochs}]\n Train set, Loss: {total_loss:.4f},MSE: {total_mse:.4f}, KL: {kl_div:.4f}, Log_like: {total_loss - kl_div:.4f} "
                )
                print(
                    f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f} "
                )

            if self.early_stopping:
                early_stopping(val_loss, self)
                if early_stopping.stop:
                    self.trained = True
                    print(
                        f"Epoch [{epoch + 1}/{epochs}]\n Train set, Loss: {total_loss:.4f},MSE: {total_mse:.4f}, KL: {kl_div:.4f}, Log_like: {total_loss - kl_div:.4f} "
                    )
                    print(
                        f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f} "
                    )
                    print(
                        f"Best model loss: {early_stopping.min_loss} \n |--Loading best model--|"
                    )
                    self.load_model(early_stopping.path)
                    break
        if self.early_stopping:
            self.load_model(early_stopping.path)
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
            x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self(x_batch)
            loss = VAE_vamp_prior_loss(
                x_batch.permute(0, 2, 1),
                x_recon,
                z,
                mu,
                log_var,
                pseudo_mu,
                pseudo_log_var,
                scale=self.log_std,
                vamp_weight=self.prior_weights,
            )
            total_loss += loss.item()
            kl = vamp_prior_kl_loss(
                z,
                mu,
                log_var,
                pseudo_mu,
                pseudo_log_var,
                vamp_weight=self.prior_weights,
            )
            mse = reconstruction_loss(x_batch.permute(0, 2, 1), x_recon)
            cur_size = x_batch.size(0)
            total_mse += mse.item() / cur_size
            kl_div += kl.mean().item()
        return total_loss, kl_div, total_mse

    def reproduce_data(
        self, data: np.ndarray, batch_size: int, n_batch: int
    ) -> torch.Tensor:
        if not self.trained:
            raise Exception("Model not trained yet")
        data1, _ = get_data_loader(
            data, batch_size, 0.8, num_worker=self.num_workers
        )
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
        return reproduced_data, data1

    def sample_from_prior(self, num_sample: int = 1) -> torch.Tensor:
        # getting the prior
        with torch.no_grad():
            mu, log_var = self.pseudo_inputs_latent()
        distrib = create_mixture(
            mu, log_var, vamp_weight=self.prior_weights.squeeze(0)
        )
        samples = distrib.sample((num_sample,))
        return samples

    def sample(self, num_samples: int, batch_size: int) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        samples = []
        for _ in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - len(samples))
            z = self.sample_from_prior(current_batch_size).to(device)
            generated_data = self.decode(z)
            samples.append(generated_data)
        final_samples = torch.cat(samples)
        return final_samples.permute(0, 2, 1)

    def save_model(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    def load_model(self, weight_file_path: str) -> None:
        self.load_state_dict(torch.load(weight_file_path))
        self.trained = True
    
    def generate_from_specific_vamp_prior(
        self, vamp_index: int, num_traj: int
    ) -> torch.Tensor:
        with torch.no_grad():
            psuedo_means, pseudo_scales = self.pseudo_inputs_latent()
            chosen_mean = psuedo_means[vamp_index]
            chosen_scale = (pseudo_scales[vamp_index] / 2).exp()
            print(chosen_mean.shape)
            print(chosen_scale.shape)

            dist = distrib.Independent(
                distrib.Normal(chosen_mean, chosen_scale), 1
            )

            sample = dist.sample(torch.Size([num_traj])).to(
                next(self.parameters()).device
            )
            print(sample.shape)
            generated_traj = self.decode(sample)

        return generated_traj.permute(0, 2, 1)

    def get_pseudo_inputs_recons(self) -> torch.Tensor:
        with torch.no_grad():
            pseudo_inputs = self.pseudo_inputs_layer.forward().permute(0, 2, 1)
            output = self.forward(pseudo_inputs)[0].permute(
                0, 2, 1
            )
        return output
