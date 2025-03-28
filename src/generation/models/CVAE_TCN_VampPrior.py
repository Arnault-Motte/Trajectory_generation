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


# Creates a train and test dataloader from a numpy array.
def get_data_loader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    train_split: float = 0.8,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    data2 = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(data2, labels_tensor)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
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
            inital_channels + 1,  # adding the label dim
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

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_label = torch.cat([x, labels], dim=1)
        x_label = self.tcn(x_label)
        x_label = self.avgpool(x_label)
        x_label = self.flaten(x_label)
        mu = self.mu_layer(x_label)
        log_var = self.logvar_layer(x_label)
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
        labels_latent_dim: int,
    ):
        super(TCN_decoder, self).__init__()
        self.seq_len = seq_len
        self.first_dense = nn.Linear(
            latent_dim + labels_latent_dim,
            in_channels * (seq_len // upsamplenum),
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

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, labels], dim=1)
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


class Pseudo_labels_generator(nn.Module):
    def __init__(
        self, label_dim: int, pseudo_inputs_num: int, dropout: float
    ) -> None:
        super(Pseudo_labels_generator, self).__init__()
        self.pseudo_inputs_num = pseudo_inputs_num
        self.first_layer = nn.Linear(pseudo_inputs_num, pseudo_inputs_num)
        self.relu = nn.ReLU()
        self.second_layer = nn.Linear(pseudo_inputs_num, label_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self) -> torch.Tensor:
        x = torch.eye(self.pseudo_inputs_num, self.pseudo_inputs_num).to(
            next(self.parameters()).device
        )
        pseudo_labels = self.first_layer(x)
        pseudo_labels = self.relu(pseudo_labels)
        pseudo_labels = self.dropout(pseudo_labels)
        pseudo_labels = self.second_layer(pseudo_labels)
        pseudo_labels = self.dropout(pseudo_labels)
        return pseudo_labels


class Label_mapping(nn.Module):
    def __init__(
        self,
        one_hot: bool,
        label_dim: int,
        latent_dim: int,
        num_classes: int = None,
    ) -> None:
        super(Label_mapping, self).__init__()
        self.one_hot = one_hot
        self.label_dim = label_dim
        self.latent_dim = latent_dim

        self.embbedings = (
            nn.Embedding(num_classes, latent_dim) if not one_hot else None
        )
        self.fcl = nn.Linear(label_dim, latent_dim) if one_hot else None

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.fcl(labels) if self.fcl else self.embbedings(labels)


class Weight_Prior_Conditioned(nn.Module):
    def __init__(self, num_pseudo_inputs: int, labels_dim: int) -> None:
        super(Weight_Prior_Conditioned, self).__init__()
        self.num_pseudo_inputs = num_pseudo_inputs
        self.fc_layer = nn.Linear(labels_dim, num_pseudo_inputs)

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        label = self.fc_layer(label)
        return label.unsqueeze(0)


# Model
class CVAE_TCN_Vamp(nn.Module):
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
        label_dim: int,
        label_latent: int,
        seq_len: int = 200,
        pseudo_input_num: int = 800,
        early_stopping: bool = False,
        patience: int = 100,
        min_delta: float = -100.0,
        num_classes: int = None,
        conditioned_prior: bool = False,
        temp_save: str = "best_model.pth",
    ):
        super(CVAE_TCN_Vamp, self).__init__()

        self.temp_save = temp_save
        self.labels_encoder_broadcast = Label_mapping(
            num_classes is None,
            label_dim,
            seq_len,
            num_classes,
        )
        self.labels_decoder_broadcast = Label_mapping(
            num_classes is None, label_dim, label_latent, num_classes
        )

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
            labels_latent_dim=label_latent,
        )

        self.pseudo_inputs_layer = Pseudo_inputs_generator(
            in_channels, seq_len, pseudo_input_num, dropout=dropout
        )
        self.pseudo_labels_layer = Pseudo_labels_generator(
            label_dim, pseudo_input_num, dropout
        )

        self.seq_len = seq_len
        self.trained = False
        self.in_channels = in_channels
        self.log_std = nn.Parameter(torch.Tensor([1]), requires_grad=True)

        self.prior_weights = nn.Parameter(
            torch.ones((1, pseudo_input_num)), requires_grad=True
        )
        self.prior_weights_layers = (
            Weight_Prior_Conditioned(
                num_pseudo_inputs=pseudo_input_num, labels_dim=label_dim
            )
            if conditioned_prior
            else None
        )
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

    def is_conditioned(self)-> bool:
        return self.prior_weights_layers is not None
    def encode(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if label == None:
            label = torch.zeros(x.size(0),1,self.seq_len).to(next(self.parameters()).device)
        else:
            label = self.labels_encoder_broadcast(label)
            label = label.view(-1, 1, self.seq_len)
        mu, log_var = self.encoder(x, label)
        return mu, log_var

    def pseudo_inputs_latent(self) -> tuple[torch.Tensor, torch.Tensor]:
        pseudo_labels = self.pseudo_labels_layer.forward()
        pseudo_inputs = self.pseudo_inputs_layer.forward()
        mu, log_var = self.encode(pseudo_inputs, pseudo_labels)
        return mu, log_var

    def reparametrize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = self.labels_decoder_broadcast(label)
        x = self.decoder(z, label)
        return x

    def get_prior_weight(self, labels: torch.Tensor = None) -> torch.Tensor:
        if self.prior_weights_layers and not labels:
            raise ValueError(
                "The prior need to be conditioned, please properly enter the labels when getting the prior weights"
            )

        if self.prior_weights_layers:
            return self.prior_weights_layers(labels)
        else:
            return self.prior_weights

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        conditioned = self.is_conditioned()
        labels_encoder = labels
        if not conditioned:
            labels_encoder = None
        mu_pseudo, log_var_pseudo = self.pseudo_inputs_latent()
        x = x.permute(0, 2, 1)  # 500 3 200
        mu, log_var = self.encode(x, labels_encoder)
        z = self.reparametrize(mu, log_var).to(next(self.parameters()).device)
        x_reconstructed = self.decode(z, labels)
        return x_reconstructed, z, mu, log_var, mu_pseudo, log_var_pseudo

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        epochs: int = 1000,
        lr: float = 1e-3,
        batch_size: int = 500,
        step_size: int = 200,
        gamma: float = 0.5,
    ) -> None:
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataloader_train, data_loader_val = get_data_loader(
            x, labels, batch_size
        )
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        early_stopping = (
            Early_stopping(
                patience=self.patience,
                min_delta=self.min_delta,
                best_model_path="data_orly/src/generation/models/saved_temp/"
                + self.temp_save,
            )
            if self.early_stopping
            else None
        )

        for epoch in range(epochs):
            total_loss = 0.0
            kl_div = 0.0
            total_mse = 0.0
            for x_batch, labels_batch in dataloader_train:
                self.train()
                x_batch = x_batch.to(next(self.parameters()).device)
                labels_batch = labels_batch.to(next(self.parameters()).device)
                x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self(
                    x_batch, labels_batch
                )
                prior_weight = (
                    self.prior_weights_layers(labels_batch)
                    if self.prior_weights_layers
                    else self.prior_weights
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
                    vamp_weight=prior_weight,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                prior_weights = (
                    self.prior_weights_layers(labels_batch)
                    if self.prior_weights_layers
                    else self.prior_weights
                )
                kl = vamp_prior_kl_loss(
                    z,
                    mu,
                    log_var,
                    pseudo_mu,
                    pseudo_log_var,
                    vamp_weight=prior_weights,
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

    def reproduce(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.trained:
            raise Exception("Model not trained yet")
        self.eval()
        return self.forward(x, labels)[0].permute(0, 2, 1)

    def compute_val_loss(
        self, val_data: DataLoader
    ) -> tuple[float, float, float]:
        total_loss = 0.0
        kl_div = 0.0
        total_mse = 0.0
        self.eval()
        for x_batch, labels in val_data:
            x_batch = x_batch.to(next(self.parameters()).device)
            labels = labels.to(next(self.parameters()).device)
            x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self(
                x_batch, labels
            )
            prior_weights = (
                self.prior_weights_layers(labels)
                if self.prior_weights_layers
                else self.prior_weights
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
                vamp_weight=prior_weights,
            )
            total_loss += loss.item()
            kl = vamp_prior_kl_loss(
                z,
                mu,
                log_var,
                pseudo_mu,
                pseudo_log_var,
                vamp_weight=prior_weights,
            )
            mse = reconstruction_loss(x_batch.permute(0, 2, 1), x_recon)
            cur_size = x_batch.size(0)
            total_mse += mse.item() / cur_size
            kl_div += kl.mean().item()
        return total_loss, kl_div, total_mse

    def reproduce_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        n_batch: int,
    ) -> tuple[torch.Tensor, DataLoader]:
        if not self.trained:
            raise Exception("Model not trained yet")
        data1, _ = get_data_loader(data, labels, batch_size, 0.8, shuffle=False)
        i = 0
        batches_reconstructed = []
        for batch, labels_batch in data1:
            if i == n_batch:
                break
            device = next(self.parameters()).device
            b = batch.to(device)
            labels_batch = labels_batch.to(device)
            x_recon = self.reproduce(b, labels_batch)
            batches_reconstructed.append(x_recon)
            i += 1
        reproduced_data = torch.cat(batches_reconstructed, dim=0)
        return reproduced_data, data1

    def sample_from_prior(
        self, num_sample: int = 1, labels: torch.Tensor = None
    ) -> torch.Tensor:
        # getting the prior
        with torch.no_grad():
            mu, log_var = self.pseudo_inputs_latent()

        distrib = create_mixture(
            mu, log_var, vamp_weight=self.prior_weights.squeeze(0)
        )
        samples = distrib.sample((num_sample,))
        return samples

    def sample_from_conditioned_prior(
        self, num_sample: int = 1, label: torch.Tensor = None
    ) -> torch.Tensor:
        # getting the prior
        with torch.no_grad():
            prior_weights = (
                self.prior_weights_layers(label)
                if self.prior_weights_layers
                else self.prior_weights
            )
            print("prior_weight", prior_weights.shape)
            print("sizes = ", )
            mu, log_var = self.pseudo_inputs_latent()
            print("mu", mu.shape)
            print("log_var ", log_var.shape)
        distrib = create_mixture(
            mu, log_var, vamp_weight=prior_weights.squeeze(0)
        )
        samples = distrib.sample((num_sample,))
        return samples

    def sample(
        self, num_samples: int, batch_size: int, labels: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        samples = []
        for _ in range(0, num_samples, batch_size):
            print("sizes fun : ", labels.shape, batch_size)
            current_batch_size = min(batch_size, num_samples - len(samples))
            z = self.sample_from_conditioned_prior(
                current_batch_size, labels[0]
            ).to(device)

            generated_data = self.decode(z, labels)
            samples.append(generated_data)
        final_samples = torch.cat(samples)
        return final_samples.permute(0, 2, 1)

    def save_model(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    def load_model(self, weight_file_path: str) -> None:
        self.load_state_dict(torch.load(weight_file_path))
        self.trained = True

    def get_pseudo_inputs_recons(self) -> torch.Tensor:
        with torch.no_grad():
            pseudo_labels = self.pseudo_labels_layer.forward()
            pseudo_inputs = self.pseudo_inputs_layer.forward().permute(0, 2, 1)
            output = self.forward(pseudo_inputs, pseudo_labels)[0].permute(
                0, 2, 1
            )
        return output

    def generate_from_specific_vamp_prior(
        self, vamp_index: int, num_traj: int
    ) -> torch.Tensor:
        with torch.no_grad():
            psuedo_means, pseudo_scales = self.pseudo_inputs_latent()
            chosen_mean = psuedo_means[vamp_index]
            chosen_scale = (pseudo_scales[vamp_index] / 2).exp()
            labels = self.pseudo_labels_layer()[vamp_index]
            print(chosen_mean.shape)
            print(chosen_scale.shape)
            print(labels.shape)

            dist = distrib.Independent(
                distrib.Normal(chosen_mean, chosen_scale), 1
            )
            sample = dist.sample(torch.Size([num_traj])).to(
                next(self.parameters()).device
            )
            labels = labels.repeat(num_traj, 1)

            print(labels.shape)
            print(sample.shape)
            generated_traj = self.decode(sample, labels)

        return generated_traj.permute(0, 2, 1)
