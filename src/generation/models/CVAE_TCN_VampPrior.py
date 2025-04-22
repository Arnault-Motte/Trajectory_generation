import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
from data_orly.src.core.early_stop import Early_stopping
from data_orly.src.core.loss import *
from data_orly.src.core.networks import *


## Encoder
class TCN_encoder(nn.Module):
    """
    VAE encoder using TCN. The number of TCN blocks
    can be adjusted.
    """

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
    """
    VAE decoder unsing TCNs. The number of TCN blocks
    can be adjusted.
    """

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

class Pseudo_labels_generator(nn.Module):
    """
    Model used to generate the pseudo labels.
    Formed of two fully connected layers.
    Used for CVAEs using VampPrior.
    """

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
    """
    Model use to map the labels to the expected
    latent dim.
    """

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
    """
    Model used to learn the weights of the prior components
    weights based on the label. Used for CVAE with a conditioned VampPrior.
    """

    def __init__(self, num_pseudo_inputs: int, labels_dim: int) -> None:
        super(Weight_Prior_Conditioned, self).__init__()
        self.num_pseudo_inputs = num_pseudo_inputs
        self.fc_layer = nn.Linear(labels_dim, num_pseudo_inputs)

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        label = self.fc_layer(label)
        return label.unsqueeze(0)


# Model
class CVAE_TCN_Vamp(nn.Module):
    """
    CVAE with a VampPrior.
    The user can control if the prior is conditioned or not.
    """

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
        num_worker: int = 4,
        init_std: float = 1,
    ):
        super(CVAE_TCN_Vamp, self).__init__()
        self.num_worker = num_worker
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
        self.log_std = nn.Parameter(torch.Tensor([init_std]), requires_grad=True)

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

    def is_conditioned(self) -> bool:
        """
        Returns true if the prior is conditioned
        """
        return self.prior_weights_layers is not None

    def encode(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the data x in entry to train the model"
        """
        if label is None:
            label = torch.zeros(x.size(0), 1, self.seq_len).to(
                next(self.parameters()).device
            )
        else:
            label = self.labels_encoder_broadcast(label)
            label = label.view(-1, 1, self.seq_len)
        mu, log_var = self.encoder(x, label)
        return mu, log_var

    def pseudo_inputs_latent(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and log variance associated with each
        pseudo input.
        """
        conditioned = self.is_conditioned()
        pseudo_labels = (
            self.pseudo_labels_layer.forward() if conditioned else None
        )
        pseudo_inputs = self.pseudo_inputs_layer.forward()
        mu, log_var = self.encode(pseudo_inputs, pseudo_labels)
        return mu, log_var

    def reparametrize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the reparametrization trick.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Generates the output from the sampled vector and the chosen label.
        """
        label = self.labels_decoder_broadcast(label)
        x = self.decoder(z, label)
        return x

    def get_prior_weight(self, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Returns the weights of each component of the prior.
        """
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
        """
        Fits the model to the data x in entry of the function. Early stopping is
        always activated.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataloader_train, data_loader_val = get_data_loader(
            x, labels, batch_size, num_worker=self.num_worker
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
        """
        Performs the forward path for the provided x and label.
        Returns the reconstructed trajectories in a tensor.
        """
        if not self.trained:
            raise Exception("Model not trained yet")
        self.eval()
        return self.forward(x, labels)[0].permute(0, 2, 1)

    def compute_val_loss(
        self, val_data: DataLoader
    ) -> tuple[float, float, float]:
        """
        Computes the validation loss.
        Returns in order : the total loss, the kl div and the mse
        """
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
        """
        Generate the reproducted trajectories for every
        trajectory of data.
        Also returns the data_loader used
        """
        if not self.trained:
            raise Exception("Model not trained yet")
        data1, _ = get_data_loader(
            data,
            labels,
            batch_size,
            0.8,
            shuffle=False,
            num_worker=self.num_worker,
        )
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

    def sample_from_prior(self, num_sample: int = 1) -> torch.Tensor:
        """
        Samples random points from the unconditioned prior
        """
        # getting the prior
        if self.is_conditioned():
            raise SyntaxError(
                "You can't sample with this function if your model uses a conditioned prior"
            )
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
        """
        Samples random points from the conditioned prior
        """
        with torch.no_grad():
            prior_weights = (
                self.prior_weights_layers(label)
                if self.prior_weights_layers
                else self.prior_weights
            )
            #print("prior_weight", prior_weights.shape)
            
            mu, log_var = self.pseudo_inputs_latent()
            #print("mu", mu.shape)
            #print("log_var ", log_var.shape)
        distrib = create_mixture(
            mu, log_var, vamp_weight=prior_weights.squeeze(0)
        )
        samples = distrib.sample((num_sample,))
        return samples

    def sample(
        self, num_samples: int, batch_size: int, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates num_samples trajectory from the VAE
        """
        self.eval()
        device = next(self.parameters()).device
        samples = []
        for _ in range(0, num_samples, batch_size):
            #print("sizes fun : ", labels.shape, batch_size)
            current_batch_size = min(batch_size, num_samples - len(samples))
            z = self.sample_from_conditioned_prior(
                current_batch_size, labels[0]
            ).to(device)

            generated_data = self.decode(z, labels).detach().cpu()
            samples.append(generated_data)
        final_samples = torch.cat(samples)
        return final_samples.permute(0, 2, 1)

    def save_model(self, file_path: str) -> None:
        """
        Saves the model weights
        """
        torch.save(self.state_dict(), file_path)

    def load_model(self, weight_file_path: str) -> None:
        """
        Loads the model weights
        """
        self.load_state_dict(torch.load(weight_file_path))
        self.trained = True

    def get_pseudo_inputs_recons(self) -> torch.Tensor:
        """
        Returns the trajectory generated from each pseudo_inputs.
        """
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
        """
        Generates a flux of trajectory from a single vamp prior.
        The pseudo input used is indicated but its index.
        """
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

    def get_pseudo_labels(self) -> torch.Tensor:
        """
        Returns the pseudo labels.
        """
        self.eval()
        with torch.no_grad():
            return self.pseudo_labels_layer.forward()
