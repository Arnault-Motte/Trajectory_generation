import os
from collections import Counter

import onnx
import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

import numpy as np
from data_orly.src.core.early_stop import Early_stopping
from data_orly.src.core.loss import *
from data_orly.src.core.networks import *


def compute_class_weights(data_loader: DataLoader, type: str = "raw") -> dict:
    """
    Computes the class weights for each label in the data
    """

    # Initialize a dictionary to store the counts of each label
    label_counts = {}

    # Iterate through the DataLoader
    for _, labels in tqdm(data_loader, desc="computing data weights"):
        for label in labels:
            label = tuple(
                label.tolist()
            )  # Convert label to a scalar (if it's a tensor)
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

    # Compute the total number of samples
    total_samples = sum(label_counts.values())

    # Compute the class weights
    if type == "raw":
        class_weights = {
            label: total_samples / count
            for label, count in label_counts.items()
        }
    else:
        class_weights = {
            label: np.log(1 + total_samples / count)
            for label, count in label_counts.items()
        }
    # print(class_weights)
    return class_weights


def divide_dataloader_per_label(
    dl: DataLoader, batch_size: int = 500
) -> dict[tuple, DataLoader]:
    # Dictionary to store data grouped by label
    data_by_label = {}

    # Iterate through the DataLoader
    for data, labels in dl:
        # print(labels)
        for x, label in zip(data, labels):
            label = tuple(
                label.tolist()
            )  # Convert label to a scalar (if it's a tensor)
            if label not in data_by_label:
                data_by_label[label] = []
            data_by_label[label].append(x)

    # Create a DataLoader for each label
    dataloaders = {}
    for label, data in data_by_label.items():
        # Convert the list of tensors to a TensorDataset
        dataset = TensorDataset(
            torch.stack(data),
            torch.tensor([list(label)] * len(data)).to(data[0].device),
        )
        dataloaders[label] = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    # first_batch = next(iter(dataloader))
    # inputs, labels = first_batch
    # print(inputs.shape)

    return dataloaders


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

        channels = inital_channels + 1

        self.tcn = TCN(
            channels,  # * 2,  # adding the label dim
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
        self.flatten = nn.Flatten()
        self.seq_len = seq_len

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels is None:
            x_label = x

        x_label = (
            torch.cat([x, labels], dim=1)  # on channels
        )

        x_label = self.tcn(x_label)
        x_label = self.avgpool(x_label)
        x_label = self.flatten(x_label)
        mu = self.mu_layer(x_label)
        log_var = self.logvar_layer(x_label)
        return mu, log_var
#ok

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
        self.dropout = nn.Dropout(dropout)

        self.sampling_factor = upsamplenum

        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x, labels], dim=1)  ##to delete
        x = self.first_dense(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))

        x = self.upsample(x)

        x = self.tcn(x)

        return x

#ok
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
        # pseudo_labels = self.dropout(pseudo_labels)
        return pseudo_labels
#ok

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
    
#ok


class Label_mapping_traj_part(nn.Module):
    """
    Model use to map the trajectory parts to the expected
    latent dim.
    """

    def __init__(
        self,
        input_channels: int,
        num_channels: list[int],
        label_dim: int = 64,
        kernel_size: int = 16,
        dropout: float = 0.2,
        return_layer: bool = False,  # to delete
    ) -> None:
        super(Label_mapping_traj_part, self).__init__()
        self.tcn = TCN_class(input_channels, num_channels, kernel_size, dropout)
        self.fcl = (
            nn.Linear(num_channels[-1], label_dim)
            if not return_layer  ## to delete test
            else None
        )

    def forward(
        self, traj: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        traj = traj.permute(0, 2, 1)
        if mask is not None:
            mask = mask.unsqueeze(1)
            # print(traj.shape)
            traj = torch.concat([traj, mask], dim=1)
        y = self.tcn(traj)
        # Use the last time step for information retrieval
        if self.fcl:  ## to delete cond test
            out = self.fcl(y[:, :, -1])
        else:
            out = y
        return out


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
#ok

class Weight_Prior_Conditioned_complexe(nn.Module):
    """
    Model used to learn the weights of the prior components
    weights based on the label. Used for CVAE with a conditioned VampPrior.
    """

    def __init__(
        self,
        num_pseudo_inputs: int,
        input_channels: int,
        num_channels: list[int],
        kernel_size: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super(Weight_Prior_Conditioned_complexe, self).__init__()
        self.in_chan = input_channels
        self.tcn = TCN_class(input_channels, num_channels, kernel_size, dropout)
        self.fcl = nn.Linear(num_channels[-1], num_pseudo_inputs)

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        # print(label.shape)
        if label.shape[1] != self.in_chan:
            label = label.permute(0, 2, 1)
        label = self.tcn(label)
        label = self.fcl(label[:, :, -1])
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
        conditioned_prior: bool = True,
        temp_save: str = "best_model.pth",
        num_worker: int = 4,
        init_std: float = 1,
        d_weight: bool = False,
        pseudo_labels: bool = True,
    ):
        super(CVAE_TCN_Vamp, self).__init__()
        self.label_dim = label_dim
        self.label_latent = label_latent
        self.pseudo_l = pseudo_labels

        self.latent_dim = latent_dim
        self.d_weight = d_weight
        self.num_worker = num_worker
        self.temp_save = temp_save
        self.pseudo_num = pseudo_input_num
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
            seq_len,  # + 1 if trajectory_label else seq_len,
            labels_latent_dim=label_latent,
        )

        self.pseudo_inputs_layer = Pseudo_inputs_generator(
            in_channels,
            seq_len,
            pseudo_input_num,
            dropout=dropout,
        )

        self.pseudo_labels_layer = Pseudo_labels_generator(
            label_dim,  # to add a channel for the mask
            pseudo_input_num,
            dropout,
        )

        self.seq_len = seq_len
        self.trained = False
        self.in_channels = in_channels
        self.log_std = nn.Parameter(
            torch.Tensor([init_std]), requires_grad=True
        )

        self.prior_weights = nn.Parameter(
            torch.ones((1, pseudo_input_num)), requires_grad=True
        )

        self.prior_weights_layers = (
            Weight_Prior_Conditioned(
                num_pseudo_inputs=pseudo_input_num,
                labels_dim=label_dim,
            )
            if conditioned_prior
            else None
        )

        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        #ok

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
            label = None
        else:
            label = self.labels_encoder_broadcast(label)
            label = label.view(-1, 1, self.seq_len)
        mu, log_var = self.encoder(x, label)
        return mu, log_var
    #ok

    def pseudo_inputs_latent(
        self, label: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and log variance associated with each
        pseudo input.
        """

        conditioned = self.is_conditioned()
        pseudo_labels = (
            self.pseudo_labels_layer.forward() if conditioned else None
        )

        pseudo_inputs = self.pseudo_inputs_layer.forward()

        # if not self.pseudo_l:
        #     batch_size = label.shape[0]
        #     pseudo_labels = label.repeat_interleave(
        #         self.pseudo_num, dim=0
        #     )  # to use the given labels
        #     pseudo_labels.reshape(-1,self.label_dim)
        #     pseudo_inputs = pseudo_inputs.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #     pseudo_inputs = pseudo_inputs.reshape(-1,self.in_channels,self.seq_len)
        #     print(pseudo_inputs.shape)
        #     print(pseudo_labels.shape)

        mini_batch_size = 600

        mu, log_var = self.encode(pseudo_inputs, pseudo_labels)
        # print("mu shape ", mu.shape)
        if label is not None:
            batch_size = label.shape[0]
            mu = mu.view(batch_size, self.pseudo_num, -1)
            log_var = log_var.view(batch_size, self.pseudo_num, -1)

            # print("mu shape ", mu.shape)

        return mu, log_var
    
    #ok

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
    
    #ok

    def get_prior_weight(self, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Returns the weights of each component of the prior.
        """
        if self.prior_weights_layers and labels is None:
            raise ValueError(
                "The prior need to be conditioned, please properly enter the labels when getting the prior weights"
            )

        if self.prior_weights_layers:
            # print(labels.shape)
            prior_weights = self.prior_weights_layers(labels.view(labels.size(0), -1))
            # print(prior_weights)
            # print(prior_weights.shape)
            return prior_weights
        else:
            return self.prior_weights
        
    #ok

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
        labels_encoder = labels  # risk
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
        loss_full: bool = False,  ## to delete true if you want the loss to reproduce the label as well
        label_msk: bool = False,
    ) -> None:
        """
        Fits the model to the data x in entry of the function. Early stopping is
        always activated.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataloader_train, data_loader_val = get_data_loader(
            x,
            labels,
            batch_size,
            num_worker=self.num_worker,
        )
        data_loaders_val = divide_dataloader_per_label(
            data_loader_val
        )  # for validation loss
        class_weights = (
            compute_class_weights(dataloader_train) if self.d_weight else None
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

        #ok cool

        for epoch in tqdm(range(epochs), desc="Epochs"):
            total_loss = 0.0
            kl_div = 0.0
            total_mse = 0.0
            for item in tqdm(dataloader_train, desc="Batch"):
                self.train()
                x_batch = item[0]
                labels_batch = item[1]

                x_batch = x_batch.to(next(self.parameters()).device)
                labels_batch = labels_batch.to(next(self.parameters()).device)
                x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self(
                    x_batch, labels_batch
                )

                prior_weight = self.get_prior_weight(labels_batch)
                #ok

                lbl_weights = (
                    torch.Tensor(
                        [
                            class_weights[tuple(label.tolist())]
                            for label in labels_batch
                        ]
                    ).to(next(self.parameters()).device)
                    if self.d_weight
                    else torch.ones(x_batch.size(0)).to(
                        next(self.parameters()).device
                    )
                )

                x_loss = x_batch.permute(0, 2, 1)  ##to delete --> maybe dangerous

                loss = CVAE_vamp_prior_loss_label_weights(
                    x_loss,  # x_batch.permute(0, 2, 1),
                    lbl_weights,
                    x_recon,
                    z,
                    mu,
                    log_var,
                    pseudo_mu,
                    pseudo_log_var,
                    scale=self.log_std,
                    vamp_weight=prior_weight,
                )
                #ok

                # gradiants
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # check losses
                kl = vamp_prior_kl_loss(
                    z,
                    mu,
                    log_var,
                    pseudo_mu,
                    pseudo_log_var,
                    vamp_weight=prior_weight,
                )
                mse = reconstruction_loss2(
                    x_loss,  # x_batch.permute(0, 2, 1),
                    x_recon,
                )
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

                print("Per data val loss")
                for lb, d_l in data_loaders_val.items():
                    val_loss, val_kl, val_recons = self.compute_val_loss(d_l)
                    val_loss = val_loss / len(d_l)
                    val_kl = val_kl / len(d_l)
                    val_recons = val_recons / len(d_l)
                    print(
                        f"Label {lb} : Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f}, Log_like: {val_loss - val_kl:.4f},scale: {self.log_std.item():.4f} "
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
            print(f"Best val loss = {early_stopping.min_loss}")
            print("|--Best model loaded--|")
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

    def compute_loss(self, data: torch.Tensor, labels: torch.Tensor):
        x_batch = data.to(next(self.parameters()).device)
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
        total_mse = mse.item() / cur_size
        kl_div = kl.mean().item()
        return loss.item(), kl_div, total_mse

    def compute_val_loss_per_label(
        self, data_loader_list: list[DataLoader]
    ) -> list[tuple[float, float, float]]:
        return [
            self.compute_val_loss(data_load) for data_load in data_loader_list
        ]

    def compute_val_loss(
        self,
        val_data: DataLoader,
    ) -> tuple[float, float, float]:
        """
        Computes the validation loss.
        Returns in order : the total loss, the kl div and the mse
        """
        total_loss = 0.0
        kl_div = 0.0
        total_mse = 0.0
        self.eval()
        for item in val_data:
            x_batch = item[0].to(next(self.parameters()).device)
            labels = item[1].to(next(self.parameters()).device)

            x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self(
                x_batch,
                labels,
            )

            prior_weights = self.get_prior_weight(labels)

            x_loss = x_batch.permute(0, 2, 1)  ##to delete

            loss = VAE_vamp_prior_loss(
                x_loss,  # x_batch.permute(0, 2, 1),
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
            mse = reconstruction_loss2(
                x_loss,  # x_batch.permute(0, 2, 1),
                x_recon,
            )
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
            0.9,
            shuffle=False,
            num_worker=self.num_worker
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
            # prior_weights = (
            #     self.prior_weights_layers(label)
            #     if self.prior_weights_layers
            #     else self.prior_weights
            # )
            print(label.shape)
            prior_weights = self.get_prior_weight(label)
            mu, log_var = self.pseudo_inputs_latent()
            # print("mu", mu.shape)
            # print("log_var ", log_var.shape)
        distrib = create_mixture(
            mu,
            log_var,
            vamp_weight=prior_weights.reshape(prior_weights.shape[-1]),
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
            current_batch_size = min(batch_size, num_samples - len(samples))
            z = self.sample_from_conditioned_prior(
                current_batch_size, labels[0].unsqueeze(0)
            ).to(device)
            print(labels[0])
            print(labels[0].shape)
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

    def generate_from_specific_vamp_prior_label(
        self, vamp_index: int, num_traj: int, label: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            psuedo_means, pseudo_scales = self.pseudo_inputs_latent()
            chosen_mean = psuedo_means[vamp_index]
            chosen_scale = (pseudo_scales[vamp_index] / 2).exp()
            dist = distrib.Independent(
                distrib.Normal(chosen_mean, chosen_scale), 1
            )
            sample = dist.sample(torch.Size([num_traj])).to(
                next(self.parameters()).device
            )
            print(sample.shape)
            labels = label.repeat(num_traj, 1)
            print(labels.shape)
            generated_traj = self.decode(sample, labels)

        return generated_traj.permute(0, 2, 1)

    def generate_from_specific_vamp_prior(
        self, vamp_index: int, num_traj: int, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates a flux of trajectory from a single vamp prior.
        The pseudo input used is indicated but its index.
        """
        self.eval()
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

    def save_model_ONNX(self, save_dir: str, opset_version: int = 17) -> None:
        """
        Exports each neural net of the CVAE into ONNX files.
        The path where these neural nets are saved is save_dir.
        The name of each submodule will define the name of each file.

        Args:
        save_dir (str): Path to the directory to save the ONNX files.
        opset_version (int): ONNX opset version.
        """
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.log_std,f"{save_dir}/log_std.pt")
        print("log std :", self.log_std)

        dummy_inputs = {
            "encoder": (
                torch.randn(1, self.in_channels, self.seq_len).to(
                    next(self.parameters()).device
                ),
                torch.randn(1, 1, self.seq_len).to(
                    next(self.parameters()).device
                ),
            ),
            "decoder": (
                torch.randn(1, self.latent_dim).to(
                    next(self.parameters()).device
                ),
                torch.randn(1, self.label_latent).to(
                    next(self.parameters()).device
                ),
            ),
            "pseudo_inputs_layer": (),
            "pseudo_labels_layer": (),
            "labels_encoder_broadcast": torch.randn(1, self.label_dim).to(
                next(self.parameters()).device
            ),
            "labels_decoder_broadcast": torch.randn(1, self.label_dim).to(
                next(self.parameters()).device
            ),
            "prior_weights_layers": torch.randn(1, self.label_dim).to(
                next(self.parameters()).device
            ),
        }

        for name, module in self.named_children():
            if isinstance(module, nn.Module):
                self.eval()
                file_path = os.path.join(save_dir, f"{name}.onnx")
                print(f"Exporting {name} to {file_path}")

                inputs = dummy_inputs[name]

                if not isinstance(inputs, tuple):
                    inputs = (inputs,)  # wrap in tuple if needed

                try:
                    torch.onnx.export(
                        module,
                        inputs,
                        file_path,
                        input_names=[f"input_{i}" for i in range(len(inputs))],
                        output_names=[f"output"],
                        dynamic_axes={
                            f"input_{i}": {0: "batch"}
                            for i in range(len(inputs))
                        },
                        opset_version=opset_version,
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to export {name}: {e}")
