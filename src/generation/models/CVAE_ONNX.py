import os
from collections import Counter

import onnx
import onnxruntime as ort
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


class CVAE_ONNX:
    """
    Class used to create
    """

    def __init__(self, onnx_dir: str, device: str = "cpu") -> None:
        self.device = device
        self.onnx_dir = onnx_dir

        # Load ONNX sessions
        self.encoder_sess = ort.InferenceSession(
            f"{onnx_dir}/encoder.onnx", providers=["CPUExecutionProvider"]
        )
        self.decoder_sess = ort.InferenceSession(
            f"{onnx_dir}/decoder.onnx", providers=["CPUExecutionProvider"]
        )
        self.pseudo_inputs_sess = ort.InferenceSession(
            f"{onnx_dir}/pseudo_inputs_layer.onnx",
            providers=["CPUExecutionProvider"],
        )
        self.pseudo_labels_sess = ort.InferenceSession(
            f"{onnx_dir}/pseudo_labels_layer.onnx",
            providers=["CPUExecutionProvider"],
        )
        self.label_enc_sess = ort.InferenceSession(
            f"{onnx_dir}/labels_encoder_broadcast.onnx",
            providers=["CPUExecutionProvider"],
        )
        self.label_dec_sess = ort.InferenceSession(
            f"{onnx_dir}/labels_decoder_broadcast.onnx",
            providers=["CPUExecutionProvider"],
        )

        encoder_input = self.encoder_sess.get_inputs()
        self.seq_len = encoder_input[0].shape[2]
        print(f"Seq_len : {self.seq_len}")
        self.log_std = torch.load(f"{onnx_dir}/log_std.pt")

        try:
            self.prior_weights_sess = ort.InferenceSession(
                f"{onnx_dir}/prior_weights_layers.onnx",
                providers=["CPUExecutionProvider"],
            )
            self.conditioned_prior = True
        except Exception:
            self.prior_weights_sess = None
            self.conditioned_prior = False

    def encode(
        self, x: torch.Tensor, label: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if label is not None:
            label_np = label.cpu().numpy().astype(np.float32)
            label_broadcast = self.label_enc_sess.run(
                None, {"input_0": label_np}
            )[0]
            label_broadcast = label_broadcast.reshape(-1, 1, x.shape[2])
        else:
            label_broadcast = np.zeros(
                (x.shape[0], 1, x.shape[2]), dtype=np.float32
            )

        x_np = x.cpu().numpy().astype(np.float32)
        mu, log_var = self.encoder_sess.run(
            None, {"input_0": x_np, "input_1": label_broadcast}
        )
        return torch.tensor(mu), torch.tensor(log_var)

    def decode(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label_np = label.cpu().numpy().astype(np.float32)
        label_latent = self.label_dec_sess.run(None, {"input_0": label_np})[0]
        z_np = z.cpu().numpy().astype(np.float32)
        x_recon = self.decoder_sess.run(
            None, {"input_0": z_np, "input_1": label_latent}
        )[0]
        return torch.tensor(x_recon)

    def pseudo_inputs_latent(
        self, label: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pseudo_inputs = self.pseudo_inputs_sess.run(None, {})[0]
        if self.conditioned_prior:
            pseudo_labels = self.pseudo_labels_sess.run(None, {})[0]
        else:
            pseudo_labels = None

        pseudo_inputs_tensor = torch.tensor(pseudo_inputs)
        pseudo_labels_tensor = (
            torch.tensor(pseudo_labels) if pseudo_labels is not None else None
        )
        return self.encode(
            pseudo_inputs_tensor, pseudo_labels_tensor
        )  # wierd no view()

    def get_prior_weight(self, label: torch.Tensor = None) -> torch.Tensor:
        if self.conditioned_prior:
            if label is None:
                raise ValueError("Label required for conditioned prior.")
            label_np = label.cpu().numpy().astype(np.float32)
            label_np = label_np.reshape(label_np.shape[0], -1)
            prior_weights = self.prior_weights_sess.run(
                None, {"input_0": label_np}
            )[0]
        else:
            raise NotImplementedError(
                "Unconditioned prior weight loading not implemented in ONNX."
            )
        return torch.tensor(prior_weights)

    def reparametrize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = x.permute(0, 2, 1)
        mu_pseudo, log_var_pseudo = self.pseudo_inputs_latent(label)
        mu, log_var = self.encode(x, label)
        z = self.reparametrize(mu, log_var)  # no permute for x
        x_recon = self.decode(z, label)
        return x_recon, z, mu, log_var, mu_pseudo, log_var_pseudo

    def reproduce(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x_recon, *_ = self.forward(x, label)
        return x_recon.permute(0, 2, 1)

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
        data1, _ = get_data_loader(
            data, labels, batch_size, 0.9, shuffle=False, num_worker=4
        )
        i = 0
        batches_reconstructed = []
        for batch, labels_batch in data1:
            if i == n_batch:
                break
            b = batch
            labels_batch = labels_batch
            x_recon = self.reproduce(b, labels_batch)
            batches_reconstructed.append(x_recon)
            i += 1
        reproduced_data = torch.cat(batches_reconstructed, dim=0)
        return reproduced_data, data1

    def sample_from_conditioned_prior(
        self, num_sample: int, label: torch.Tensor
    ) -> torch.Tensor:  # modify sampling and test
        """ """
        mu, log_var = self.pseudo_inputs_latent()
        prior_weights = self.get_prior_weight(label).cpu()
        # weights = prior_weights.squeeze().cpu().numpy()
        # weights /= weights.sum()

        mu = mu.cpu()
        log_var = log_var.cpu()

        distrib = create_mixture(
            mu, log_var, prior_weights.reshape(prior_weights.shape[-1])
        )
        samples = distrib.sample((num_sample,))
        return samples

    def sample(
        self, num_samples: int, batch_size: int, label: torch.Tensor
    ) -> torch.Tensor:
        samples = []
        for _ in range(0, num_samples, batch_size):
            cur_batch = min(batch_size, num_samples - len(samples))

            self.encoder_sess.get_inputs()[0].shape
            z = self.sample_from_conditioned_prior(
                cur_batch, label[0].unsqueeze(0)
            )

            x_gen = self.decode(z, label)
            samples.append(x_gen)
        return torch.cat(samples).permute(0, 2, 1)

    def get_pseudo_inputs_recons(self) -> torch.Tensor:
        pseudo_labels = self.pseudo_labels_sess.run(None, {})[0]
        pseudo_inputs = self.pseudo_inputs_sess.run(None, {})[0]
        pseudo_inputs_tensor: torch.Tensor = torch.tensor(
            pseudo_inputs
        ).permute(0, 2, 1)
        pseudo_labels_tensor = torch.tensor(pseudo_labels)
        return self.reproduce(pseudo_inputs_tensor, pseudo_labels_tensor)

    def generate_from_specific_vamp_prior(
        self, vamp_index: int, num_traj: int, label: torch.Tensor
    ) -> torch.Tensor:
        mu, log_var = self.pseudo_inputs_latent(label)
        mean = mu[vamp_index]
        scale = torch.exp(0.5 * log_var[vamp_index])
        dist = distrib.Independent(distrib.Normal(mean, scale), 1)
        samples = dist.sample(torch.Size([num_traj]))
        labels = label.expand(num_traj, -1)
        x_gen = self.decode(samples, labels)
        return x_gen.permute(0, 2, 1)

    def compute_loss(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> tuple[float, float, float]:
        data = data
        x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self.forward(
            data, labels
        )
        # x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = (
        #     x_recon.cpu(),
        #     z.cpu(),
        #     mu.cpu(),
        #     log_var.cpu(),
        #     pseudo_mu.cpu(),
        #     pseudo_log_var.cpu(),
        # )
        prior_weights = self.get_prior_weight(labels)
        loss = VAE_vamp_prior_loss(
            data.permute(0, 2, 1),
            x_recon,
            z,
            mu,
            log_var,
            pseudo_mu,
            pseudo_log_var,
            scale=self.log_std.cpu(),
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
        mse = reconstruction_loss(data.permute(0, 2, 1), x_recon)
        cur_size = data.size(0)
        total_mse = mse.item() / cur_size
        kl_div = kl.mean().item()
        return loss.item(), kl_div, total_mse
