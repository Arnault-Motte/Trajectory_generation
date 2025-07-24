import os
from collections import Counter

import numpy as np
import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.core.early_stop import Early_stopping
from src.core.loss import *
from src.core.networks import *
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


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
        trajectory_label: bool = False,
        concat_on_channels: bool = False,  # True if the trajectory prefix must be concatenated on the channel dim
    ):
        super(TCN_encoder, self).__init__()

        self.concat_chann = concat_on_channels

        channels = inital_channels
        if not trajectory_label:
            channels = inital_channels + 1
        elif concat_on_channels:  # in case of a
            channels = inital_channels * 2
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
        self.trajectory_label = trajectory_label
        self.seq_len = seq_len

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels is None:
            x_label = x
        elif mask is None:
            x_label = (
                torch.cat([x, labels], dim=1)  # on channels
                if not self.trajectory_label or self.concat_chann
                else torch.cat([labels, x], dim=2)  # on the number of points
            )
        else:
            mask_expanded = mask.unsqueeze(1).expand_as(
                x
            )  # use mask to combine the label and the prior
            x_label = x.clone()
            x_label[mask_expanded] = labels[mask_expanded]
            x_label = torch.cat(
                [x, mask.unsqueeze(1)], dim=1
            )  # help the model distinguish label from input

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
        trajectory_label: bool = False,  # to delete
        test_concat: bool = False,  ##to delete
        test_msk: bool = False,  ##to delete
        final_fcl: int = 0,
    ):
        super(TCN_decoder, self).__init__()
        self.test_concat = test_concat  ##to delete
        self.seq_len = seq_len
        self.first_dense = nn.Linear(
            latent_dim + labels_latent_dim
            if not test_concat
            else latent_dim,  ##to delete cond
            in_channels * (seq_len // upsamplenum),
        )
        self.upsample = nn.Upsample(scale_factor=upsamplenum)
        if test_msk:  # to delete
            in_channels = in_channels + 1
        elif test_concat:
            in_channels = in_channels * 2
        self.tcn = TCN(
            in_channels,
            # if not test_concat or test_msk
            # else in_channels * 2,  ## to delete cond
            latent_dim,
            out_channels,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
        )
        self.dropout = nn.Dropout(dropout)
        if final_fcl == 0:
            self.last_fcn = None
            print(final_fcl)
        elif final_fcl == 1:
            self.last_fcn = nn.Linear(
                seq_len * out_channels, seq_len * out_channels
            )
        elif final_fcl == 2:
            # self.last_fcn = (
            #     nn.Linear(seq_len * out_channels, seq_len * out_channels)
            #     if False
            #     else nn.Linear(2 * seq_len * out_channels, seq_len * out_channels)
            # )
            self.last_fcn = nn.Sequential(
                nn.Linear(seq_len * out_channels, seq_len * out_channels),
                nn.ReLU(),
                # self.dropout,
                nn.Linear(seq_len * out_channels, seq_len * out_channels),
            )

        self.sampling_factor = upsamplenum

        self.trajectory_label = trajectory_label
        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
        og_label: torch.Tensor = None,
    ) -> torch.Tensor:
        x = (
            torch.cat([x, labels], dim=1) if not self.test_concat else x
        )  ##to delete
        x = self.first_dense(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        # print("recons")
        # print(x.shape)
        x = self.upsample(x)

        if self.test_concat and mask is None:  ## to delete
            x = torch.cat([x, labels], dim=1)  ##to delete add on channel dim
        elif self.test_concat:
            mask_expanded = mask.unsqueeze(1).expand_as(
                x
            )  # use mask to combine the label and the prior
            x_label = x  # .clone()
            x_label[mask_expanded] = labels[mask_expanded]
            x_label = torch.cat(
                [x, mask.unsqueeze(1)], dim=1
            )  # help the model distinguish label from input
        # print(x.shape)
        x = self.tcn(x) if mask is None else self.tcn(x_label)

        if self.last_fcn is not None:
            # if mask is None
            # print(x.shape)
            # print(self.seq_len,' ',self.out_channels)
            x = x.reshape(x.shape[0], -1)
            # og_label = og_label.reshape(og_label.shape[0], -1)
            # x = torch.cat([x, og_label], dim=1)
            # x = self.dropout(x)
            x = self.last_fcn(x)
            x = x.reshape(x.shape[0], self.out_channels, self.seq_len)
            # if mask is not None:
            #     x_label[mask_expanded] = labels[mask_expanded]
            #     x_label = torch.cat(
            #         [x, mask.unsqueeze(1)], dim=1
            #     )

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


# class Label_mapping_traj_part(nn.Module):
#     """
#     Model use to map the trajectory parts to the expected
#     latent dim.
#     """

#     def __init__(
#         self,
#         label_dim: int,
#         latent_dim: list[int],
#         dropout: float = 0.2,
#     ) -> None:
#         super(Label_mapping_traj_part, self).__init__()
#         self.first_layer = nn.Linear(label_dim, latent_dim[0])
#         self.dropout = nn.Dropout(dropout)
#         layers = [self.first_layer, self.dropout, nn.ReLU()]
#         for i in range(len(latent_dim) - 1):
#             layer = nn.Linear(latent_dim[i], latent_dim[i + 1])
#             layers.append(layer)
#             if i != len(latent_dim) - 2:
#                 layers.append(self.dropout)
#                 layers.append(nn.ReLU())
#         self.layers = nn.Sequential(*layers)

#     def forward(self, traj: torch.Tensor) -> torch.Tensor:
#         traj = traj.reshape(traj.size(0), -1)
#         return self.layers(traj)


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
class CVAE_TCN_Vamp_old(nn.Module):
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
        d_weight: bool = False,
        condition_pseudo_inputs: bool = False,  # true if the pseudo inputs are to be conditioned on the label
        trajectory_label: bool = False,
        complexe_weight: bool = False,
        padding: int = None,
        test_concat: bool = False,
        test_concat_mask: bool = False,
        final_fcl: int = 0,
    ):
        super(CVAE_TCN_Vamp_old, self).__init__()

        # -----------------------test to delete
        self.test_concat = test_concat
        self.test_concat_mask = test_concat_mask
        # -------------------------------------

        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.label_latent = label_latent

        self.padding = padding
        self.complexe_w = complexe_weight
        self.d_weight = d_weight
        self.num_worker = num_worker
        self.temp_save = temp_save
        self.pseudo_num = pseudo_input_num
        self.labels_encoder_broadcast = (
            Label_mapping(
                num_classes is None,
                label_dim,
                seq_len,
                num_classes,
            )
            if not trajectory_label
            else nn.Identity()
        )
        # self.labels_encoder_broadcast = self.labels_encoder_broadcast.to("cuda")
        # print("identity VRAM after encoder initialization:")
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        self.trajectory_label = trajectory_label

        self.labels_decoder_broadcast = (
            Label_mapping(
                num_classes is None, label_dim, label_latent, num_classes
            )
            if not trajectory_label
            else Label_mapping_traj_part(
                input_channels=in_channels
                if padding is None
                else in_channels + 1,
                num_channels=[16, 32, 64, latent_dim],  # [16, 32, 64, 64],
                label_dim=label_latent,
                kernel_size=kernel_size,
                dropout=dropout,
                return_layer=test_concat,  # to delete
            )
            # else Label_mapping_traj_part(
            #     label_dim, latent_dim=[128, 64, label_latent]
            # )
        )
        # self.labels_decoder_broadcast = self.labels_decoder_broadcast.to("cuda")

        # print("mapping VRAM after encoder initialization:")
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        if (
            trajectory_label and padding is None
        ):  # in case whe only use the first halves of trajectory with no padding
            seq_len = seq_len * 2

        self.encoder = TCN_encoder(
            in_channels if padding is None else in_channels + 1,
            out_channels,
            latent_dim,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
            avgpoolnum,
            seq_len,
            trajectory_label=trajectory_label,
            # concat_on_channels=padding is not None,
        )
        # self.encoder = self.encoder.to("cuda")
        # print("enc VRAM after encoder initialization:")
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        self.decoder = TCN_decoder(
            latent_dim,
            in_channels,  # if padding is None else in_channels + 1,
            latent_dim,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
            upsamplenum,
            seq_len,  # + 1 if trajectory_label else seq_len,
            labels_latent_dim=label_latent,
            trajectory_label=trajectory_label,
            test_concat=test_concat,
            test_msk=test_concat_mask,
            final_fcl=final_fcl,
        )
        # self.decoder = self.decoder.to("cuda")
        # print("dec VRAM after encoder initialization:")
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        self.conditioned_pseudo_in = condition_pseudo_inputs

        self.pseudo_inputs_layer = (
            Pseudo_inputs_generator(
                in_channels if padding is None else in_channels + 1,
                seq_len,
                pseudo_input_num,
                dropout=dropout,
            )
            if not condition_pseudo_inputs
            else Pseudo_inputs_generator_conditioned(
                in_channels,
                seq_len,
                pseudo_input_num,
                dropout=dropout,
                label_dim=label_dim,
                hidden_dim=0,
            )
        )

        # self.pseudo_inputs_layer = self.pseudo_inputs_layer.to("cuda")
        # print("pseudo in VRAM after encoder initialization:")
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        self.pseudo_labels_layer = Pseudo_labels_generator(
            label_dim,  # to add a channel for the mask
            pseudo_input_num,
            dropout,
        )

        # self.pseudo_labels_layer = self.pseudo_labels_layer.to("cuda")
        # print("pseudo label VRAM after encoder initialization:")
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

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

        if complexe_weight:
            self.prior_weights_layers = Weight_Prior_Conditioned_complexe(
                num_pseudo_inputs=pseudo_input_num,
                input_channels=in_channels
                if padding is None
                else in_channels + 1,
                num_channels=[16, 32, 64, 128],
                kernel_size=kernel_size,
                dropout=dropout,
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
        self, x: torch.Tensor, label: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the data x in entry to train the model"
        """
        if label is None:
            # label = torch.zeros(x.size(0), 1, self.seq_len).to(
            #     next(self.parameters()).device
            # )
            label = None
        else:
            label = self.labels_encoder_broadcast(label)
            label = (
                label.view(-1, 1, self.seq_len)
                if not self.trajectory_label
                else label.view(-1, self.in_channels, self.seq_len)
            )
            # print(label.shape)
        
        x = x.to(next(self.encoder.parameters()).device)
        label = label.to(
            next(self.encoder.parameters()).device
        )
        if mask is not None:
            mask = mask.to(next(self.encoder.parameters()).device)

        mu, log_var = self.encoder(x, label, mask)

        mu = mu.to(next(self.decoder.parameters()).device)
        log_var = log_var.to(next(self.decoder.parameters()).device)
        x = x.to(next(self.decoder.parameters()).device)
        label = label.to(
            next(self.decoder.parameters()).device
        )
        if mask is not None:
            mask = mask.to(next(self.decoder.parameters()).device)
        return mu, log_var

    def pseudo_inputs_latent(
        self, label: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and log variance associated with each
        pseudo input.
        """
        if self.conditioned_pseudo_in and label is None:
            raise ValueError(
                "The pseudo inputs must be conditioned on the label."
            )

        conditioned = self.is_conditioned()
        pseudo_labels = (
            (self.pseudo_labels_layer.forward() if conditioned else None)
            if not self.conditioned_pseudo_in
            else label.repeat_interleave(
                self.pseudo_num, dim=0
            )  # to use the given labels
        )

        if not self.conditioned_pseudo_in:
            pseudo_inputs = self.pseudo_inputs_layer.forward()

        # # print("dim ",pseudo_inputs.shape)
        # if self.conditioned_pseudo_in:  # to be able to use every pseudo inputs, risks of having to big of a batch
        #     pseudo_inputs = pseudo_inputs.view(
        #         -1,
        #         self.in_channels,
        #         self.seq_len,
        #     )

        # print("dim2",pseudo_inputs.shape)
        # print("dim_lbl",pseudo_labels.shape)

        mini_batch_size = 600

        if self.conditioned_pseudo_in:

            # unique_labels, inverse_indices = torch.unique(label, dim=0, return_inverse=True)
            # pseudo_inputs = self.pseudo_inputs_layer.forward(unique_labels)  # Shape: [num_unique, pseudo_num, ...]
            # pseudo_inputs = pseudo_inputs.reshape(-1, self.in_channels, self.seq_len)

            # expanded_labels = unique_labels.repeat_interleave(self.pseudo_num, dim=0)
            # mu_all, log_var_all = self.encode(pseudo_inputs, expanded_labels)
            # mu = mu_all.view(len(unique_labels), self.pseudo_num, -1)[inverse_indices]
            # log_var = log_var_all.view(len(unique_labels), self.pseudo_num, -1)[inverse_indices]
            # return mu, log_var

            ##ol ver
            unique = torch.unique(label, dim=0)

            pseudo_inputs_list = []
            mus = []
            log_vars = []
            for l in unique:



                
                pseudo_input = self.pseudo_inputs_layer.forward(l.unsqueeze(0))
                pseudo_inputs_list.append(pseudo_input)
                pseudo_input = pseudo_input.reshape(
                    -1, self.in_channels, self.seq_len
                )
                temp = l.unsqueeze(0).repeat_interleave(self.pseudo_num, dim=0)
                mu, log_var = self.encode(pseudo_input, temp)
                mus.append(mu)
                log_vars.append(log_var)

            mu = torch.cat(mus, dim=0)
            log_var = torch.cat(log_vars, dim=0)

            pseudo_inputs: torch.Tensor = torch.cat(pseudo_inputs_list, dim=0)

            # # pseudo_inputs = self.pseudo_inputs_layer.forward(unique)
            # pseudo_inputs = pseudo_inputs.reshape(
            #     -1, self.in_channels, self.seq_len
            # )

            # # flattening the pseudo inputs

            # # print(pseudo_inputs.shape)
            # # temp_label = pseudo_inputs.unsqueeze(0).repeat(self.pseudo_num, 1)
            # # temp = torch.repeat_interleave(unique,self.pseu , dim=1)
            # temp = unique.repeat_interleave(self.pseudo_num, dim=0)
            # # print(temp.shape)
            # mu, log_var = self.encode(pseudo_inputs, temp)

            label_to_unique = {
                tuple(unique[i].tolist()): (
                    mu[self.pseudo_num * i : self.pseudo_num * (i + 1)],
                    log_var[self.pseudo_num * i : self.pseudo_num * (i + 1)],
                )
                for i in range(len(unique))
            }

            vectors = [
                label_to_unique[tuple(label[i].tolist())]
                for i in range(len(label))
            ]
            mu = torch.concat([val for val, _ in vectors], dim=0)

            log_var = torch.concat([val for _, val in vectors], dim=0)

            # all_mu = []
            # all_log_var = []

            # num_batches = pseudo_inputs.shape[0] // mini_batch_size

            # for i in range(num_batches):
            #     #print(i)
            #     mini_batch = pseudo_inputs[i * mini_batch_size : (i + 1) * mini_batch_size]
            #     labels_batch = pseudo_labels[i * mini_batch_size : (i + 1) * mini_batch_size]

            #     # Passing the mini-batch through the encoder
            #     mu, log_var = self.encode(mini_batch, labels_batch)

            #     all_mu.append(mu)
            #     all_log_var.append(log_var)

            # if pseudo_inputs.shape[0] % mini_batch_size != 0:
            #     mini_batch = pseudo_inputs[num_batches * mini_batch_size :]
            #     labels_batch = pseudo_labels[num_batches * mini_batch_size :]

            #     # Passing the remaining mini-batch through the encoder
            #     mu, log_var = self.encode(mini_batch, labels_batch)

            #     all_mu.append(mu)
            #     all_log_var.append(log_var)

            # mu = torch.cat(all_mu, dim=0)
            # log_var = torch.cat(all_log_var, dim=0)
            # print(mu.shape)
            # print(log_var.shape)

        else:
            # if self.trajectory_label:

            #     pseudo_labels = pseudo_labels.view(-1,self.in_channels,self.seq_len)
            #     print(pseudo_labels.shape)
            #     print(pseudo_inputs.shape)

            if self.padding is not None:
                pseudo_labels = None

            mu, log_var = self.encode(pseudo_inputs, pseudo_labels)

        if label is not None:
            batch_size = label.shape[0]
            mu = mu.view(batch_size, self.pseudo_num, -1)
            log_var = log_var.view(batch_size, self.pseudo_num, -1)

            # print("mu shape ", mu.shape)

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

    def decode(
        self, z: torch.Tensor, label: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generates the output from the sampled vector and the chosen label.
        """
        old_label = label
        label = (
            self.labels_decoder_broadcast(label, mask)
            if self.trajectory_label
            else self.labels_decoder_broadcast(label)
        )
        # print(label.shape,z.shape)
        x = (
            self.decoder(z, label)
            if not self.test_concat_mask  ## to delete
            else self.decoder(
                z, label, ~mask, old_label if True else None
            )  # for old label test
        )
        return x

    def get_prior_weight(
        self, labels: torch.Tensor = None, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Returns the weights of each component of the prior.
        """
        if self.prior_weights_layers and labels is None:
            raise ValueError(
                "The prior need to be conditioned, please properly enter the labels when getting the prior weights"
            )

        if self.prior_weights_layers:
            if self.trajectory_label:
                labels = labels.permute(0, 2, 1)
                if mask is not None:
                    mask = mask.unsqueeze(1)
                    labels = torch.concat([labels, mask], dim=1)
                    return self.prior_weights_layers(labels)
            return self.prior_weights_layers(labels.view(labels.size(0), -1))
        else:
            return self.prior_weights

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
        label_msk: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        conditioned = self.is_conditioned()
        labels_encoder = (
            labels if not self.trajectory_label else labels.permute(0, 2, 1)
        )  # risk
        if not conditioned:
            labels_encoder = None

        labels_pseudo_inputs = labels if self.conditioned_pseudo_in else None

        mu_pseudo, log_var_pseudo = self.pseudo_inputs_latent(
            labels_pseudo_inputs
        )
        x = x.permute(0, 2, 1)  # 500 3 200
        # if mask is not None:
        #     mask = mask.permute(1, 0)
        #     print("m2 ",mask.shape)
        # print(labels_encoder.shape)
        # print(x.shape)
        mu, log_var = self.encode(x, labels_encoder, mask)

        z = self.reparametrize(mu, log_var).to(next(self.parameters()).device)
        l_msk = label_msk if label_msk is not None else mask
        if l_msk is not None:
            l_msk = ~l_msk
        x_reconstructed = self.decode(z, labels, l_msk)
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
        offload_enc: str = "",
    ) -> None:
        """
        Fits the model to the data x in entry of the function. Early stopping is
        always activated.
        """
        self.train()
        if offload_enc == "":
            offload_enc = next(self.parameters()).device

        self.encoder = self.encoder.to(offload_enc)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataloader_train, data_loader_val = get_data_loader(
            x,
            labels,
            batch_size,
            num_worker=self.num_worker,
            padding=self.padding,
        )
        data_loaders_val = (
            divide_dataloader_per_label(data_loader_val)
            if not self.trajectory_label
            else None
        )
        class_weights = (
            compute_class_weights(dataloader_train) if self.d_weight else None
        )
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        early_stopping = (
            Early_stopping(
                patience=self.patience,
                min_delta=self.min_delta,
                best_model_path="src/generation/models/saved_temp/"
                + self.temp_save,
            )
            if self.early_stopping
            else None
        )

        for epoch in tqdm(range(epochs), desc="Epochs"):
            total_loss = 0.0
            kl_div = 0.0
            total_mse = 0.0
            for item in tqdm(dataloader_train, desc="Batch"):
                self.train()
                x_batch = item[0]
                labels_batch = item[1]
                mask = None
                label_mask = None
                if self.padding is not None:
                    mask = (
                        (x_batch == self.padding)
                        .all(dim=2)
                        .to(next(self.parameters()).device)
                    )

                    if label_msk:
                        label_mask = (
                            (labels_batch != self.padding)
                            .all(dim=2)
                            .to(next(self.parameters()).device)
                        )

                    # print(mask.shape)
                    # size_prefix = item[2]
                    # test_p = size_prefix[0].item()
                    # # print(test_p)
                    # # print(x_batch[0,:,0].cpu().tolist().count(-10))

                    # idx = torch.arange(self.seq_len).unsqueeze(0)
                    # size_expand = size_prefix.unsqueeze(1)
                    # mask = (idx >= size_expand -1).to(
                    #     next(self.parameters()).device
                    # )  # used to zero out element in the reconstruction loss
                    # print(mask[0])
                    # input("pause")

                x_batch = x_batch.to(next(self.parameters()).device)
                labels_batch = labels_batch.to(next(self.parameters()).device)
                # print(labels_batch)
                x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self(
                    x_batch, labels_batch, mask, label_mask
                )

                # print(x_recon.shape)

                # last_row = labels_batch[
                #     :, -1:, :
                # ]  # to get the last point of the label
                # x_batch = torch.cat(
                #     [last_row,x_batch], dim=1
                # )  # (batch_size,101,4)

                # print("xb ",x_batch.shape)

                # prior_weight = (
                #     self.prior_weights_layers(
                #         labels_batch.view(labels_batch.size(0), -1)
                #         if not self.complexe_w
                #         else labels_batch,

                #     )
                #     if self.prior_weights_layers
                #     else self.prior_weights
                # )
                if label_mask is None:
                    label_mask = mask  ##to delete
                prior_weight = self.get_prior_weight(labels_batch, label_mask)

                # print("prior shape ", labels_batch.shape)
                # print("prior shape ", prior_weight.shape)
                # print("mu shape ", pseudo_mu.shape)
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
                # print(Counter(lbl_weights.tolist()).most_common())
                # print(x_batch.shape)
                # print(x_recon.shape)

                x_loss = x_batch.permute(0, 2, 1)  ##to delete
                if (
                    loss_full
                ):  # combining label and predcit to have a single output
                    x_loss = x_loss.clone()
                    mask_expanded = mask.unsqueeze(1).expand_as(
                        x_loss
                    )  # use mask to combine the label and the prior
                    x_loss[mask_expanded] = labels_batch.permute(0, 2, 1)[
                        mask_expanded
                    ]
                # print("_______________________________")
                # print(x_loss[0][0])
                # print(x_recon[0][0])

                loss_mask = (
                    ~mask if (not loss_full) and (mask is not None) else None
                )  # to delete
                # print(loss_mask)
                # print("_______________________________")
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
                    mask=loss_mask,  # 1 where we should predict
                )
                # loss = VAE_vamp_prior_loss(
                #     x_batch.permute(0, 2, 1),
                #     x_recon,
                #     z,
                #     mu,
                #     log_var,
                #     pseudo_mu,
                #     pseudo_log_var,
                #     scale=self.log_std,
                #     vamp_weight=prior_weight,
                # )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # prior_weights = (
                #     self.prior_weights_layers(labels_batch)
                #     if self.prior_weights_layers
                #     else self.prior_weights
                # )
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
                    mask=loss_mask,
                )
                cur_size = x_batch.size(0)
                total_mse += mse.item() / cur_size
                kl_div += kl.mean().item()
            scheduler.step()
            # validation
            val_loss, val_kl, val_recons = self.compute_val_loss(
                data_loader_val, loss_full, label_msk
            )

            total_loss = total_loss / len(dataloader_train)
            total_mse = total_mse / len(dataloader_train)
            kl_div = kl_div / len(dataloader_train)
            val_loss = val_loss / len(data_loader_val)
            val_kl = val_kl / len(data_loader_val)
            val_recons = val_recons / len(data_loader_val)
            if epoch % 1 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}]\n Train set, Loss: {total_loss:.4f},MSE: {total_mse:.4f}, KL: {kl_div:.4f}, Log_like: {total_loss - kl_div:.4f},scale: {self.log_std.item():.4f} "
                )
                print(
                    f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f}, Log_like: {val_loss - val_kl:.4f},scale: {self.log_std.item():.4f} "
                )

                if not self.trajectory_label:
                    print("Per data val loss")
                    for lb, d_l in data_loaders_val.items():
                        val_loss, val_kl, val_recons = self.compute_val_loss(
                            d_l
                        )
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
        loss_full: bool = False,
        label_msk: bool = False,
    ) -> tuple[float, float, float]:
        """
        Computes the validation loss.
        Returns in order : the total loss, the kl div and the mse
        """
        total_loss = 0.0
        kl_div = 0.0
        total_mse = 0.0
        self.eval()
        with torch.no_grad():
            for item in val_data:
                x_batch = item[0].to(next(self.parameters()).device)
                labels = item[1].to(next(self.parameters()).device)
                mask = None
                label_mask = None  # to delete
                if self.padding is not None:
                    mask = (
                        (x_batch == self.padding)
                        .all(dim=2)
                        .to(next(self.parameters()).device)
                    )

                    if label_msk:  # to delete
                        label_mask = (
                            (labels != self.padding)
                            .all(dim=2)
                            .to(next(self.parameters()).device)
                        )

                    # print(mask.shape)
                # mask = None
                # if self.padding is not None:
                #     size_prefix = item[2]
                #     idx = torch.arange(self.seq_len).unsqueeze(0)
                #     size_expand = size_prefix.unsqueeze(1)
                #     mask = (idx >= size_expand).to(
                #         next(self.parameters()).device
                #     )  # used to zero out element in the reconstruction loss

                # x_batch = x_batch.to(next(self.parameters()).device)
                # labels = labels.to(next(self.parameters()).device)

                x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self(
                    x_batch, labels, mask, label_mask
                )
                # prior_weights = (
                #     self.prior_weights_layers(
                #         labels.view(labels.size(0), -1)
                #         if not self.complexe_w
                #         else labels
                #     )
                #     if self.prior_weights_layers
                #     else self.prior_weights
                # )
                if label_mask is None:
                    label_mask = mask  ##to delete
                prior_weights = self.get_prior_weight(labels, label_mask)

                x_loss = x_batch.permute(0, 2, 1)  ##to delete
                if loss_full:
                    x_loss = x_loss.clone()
                    mask_expanded = mask.unsqueeze(1).expand_as(
                        x_loss
                    )  # use mask to combine the label and the prior
                    x_loss[mask_expanded] = labels.permute(0, 2, 1)[
                        mask_expanded
                    ]

                loss_mask = (
                    ~mask if (not loss_full) and (mask is not None) else None
                )  # to delete
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
                    mask=loss_mask,
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
                    loss_mask,
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
            0.8,
            shuffle=False,
            num_worker=self.num_worker,
            padding=self.padding,
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
            print("prior_weight", prior_weights.shape)

            print(self.prior_weights.tolist())

            mu, log_var = (
                self.pseudo_inputs_latent()
                if not self.conditioned_pseudo_in
                else self.pseudo_inputs_latent(label)
            )
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
            # print("sizes fun : ", labels.shape, batch_size)
            current_batch_size = min(batch_size, num_samples - len(samples))
            z = (
                self.sample_from_conditioned_prior(
                    current_batch_size, labels[0]
                ).to(device)
                if not self.trajectory_label
                else self.sample_from_conditioned_prior(
                    current_batch_size,
                    labels[0].flatten() if not self.complexe_w else labels,
                ).to(device)
            )
            print(labels.shape)
            # if z.shape[1] != self.in_channels:
            #     z = z.permute(0,2,1)
            if self.complexe_w:
                labels = labels.permute(0, 2, 1)
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

        torch.save(self.log_std, f"{save_dir}/log_std.pt")

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
            "pseudo_inputs_layer": ()
            if not self.conditioned_pseudo_in
            else torch.randn(1, self.label_dim).to(
                next(self.parameters()).device
            ),
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
