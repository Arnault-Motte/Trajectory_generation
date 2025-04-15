import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np


# Creates a train and test dataloader from a numpy array.
def get_data_loader(
    data: np.ndarray,
    labels: np.ndarray = None,
    batch_size: int = 500,
    train_split: float = 0.8,
    shuffle: bool = True,
    num_worker: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns a train and a val dataloader for the data in entry.
    """
    lab = labels is not None

    data2 = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32) if lab else None
    dataset = (
        TensorDataset(data2, labels_tensor) if lab else TensorDataset(data2)
    )
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
    )
    test_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker
    )
    return train_loader, test_loader


# Basic TCN blocks
class TCNBlock(nn.Module):
    """
    Basic TCN block. Object represents a single conv layer,
    with a relu activation (active or not), and dropout.
    Deals with padding.
    """

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
    """
    A complete residual block formed of 2 TCN Blocks
    """

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
    """
    A full TCN architecture.
    Composed of successive TCN residual blocks.
    """

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


# VampPrior Pseudo inputs generator
class Pseudo_inputs_generator(nn.Module):
    """
    Model used to generate the pseudo inputs.
    Is formed of two fully connected layers.
    """

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
