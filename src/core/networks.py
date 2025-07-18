from collections import Counter

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
    padding : float = None,
    testing: bool = False, # True if you want  to return a single dataset
) -> tuple[DataLoader, DataLoader]:
    """
    Returns a train and a val dataloader for the data in entry.
    """
    lab = labels is not None
    #print(labels.shape)

    if padding is not None:
        mask = (labels == padding)
        lines_w_padding = np.any(mask,axis=2)
        prefix_sizes = np.array(np.sum(lines_w_padding,axis=1))
        prefix_tensor = torch.tensor(prefix_sizes, dtype=torch.float32)
        

    data2 = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32) if lab else None

    #add the size of the prefix if needed:

    

    dataset = (
        TensorDataset(data2, labels_tensor) if lab else TensorDataset(data2)
    ) 
    if padding is not None : 
        dataset = TensorDataset(data2, labels_tensor,prefix_tensor)

    if not testing:
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
    else :
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, #keeping the ordern as much as possible
            num_workers=num_worker,
        )
        test_loader = None

    # if labels is not None:
    #     print(
    #         "Val repartition :",
    #         Counter([tuple(x[1].tolist()) for x in val_data]),
    #     )
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


class TCN_class(nn.Module):
    """
    TCN groupement of blocks used for classification
    """
    def __init__(
        self,
        input_channels:int,
        num_channels:list[int],
        kernel_size:int=16,
        dropout:float=0.2,
        stride:int=1,
    )-> None:
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = input_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2**i
            layers.append(
                TCN_residual_block(
                    in_ch, out_ch, kernel_size, stride, dilation, dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return self.network(x)


class Pseudo_inputs_generator_conditioned(nn.Module):
    # def __init__(
    #     self,
    #     number_of_channels: int,
    #     number_of_points: int,
    #     pseudo_inputs_num: int,
    #     dropout: float,
    #     label_dim: int,
    #     hidden_dim: int,
    # ) -> None:
    #     super(Pseudo_inputs_generator_conditioned, self).__init__()
    #     self.dropout = nn.Dropout(dropout)
    #     self.fc = nn.Sequential(
    #         nn.Linear(label_dim, hidden_dim),
    #         self.dropout,
    #         nn.ReLU(),
    #         nn.Linear(
    #             hidden_dim,
    #             pseudo_inputs_num * number_of_channels * number_of_points,
    #         ),
    #     )
    #     self.num_pseudo = pseudo_inputs_num
    #     self.channels = number_of_channels
    #     self.seq_len = number_of_points

    # def forward(self, label: torch.Tensor) -> torch.Tensor:  # y: (batch_size,)
    #     out = self.fc(label)  # shape: (batch_size, num_pseudo * input_dim)
    #     out = out.view(-1, self.num_pseudo, self.channels, self.seq_len)
    #     return out

    def __init__(
        self,
        number_of_channels: int,
        number_of_points: int,
        pseudo_inputs_num: int,
        dropout: float,
        label_dim: int,
        hidden_dim: int,
    ) -> None:
        super(Pseudo_inputs_generator_conditioned, self).__init__()
        self.number_of_channels = number_of_channels
        self.number_of_points = number_of_points
        self.pseudo_inputs_num = pseudo_inputs_num

        first_dim = pseudo_inputs_num + label_dim
        second_dim = number_of_channels * number_of_points


        self.first_layer = nn.Linear(first_dim, pseudo_inputs_num)
        self.relu = nn.ReLU()
        self.second_layer = nn.Linear(pseudo_inputs_num, second_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self,label:torch.Tensor) -> torch.Tensor:
        # print(label.shape)
        # print(label,len(label))
        x = torch.eye(self.pseudo_inputs_num, self.pseudo_inputs_num).to(
            next(self.parameters()).device
        )

        x = torch.cat([x for _ in range(len(label))],dim = 0)
        # print(x.shape)
   
        label = label.repeat_interleave(self.pseudo_inputs_num, dim=0)
        # print(label.shape)
        x = torch.cat((x,label),dim =1)

        # print("x", x.shape)

        pseudo_inputs = self.first_layer(x)
        pseudo_inputs = self.relu(pseudo_inputs)
        pseudo_inputs = self.dropout(pseudo_inputs)
        pseudo_inputs = self.second_layer(pseudo_inputs)
        #pseudo_inputs = self.dropout(pseudo_inputs)

        # print(pseudo_inputs.shape)
        # print(pseudo_inputs.view(
        #     -1, self.number_of_channels, self.number_of_points
        # ).shape)
        return pseudo_inputs.view(
            -1, self.number_of_channels, self.number_of_points
        )


def pad_tensor_with_last_row(tensor, n):
    current_rows = tensor.size(0)
    if current_rows >= n:
        return tensor  # Already has n or more rows
    last_row = tensor[-1].unsqueeze(0)  # shape: (1, features)
    num_missing = n - current_rows
    repeated_rows = last_row.repeat(num_missing, 1)  # repeat last row
    padded_tensor = torch.cat([tensor, repeated_rows], dim=0)
    return padded_tensor

# VampPrior Pseudo inputs generator
class Pseudo_inputs_generator(nn.Module):
    """
    Model used to generate the pseudo inputs.
    Is formed of two fully connected layers.
    Can be conditioned on labels by inputing label_dim and setting generate on label to true
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

        first_dim = pseudo_inputs_num
        second_dim = number_of_channels * number_of_points


        self.first_layer = nn.Linear(first_dim, pseudo_inputs_num)
        self.relu = nn.ReLU()
        self.second_layer = nn.Linear(pseudo_inputs_num, second_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self) -> torch.Tensor:
        # print(label.shape)

        x = torch.eye(self.pseudo_inputs_num, self.pseudo_inputs_num).to(
            next(self.parameters()).device
        )


        # print("x", x.shape)

        pseudo_inputs = self.first_layer(x)
        pseudo_inputs = self.relu(pseudo_inputs)
        pseudo_inputs = self.dropout(pseudo_inputs)
        pseudo_inputs = self.second_layer(pseudo_inputs)
        #pseudo_inputs = self.dropout(pseudo_inputs)

        pseudo_inputs = pseudo_inputs.view(
            -1, self.number_of_channels, self.number_of_points
        )

        # Forward: binary, backward: through sigmoid
        mask_logits = pseudo_inputs[:, -1, :]
        mask_prob = torch.sigmoid(mask_logits) #between 0 and 1
        mask_binary = (mask_prob > 0.5).float() # 1 or 0

        # Straight-through estimator for the back propagation
        mask_final = mask_binary + (mask_prob - mask_prob.detach())

        pseudo_inputs[:, -1, :] = mask_final

        return pseudo_inputs.view(
            -1, self.number_of_channels, self.number_of_points
        )


#for attention based TCN 


class SpatialAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)  # projects N → 1

    def forward(self, x):  # x: [B, N, T]
        B, N, T = x.shape
        # Transpose to [B, T, N] for per-time-step attention
        x_t = x.permute(0, 2, 1)  # [B, T, N]
        scores = self.linear(x_t)  # [B, T, 1]
        weights = torch.softmax(scores, dim=2)  # softmax over features
        attended = weights * x_t  # element-wise
        return attended.permute(0, 2, 1)  # back to [B, N, T]

class TemporalAttention(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.linear = nn.Linear(time_dim, 1)  # projects T → 1

    def forward(self, x):  # x: [B, N, T]
        B, N, T = x.shape
        # Apply over time for each feature
        x_n = x  # [B, N, T]
        scores = self.linear(x_n)  # [B, N, 1]
        weights = torch.softmax(scores, dim=2)  # softmax over time
        attended = weights * x_n
        return attended