import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import numpy as np


def get_data_loader(data: np.ndarray, batch_size: int) -> DataLoader:
    data2 = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data2)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %%
# Loss function
def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x, reduction="sum")


# BAsic TCN blocks
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
        self.dense_layer = nn.Linear(self.dense_dim, latent_dim)
        self.flaten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.avgpool(x)
        x = self.flaten(x)
        x = self.dense_layer(x)
        return x


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


class AE_TCN(nn.Module):
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
        super(AE_TCN, self).__init__()
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # 500 3 200
        x = self.encode(x)
        x_reconstructed = self.decode(x)
        # x_reconstructed = x_reconstructed.permute(0,2,1)
        return x_reconstructed

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
        dataloader = get_data_loader(x, batch_size)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch in dataloader:
                x_batch = x_batch[0].to(next(self.parameters()).device)
                x_recon = self(x_batch)
                loss = reconstruction_loss(x_batch.permute(0, 2, 1), x_recon)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")
            if epoch == epochs - 1:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

        self.trained = True

    def reproduce(self, x: torch.Tensor) -> torch.Tensor:
        if not self.trained:
            raise Exception("Model not trained yet")
        self.eval()
        # x =x.permute(0,2,1)
        return self.forward(x).permute(0, 2, 1)

    def reproduce_data(
        self, data: np.ndarray, batch_size: int, n_batch: int
    ) -> torch.Tensor:
        if not self.trained:
            raise Exception("Model not trained yet")
        data1 = get_data_loader(data, batch_size)
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
        return reproduced_data
