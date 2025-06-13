from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np

from data_orly.src.core.loss import *
from data_orly.src.core.networks import *




class TCN_classifier(nn.Module):
    def __init__(
        self,
        layers_dims: list[int],
        kernel_size: int = 16,
        dropout: float = 0.2,
        activation: nn.Module = nn.ReLU,
        stride: float = 1,
    ) -> None:
        super(TCN_classifier, self).__init__()

        self.TCNs =TCN_class(layers_dims[0],layers_dims[1:-1],kernel_size,dropout,stride)
        self.fcl = nn.Linear(layers_dims[-2], layers_dims[-1])

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: (batch_size, input_channels, seq_len)
        y = self.tcn(x)
        # Use the last time step for classification
        out = self.fc(y[:, :, -1])
        return out
