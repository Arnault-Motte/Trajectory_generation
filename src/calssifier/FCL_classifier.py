from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np


class FCL_calssifier(nn.Module):
    """
    A simple FCL clasifier. Layer_dims controlls the dimension of each of the layers.
    """
    def __init__(self, layers_dim: list[int], dropout: float = 0.2, activation:nn.Module=  nn.ReLU) -> None:
        super(FCL_calssifier, self).__init__()

        layers =[]

        for i in range(len(layers_dim)-2):
            layers.append(nn.Linear(layers_dim[i],layers_dim[i + 1]))
            layers.append(nn.Dropout(dropout))
            layers.append(activation())
        layers.append(nn.Linear(layers_dim[-2],layers_dim[-1]))


        layers = [
            nn.Linear(layers_dim[i], layers_dim[i + 1])
            for i in range(len(layers_dim) - 1)
        ]
        self.layers = nn.Sequential(layers)

    def forward(self,x:torch.Tensor)-> torch.Tensor:
        x = self.layers(x)
        return x
