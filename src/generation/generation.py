import datetime
from collections import Counter

import torch
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from numpy._typing._array_like import NDArray
from tqdm import tqdm
from traffic.algorithms.generation import compute_latlon_from_trackgs
from traffic.core import Flight, Traffic

from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_TCN_VampPrior import (
    get_data_loader as get_data_loader_labels,
)

from data_orly.src.generation.data_process import (
    Data_cleaner,
    compute_vertical_rate,
    jason_shanon,
)


class Generator:
    def __init__(self,model:CVAE_TCN_Vamp, data_clean:Data_cleaner, model_v:VAE_TCN_Vamp = None)-> None:
        self.model = model if model is not None else model_v
        self.data_clean = data_clean
        self.cond = model is not None
    
    def generate_n_flight(self,n_points:int,batch_size:int = 500) ->Traffic:
        model = self.model
        sampled = model.sample(num_samples=n_points,batch_size= batch_size)
        sampled = sampled.permute(0, 2, 1) 
        traf = self.data_clean.output_converter(sampled,landing=True)
        return traf



    def generate_n_flight_per_labels(self,labels: list[str],n_points:int,batch_size:int = 500)-> list[Traffic]:
        model = self.model
        traff_list = []
        for label in tqdm(labels,desc='Generation'):
             with tqdm(total=4, desc=f"Processing {label}") as pbar:
                labels_array = np.full(batch_size, label).reshape(-1, 1)
                transformed_label = self.data_clean.one_hot.transform(
                    labels_array
                )

                pbar.update(1)

                # print("|--Sampling--|")
                labels_final = torch.Tensor(transformed_label).to(
                    next(model.parameters()).device
                )
                sampled = model.sample(n_points, batch_size, labels_final)
                pbar.update(1)

                traf = self.data_clean.output_converter(sampled,landing=True)
                pbar.update(1)
                traff_list.append(traf)

        return traff_list
