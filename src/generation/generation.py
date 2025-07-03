import datetime
from collections import Counter

import torch
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
from data_orly.src.generation.data_process import (
    Data_cleaner,
    compute_vertical_rate,
    jason_shanon,
)
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_ONNX import CVAE_ONNX
from data_orly.src.generation.models.CVAE_TCN_VampPrior import (
    get_data_loader as get_data_loader_labels,
)
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from numpy._typing._array_like import NDArray
from traffic.algorithms.generation import compute_latlon_from_trackgs
from traffic.core import Flight, Traffic


class Generator:
    def __init__(
        self, model: CVAE_TCN_Vamp | VAE_TCN_Vamp, data_clean: Data_cleaner
    ) -> None:
        self.model = model
        self.data_clean = data_clean
        self.cond = model is not None

    def generate_n_flight(
        self, n_points: int, batch_size: int = 500, lat_long: bool = True
    ) -> Traffic:
        model = self.model
        sampled = model.sample(num_samples=n_points, batch_size=batch_size)
        # sampled = sampled.permute(0, 2, 1)
        traf = self.data_clean.output_converter(
            sampled, landing=True, lat_lon=lat_long
        )
        return traf

    def generate_flight_for_label_vamp(
        self,
        label: str,
        vamp: int,
        n_points: int,
        batch_size: int = 500,
        lat_lon: bool = True,
    ) -> Traffic:
        model = self.model
        model.eval()

        with torch.no_grad():
            with tqdm(total=3, desc=f"Processing {label}") as pbar:
                print(label)
                labels_array = np.array([label]).reshape(-1, 1)
                transformed_label = self.data_clean.one_hot.transform(
                    labels_array
                )

                pbar.update(1)

                # print("|--Sampling--|")
                labels_final = torch.Tensor(transformed_label).to(
                    next(model.parameters()).device
                )
                sampled = model.generate_from_specific_vamp_prior_label(
                    vamp, n_points, labels_final
                )
                pbar.update(1)

                traf = self.data_clean.output_converter(
                    sampled, landing=True, lat_lon=lat_lon
                )
                pbar.update(1)

        return traf

    def generate_n_flight_per_labels(
        self,
        labels: list[str],
        n_points: int,
        batch_size: int = 500,
        lat_long: bool = True,
    ) -> list[Traffic]:
        model: CVAE_TCN_Vamp = self.model
        model.eval()
        traff_list = []
        with torch.no_grad():
            for label in tqdm(labels, desc="Generation"):
                with tqdm(total=3, desc=f"Processing {label}") as pbar:
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

                    traf = self.data_clean.output_converter(
                        sampled, landing=True, lat_lon=lat_long
                    )
                    pbar.update(1)
                    traff_list.append(traf)

        return traff_list


class ONNX_Generator:
    def __init__(self, model: CVAE_ONNX, data_clean: Data_cleaner) -> None:
        self.model = model
        self.data_clean = data_clean
        self.cond = model is not None

    def generate_flight_for_label_vamp(
        self,
        label: str,
        vamp: int,
        n_points: int,
        batch_size: int = 500,
        lat_lon: bool = True,
    ) -> Traffic:
        """
        Generates n_points flights, using the entire model limiting us to one vamp prior component speifed by vamp.
        """

        model = self.model


        with tqdm(total=3, desc=f"Processing {label}") as pbar:
                # print(label)
                labels_array = np.array([label]).reshape(-1, 1)
                transformed_label = self.data_clean.one_hot.transform(
                    labels_array
                )
                pbar.update(1)

                # print("|--Sampling--|")
                labels_final = torch.Tensor(transformed_label)
                sampled = model.generate_from_specific_vamp_prior(
                    vamp, n_points, labels_final
                )
                pbar.update(1)

                traf = self.data_clean.output_converter(
                    sampled, landing=True, lat_lon=lat_lon
                )
                pbar.update(1)

        return traf
    
    def generate_n_flight_per_labels(
        self,
        labels: list[str],
        n_points: int,
        batch_size: int = 500,
        lat_long: bool = True,
    ) -> list[Traffic]:
        model = self.model
        traff_list = []
        with torch.no_grad():
            for label in tqdm(labels, desc="Generation"):
                with tqdm(total=3, desc=f"Processing {label}") as pbar:
                    labels_array = np.full(batch_size, label).reshape(-1, 1)
                    transformed_label = self.data_clean.one_hot.transform(
                        labels_array
                    )

                    pbar.update(1)

                    # print("|--Sampling--|")
                    labels_final = torch.Tensor(transformed_label)
                    sampled = model.sample(n_points, batch_size, labels_final)
                    pbar.update(1)

                    traf = self.data_clean.output_converter(
                        sampled, landing=True, lat_lon=lat_long
                    )

                    pbar.update(1)
                    traff_list.append(traf)

        return traff_list
