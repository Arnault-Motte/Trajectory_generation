import datetime
import multiprocessing as mp
from collections import Counter
from datetime import time, timedelta
from multiprocessing import Pool

import torch
from scipy.spatial.distance import cdist, jensenshannon
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
from data_orly.src.generation.generation import Generator
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_TCN_VampPrior import (
    get_data_loader as get_data_loader_labels,
)
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from numpy._typing._array_like import NDArray
from traffic.algorithms.generation import compute_latlon_from_trackgs
from traffic.core import Flight, Traffic

DEFAULT_AIRCRAFT = "A320"


def e_distance(
    traj_gen_flt: np.ndarray,
    traj_og_flt: np.ndarray,
    metric: str = "euclidean",
) -> float:
    x = traj_gen_flt
    y = traj_og_flt
    dxy = cdist(x, y, metric=metric).mean()
    dxx = cdist(x, x, metric=metric).mean()
    dyy = cdist(y, y, metric=metric).mean()

    m, n = len(x), len(y)

    e_distance = 2 * dxy - dxx - dyy

    return e_distance


def e_distance_batched(
    traj_gen_flt: np.ndarray,
    traj_og_flt: np.ndarray,
    metric: str = "euclidean",
    batch_size: int = 1000,
) -> float:
    x = traj_gen_flt  # shape (m, d)
    y = traj_og_flt  # shape (n, d)
    m, n = len(x), len(y)

    # 1. Compute (2 / mn) * sum_{i,j} ||x_i - y_j||
    sum_dxy = 0.0
    for i in range(0, m, batch_size):
        x_batch = x[i : i + batch_size]
        dists = cdist(x_batch, y, metric=metric)  # shape (batch_size, n)
        sum_dxy += dists.sum()

    # 2. Compute (1 / m²) * sum_{i,j} ||x_i - x_j||
    sum_dxx = 0.0
    for i in range(0, m, batch_size):
        x_batch = x[i : i + batch_size]
        dists = cdist(x_batch, x, metric=metric)
        sum_dxx += dists.sum()

    # 3. Compute (1 / n²) * sum_{i,j} ||y_i - y_j||
    sum_dyy = 0.0
    for i in range(0, n, batch_size):
        y_batch = y[i : i + batch_size]
        dists = cdist(y_batch, y, metric=metric)
        sum_dyy += dists.sum()

    # Combine everything
    e_dist = (
        (2 / (m * n)) * sum_dxy
        - (1 / (m**2)) * sum_dxx
        - (1 / (n**2)) * sum_dyy
    )
    return e_dist


def traff_to_flatten_array(traff: Traffic) -> np.ndarray:
    traff_array = traff.data.to_numpy()


def convert_time_delta_to_start_str(time_delta: int) -> str:
    time = str(timedelta(seconds=time_delta))
    return time + ">"


def clean_time_stamp_flight(flight: Flight) -> Flight:
    """
    If the first timedelta is inferior to 0, adds its oposite to all timedeltas to have positive timedeltas.
    """
    data = flight.data
    data = data.reset_index()
    first_val = data.loc[0, "timedelta"]
    if first_val < 0:
        data["timedelta"] = data["timedelta"] - first_val
    return Flight(data)


def clean_time_stamp_traffic(
    traff: Traffic, desc: str = "Cleaning timestamps", max_workers: int = 1
) -> Traffic:
    res = traff.pipe(clean_time_stamp_flight).eval(
        desc=desc, max_workers=max_workers
    )
    return res


def create_scn_line(
    command: str, data: pd.Series, typecode: bool = False
) -> str:
    time = data["timedelta"]
    time_str = convert_time_delta_to_start_str(time)
    plane_id = f"AC{data['flight_id']}"
    line = f"{time_str} {command} {plane_id},"
    heading = data["track"]
    if typecode:
        line += f" {data['aircraft']},"
    line += f" {data['latitude']}, {data['longitude']}, {heading}, {data['altitude']}, {data['groundspeed']}"
    return line


def process_sub_chunk(chunk_df: pd.DataFrame) -> str:
    chunk_lines = []
    for _, row in chunk_df.iterrows():
        command = "CRE" if row["first"] else "MOVE"
        type_code = row["first"]
        # You must define create_scn_line() appropriately.
        line = create_scn_line(command, row, type_code)
        chunk_lines.append(line)
    return "\n".join(chunk_lines)


import traj_dist.distance as tdist


def reset_timestamp(traff: Traffic, date_og: datetime) -> Traffic:
    """
    Reset the timestamp of the traffic to at date_og
    """
    traff_data = traff.data
    traff_data["timestamp"] = (
        pd.to_timedelta(traff_data["timedelta"], unit="s") + date_og
    )
    traffic = Traffic(traff_data)
    return traffic


def compute_distances(
    traff: Traffic, simulated_traff: Traffic, num_point: int
) -> dict:
    """
    Computes the different distances between two traffs
    """
    from cartopy.crs import EuroPP

    traff = traff.compute_xy(EuroPP())
    simulated_traff = simulated_traff.compute_xy(EuroPP())

    traff = reset_timestamp(traff, simulated_traff.data["timestamp"].iloc[0])
    simulated_traff = reset_timestamp(
        simulated_traff, traff.data["timestamp"].iloc[0]
    )
    ret = {}
    for f2 in tqdm(simulated_traff):
        flight_id = f2.flight_id
        f1 = traff[flight_id]
        f1: Flight = f1.drop_duplicates(
            subset="timestamp"
        )  # the data is sometimes bad
        f1 = f1.before(f2.stop)
        f2 = f2.before(f1.stop)  # we want the flight to have the same duration

        # print(f1.resample(num_point).data[["x", "y"]].reset_index(drop=True).head())
        X1, X2 = (
            f1.resample(num_point).data[["x", "y"]].to_numpy(),
            f2.resample(num_point).data[["x", "y"]].to_numpy(),
        )
        ret[flight_id] = {
            "dtw": tdist.dtw(X1, X2),
            "edr": tdist.edr(X1, X2),
            "erp": tdist.erp(X1, X2),
            "frechet": tdist.frechet(X1, X2),
            "hausdorff": tdist.hausdorff(X1, X2),
            "lcss": tdist.lcss(X1, X2),
            "sspd": tdist.sspd(X1, X2),
        }

    return ret


class Evaluator:
    def __init__(
        self,
        gen: Generator,
        ini_traff: Traffic,
        max_num_wrokers: int = 0,
        seq_len: int = 200,
    ) -> None:
        self.gen = gen
        # if "flight_id" not in self.gen_traff.data.columns:
        #     self.gen_traff = self.gen_traff.assign_id().eval(
        #         desc="Generating flight ids", max_workers=max_num_wrokers
        #     )
        # self.gen_traff = clean_time_stamp_traffic(
        #     self.gen_traff, max_workers=max_num_wrokers
        # )
        self.ini_traff = ini_traff
        self.seq_len = seq_len
        self.max_num_workers = (
            max_num_wrokers if max_num_wrokers > 0 else mp.cpu_count()
        )

    def e_dist(
        self, n_t: int, label: str = "", scaler: MinMaxScaler = None
    ) -> float:
        """
        Computes the e-distance between the generated traffic and the original traffic.
        n controls the max number of trajectory considered
        """
        cond = label != ""
        # print("Conditioned : ", cond)

        generated = (
            self.gen.generate_n_flight_per_labels(
                labels=[label], n_points=n_t, lat_long=False
            )[0]
            .assign_id()
            .eval()
            if cond
            else self.gen.generate_n_flight(n_t, lat_long=False)
            .assign_id()
            .eval()
        )

        n, m = len(generated), len(self.ini_traff)

        # if n_t and (n_t > n or n_t > m):
        #     n_t = min(n, m)

        np_gen = convert_traff_numpy(generated.sample(n_t), scaler=scaler)
        np_traff = convert_traff_numpy(
            self.ini_traff.sample(min(n_t, m)), scaler=scaler
        )

        return e_distance(np_gen, np_traff)

    def compute_e_dist(
        self,
        number_of_trial: int = 100,
        n_t: int = 3000,
        label: str = "",
        scaler: MinMaxScaler = None,
    ) -> float:
        """
        Computes the mean of the e_distance after n number of trials
        """

        cond = label != ""
        # print("Conditioned : ", cond)

        list_gen = []

        for _ in tqdm(range(number_of_trial), desc="generating"):
            generated = (
                self.gen.generate_n_flight_per_labels(
                    labels=[label],
                    n_points=n_t,
                    lat_long=False,
                )[0]
                .assign_id()
                .eval()
                if cond
                else self.gen.generate_n_flight(n_t, lat_long=False)
                .assign_id()
                .eval()
            )
            list_gen.append(convert_traff_numpy(generated, scaler=scaler))

        m = len(self.ini_traff)

        list_traff = [
            convert_traff_numpy(
                self.ini_traff.sample(min(n_t, m)), scaler=scaler
            )
            for _ in range(number_of_trial)
        ]

        args_list = [(list_traff[i], list_gen[i]) for i in range(number_of_trial)]

        with Pool(processes=10) as pool:
            # e_distances = pool.map(self.e_dist_chunk, args_list)
            e_distances = list(
                tqdm(
                    pool.imap(e_dist_chunk, args_list),
                    total=len(args_list),
                )
            )

        return sum(e_distances) / number_of_trial

    def flight_navpoints_simulation(f: Flight) -> Flight:
        pass


def e_dist_chunk(
    args: tuple[np.ndarray, np.ndarray, int, MinMaxScaler],
) -> float:
    np_traff, np_gen = args

    return e_distance(np_gen, np_traff)


def convert_traff_numpy(
    traff: Traffic,
    columns: list[str] = ["track", "groundspeed", "timedelta", "altitude"],
    scaler: MinMaxScaler = None,
) -> np.ndarray:
    """
    convert a traff with the specified columns to a numpy array.
    """
    data = traff.data[columns]
    array = data.to_numpy(dtype=np.float32)
    if scaler:
        array = scaler.transform(array)
    array = array.reshape(len(traff), -1, len(columns))
    array = array.reshape(len(traff), -1)
    return array
