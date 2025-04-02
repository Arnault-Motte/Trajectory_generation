import datetime
import multiprocessing as mp
from collections import Counter
from datetime import timedelta

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
    traj_gen_flt=np.ndarray, traj_og_flt=np.ndarray, metric:str="euclidean"
) -> float:
    x = traj_gen_flt
    y = traj_og_flt
    dxy = cdist(x, y, metric=metric)
    dxx = cdist(x, y, metric=metric)
    dyy = cdist(y, y, metric=metric)

    m, n = len(x), len(y)

    e_distance = (
        (2 / (m * n)) * np.sum(dxy)
        - (1 / (m**2)) * np.sum(dxx)
        - (1 / (n**2)) * np.sum(dyy)
    )

    return e_distance


def traff_to_flatten_array(traff : Traffic) -> np.ndarray:
    traff_array = traff.data.to_numpy()
    


def convert_time_delta_to_start_str(time_delta: int) -> str:
    time = str(timedelta(seconds=time_delta))
    return time + ">"


def clean_time_stamp_flight(flight: Flight) -> Flight:
    """
    If the first timedelta is inferior to 0 it adds it to all timedeltas in order to only have positive timedeltas values
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


class Evaluator:
    def __init__(
        self,
        gen_traff: Traffic,
        ini_traff: Traffic,
        max_num_wrokers: int = 0,
        labels: list[str] = [],
        seq_len: int = 200,
    ) -> None:
        self.gen_traff = gen_traff
        if "flight_id" not in self.gen_traff.data.columns:
            self.gen_traff = self.gen_traff.assign_id().eval(
                desc="Generating flight ids", max_workers=max_num_wrokers
            )
        self.gen_traff = clean_time_stamp_traffic(
            self.gen_traff, max_workers=max_num_wrokers
        )
        self.ini_traff = ini_traff
        self.labels = labels
        self.seq_len = seq_len
        self.max_num_workers = (
            max_num_wrokers if max_num_wrokers > 0 else mp.cpu_count()
        )

    def scn_for_traff(
        self, scn_filename: str, traff: Traffic, chunck_size: int = 1000
    ) -> None:
        data = self.update_traff(traff)
        chunck_num = len(data) // chunck_size
        data_list = np.array_split(data, chunck_num)

        with open(scn_filename, "w") as file:
            for data_i in tqdm(data_list, desc="writing"):
                sub_chunk = np.array_split(data_i, self.max_num_workers)
                with mp.Pool(self.max_num_workers) as pool:
                    res = pool.map(process_sub_chunk, sub_chunk)
                final = "\n".join(res)
                file.write(final)

    def update_traff(self, traff: Traffic) -> pd.DataFrame:
        seq_len = self.seq_len
        data = traff.data
        cd = data.index % seq_len == 0
        data["first"] = False
        data.loc[cd & (data["timedelta"] < 0), "timedelta"] = 0
        data.loc[cd, "first"] = True
        data["aircraft"] = [self.labels[i // seq_len] for i in range(len(data))]
        data = data.sort_values(by="timedelta", ascending=True)
        return data

    def create_scn(self, scn_file_name: str) -> None:
        """
        Creates two scn files corresponding to the generated trajectories.
        Don't put the .scn at the end of the filename.
        """
        self.scn_for_traff(scn_file_name + "_generated.scn", self.gen_traff)
        self.scn_for_traff(scn_file_name + "_generated.scn", self.ini_traff)

    def e_dist(self,n_t:int = None)->float:
        """
        Computes the e-distance between the generated traffic and the original traffic.
        n controls the max number of trajectory considered
        """
        n,m = len(self.gen_traff),len(self.ini_traff)

        

        if n_t and (n_t > n or n_t > m):
            n_t = min(n,m)
        if not n_t:
            return e_distance()
        
    def flight_navpoints_simulation(f: Flight)->Flight:
        pass