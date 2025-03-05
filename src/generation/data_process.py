import datetime

import torch
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from traffic.algorithms.generation import compute_latlon_from_trackgs
from traffic.core import Flight, Traffic


class Data_cleaner:
    def __init__(
        self,
        file_name: str = "landings_LFPO_06.pkl",
        columns: list[str] = ["track", "altitude", "groundspeed", "timedelta"],
        seq_len: int = 200,
    ):
        self.basic_traffic_data = Traffic.from_file(file_name)
        self.columns = columns
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.n_traj = len(self.basic_traffic_data)
        self.seq_len = seq_len
        self.num_channels = len(columns)

    def output_converter(self, x_recon: torch.Tensor) -> Traffic:
        mean_point = self.get_mean_end_point()
        coordinates = {"latitude": mean_point[0], "longitude": mean_point[1]}

        batch_size = x_recon.size(0)
        x_np = x_recon.cpu().detach().numpy()
        x_np = self.unscale(x_np)
        x_np = x_np.reshape(-1, self.num_channels)
        x_df = pd.DataFrame(x_np, columns=self.columns)

        if "altitude" not in self.columns:
            altitude = [10000 for _ in range(len(x_df))]
            x_df["altitude"] = altitude

        if "timedelta" not in self.columns:
            time_delta = [2 * (i // self.seq_len) for i in range(len(x_df))]
            x_df["timedelta"] = time_delta

        x_df["timestamp"] = datetime.datetime(2020, 5, 17) + x_df[  # noqa: DTZ001
            "timedelta"
        ].apply(lambda x: datetime.timedelta(seconds=x))
        x_df["icao24"] = [str(i // self.seq_len) for i in range(len(x_df))]

        x_df = compute_latlon_from_trackgs(
            x_df,
            n_samples=batch_size,
            n_obs=self.seq_len,
            coordinates=coordinates,
            forward=False,
        )
        return Traffic(x_df)

    def clean_data(self) -> np.ndarray:
        flights = []
        for flight in self.basic_traffic_data:
            data = flight.data
            data = data[self.columns]
            np_data = data.to_numpy()
            flights.append(np_data)
            # create data loader

        trajectories = np.stack(flights).astype(np.float32)

        # Store the original shape for later reconstruction
        original_shape = (
            trajectories.shape
        )  # e.g., (n_flights, n_rows, n_features)  # noqa: E501

        # Reshape to 2D for scaling: each row is an observation
        trajectories_reshaped = trajectories.reshape(-1, original_shape[-1])

        # Initialize and fit the scaler
        trajectories_scaled = self.scaler.fit_transform(trajectories_reshaped)

        # Reshape back to the original dimensions
        trajectories_scaled = trajectories_scaled.reshape(original_shape)

        return trajectories_scaled

    ### Function to get the mean end point of the trajectories
    def get_mean_end_point(self) -> tuple:
        total_lat = 0
        total_long = 0
        for flight in self.basic_traffic_data:
            lat_f = flight.data.iloc[-1]["latitude"]
            long_f = flight.data.iloc[-1]["longitude"]

            total_lat += lat_f
            total_long += long_f
            # print(f'({lat_f},{long_f})')

        mean_point = (total_lat / self.n_traj, total_long / self.n_traj)
        return mean_point

    ### Function to unscale the data
    def unscale(self, traj: Traffic) -> np.ndarray:
        original_shape = traj.shape
        traj_reshaped = traj.reshape(-1, original_shape[-1])
        traj_unscaled = self.scaler.inverse_transform(traj_reshaped)
        traj_unscaled = traj_unscaled.reshape(original_shape)
        return traj_unscaled

    def first_n_flight_delta_time(
        self, traf: Traffic, n_flight: int = 10, n_point: int = 10
    ) -> list[list[float]]:
        return [
            flight.data["timedelta"][:n_point] for flight in traf[:n_flight]
        ]


def filter_outlier(traff: Traffic) -> Traffic:
    def simple(flight:Flight) -> Flight:
        return flight.assign(simple=lambda x: flight.shape.is_simple)

    t_to = traff.iterate_lazy().pipe(simple).eval(desc ="")
    t_to = t_to.query("simple")
    return t_to
