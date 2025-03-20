import datetime
from collections import Counter

import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from numpy._typing._array_like import NDArray
from traffic.algorithms.generation import compute_latlon_from_trackgs
from traffic.core import Flight, Traffic


# def traffic_random_merge(self,file_names : list[str]) -> None:
#     traffics = [Traffic.from_file(file_name).aircraft_data() for file_name in file_names]
#     max_traffic_size = max()

class Data_cleaner:
    def __init__(
        self,
        file_name: str = "landings_LFPO_06.pkl",
        columns: list[str] = ["track", "altitude", "groundspeed", "timedelta"],
        seq_len: int = 200,
        #file_names: list[str]= None,
    ):
        # if not file_name and not file_name:
        #     raise  ValueError("file_name and file_names are both None")
        self.basic_traffic_data = Traffic.from_file(file_name).aircraft_data()
        self.columns = columns
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.n_traj = len(self.basic_traffic_data)
        self.seq_len = seq_len
        self.num_channels = len(columns)
        self.one_hot = OneHotEncoder(sparse_output=False)


    
    def dataloader_traffic_converter(
        self, data: DataLoader, num_epoch: int
    ) -> Traffic:
        list_traff = []
        i = 0
        for batch in data:
            if i == num_epoch:
                break
            i += 1
            list_traff.append(batch[0])
        total_tensor = torch.cat(list_traff, dim=0)
        return self.output_converter(total_tensor)

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

    def clean_data_specific(self, data_to_clean) -> np.ndarray:
        flights = []
        for flight in data_to_clean:
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

    def return_labels(self) -> np.ndarray:
        labels = np.array(
            [flight.typecode for flight in self.basic_traffic_data]
        ).reshape(-1, 1)
        return self.one_hot.fit_transform(labels)
        # return np.array([flight.typecode for flight in self.basic_traffic_data])

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

    def get_top_10_planes(self) -> list[tuple[str, int]]:
        data = self.basic_traffic_data
        # start_time = [flight.start for flight in data]
        df = data.data

        flight_counts = df["typecode"].value_counts()
        top_10_planes = flight_counts.head(10).items()
        return list(top_10_planes)

        # df["start"] = [start_time[i//200] for i in range(len(df))]
        # data = Traffic(df)
        # landings_temp = data.groupby("icao24").first().reset_index()
        # planes = Counter(landings_temp['typecode'])
        # return planes.most_common(10)


def filter_outlier(traff: Traffic) -> Traffic:
    def simple(flight: Flight) -> Flight:
        return flight.assign(simple=lambda x: flight.shape.is_simple)

    t_to = traff.iterate_lazy().pipe(simple).eval(desc="")
    t_to = t_to.query("simple")
    return t_to


# Returns true if the flight path is considered smooth enough checks at each segment that it is in a similar direction as the precedent n_segment
def flight_soft(flight: Flight, n_segment: int = 3, limit: float = 0) -> bool:
    # data = flight.data
    # latitudes = data["latitude"]
    # longitude = data["longitude"]
    # latitude_vectors = latitudes[1:] - latitudes[:-1]
    # longitude_vectors = longitude[1:] - longitude[:-1]
    # latitudes_padding = pd.Series([0 for _ in range(n_segment-1)],name="latitude")
    # longitude_padding = pd.Series([0 for _ in range(n_segment-1)],name="longitude")
    # latitude_vectors = pd.concat([latitudes_padding,latitude_vectors])
    # longitude_vectors = pd.concat([longitude_padding,longitude_vectors])
    # latitude_mean_vectors = latitude_vectors.rolling(window=n_segment).mean()
    # longitude_mean_vectors = longitude_vectors.rolling(window=n_segment).mean()
    # dot_product = (latitude_vectors[n_segment-1:] * latitude_mean_vectors[n_segment-1:] ) + (longitude_vectors[n_segment-1:]  * longitude_mean_vectors[n_segment-1:] )
    # return (dot_product > limit).any()
    data = flight.data
    latitudes = data["latitude"]
    longitude = data["longitude"]

    # Compute vectors between consecutive points
    latitude_vectors = latitudes.diff().fillna(0)
    longitude_vectors = longitude.diff().fillna(0)

    # Compute rolling means of the vectors
    latitude_mean_vectors = latitude_vectors.rolling(window=n_segment).mean()
    longitude_mean_vectors = longitude_vectors.rolling(window=n_segment).mean()

    # Compute dot product
    dot_product = (latitude_vectors * latitude_mean_vectors) + (
        longitude_vectors * longitude_mean_vectors
    )

    # Check if any dot product exceeds the limit
    return (dot_product > limit).any()


def filter_smoothness(
    traff: Traffic, n_segment: int = 3, limit: float = 0
) -> Traffic:
    def simple(flight: Flight) -> Flight:
        test_val = flight_soft(flight, n_segment, limit)
        return flight.assign(simple=lambda _: test_val)

    t_to = traff.iterate_lazy().pipe(simple).eval(desc="")
    t_to = t_to.query("simple")
    return t_to

def compute_vertical_rate_flight(flight : Flight) -> Flight:
        data = flight.data
        delta_time = data["timedelta"].diff()
        delta_altitude = data["altitude"].diff()
        data["vertical_rate"] = delta_altitude / delta_time *60
        data.loc[data.index[0], "vertical_rate"] = data["vertical_rate"].iloc[1]
        return Flight(data)


def compute_vertical_rate(traffic: Traffic) -> Traffic:
    return traffic.iterate_lazy().pipe(compute_vertical_rate_flight).eval()  