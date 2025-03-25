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


### Function to get the mean end point of the trajectories
def get_mean_end_point(traff: Traffic) -> tuple:
    total_lat = 0
    total_long = 0
    for flight in traff:
        lat_f = flight.data.iloc[-1]["latitude"]
        long_f = flight.data.iloc[-1]["longitude"]

        total_lat += lat_f
        total_long += long_f
        # print(f'({lat_f},{long_f})')
    n_traff = len(traff)
    mean_point = (total_lat / n_traff, total_long / n_traff)
    return mean_point


def get_mean_start_point(traff: Traffic) -> tuple:
    total_lat = 0
    total_long = 0
    for flight in traff:
        lat_f = flight.data.iloc[0]["latitude"]
        long_f = flight.data.iloc[0]["longitude"]

        total_lat += lat_f
        total_long += long_f
        # print(f'({lat_f},{long_f})')

    n_traff = len(traff)
    mean_point = (total_lat / n_traff, total_long / n_traff)
    return mean_point


def traffic_random_merge(file_names: list[str], total: bool = False) -> Traffic:
    traffics = [
        Traffic.from_file(file_name).aircraft_data() for file_name in file_names
    ]
    max_traffic_size = max([len(traf) for traf in traffics])
    num_to_draw = (
        max_traffic_size if total else max_traffic_size // len(traffics)
    )  # number of traffic to sample to have a balanced dataset (that is not to big)
    print(num_to_draw)
    print(max_traffic_size)
    print(len(traffics))
    lis_traf = [traf.sample(n=num_to_draw) for traf in traffics]
    sampled_traffic = sum(lis_traf)
    print("Size of custom traffic : ", len(sampled_traffic))
    return sampled_traffic


def is_ascending(traff: Traffic) -> bool:
    vr_mean = traff.data["vertical_rate"].mean()
    return vr_mean > 0


class Data_cleaner:
    def __init__(
        self,
        file_name: str = None,
        columns: list[str] = ["track", "altitude", "groundspeed", "timedelta"],
        seq_len: int = 200,
        file_names: list[str] = [],
        airplane_types_num: int = -1,
        total: bool = False,
    ):
        if len(file_names) == 0 and not file_name:
            raise ValueError("file_name and file_names are both None")
        self.basic_traffic_data = (
            Traffic.from_file(file_name).aircraft_data()
            if file_name
            else traffic_random_merge(file_names, total)
        )
        self.file_names = file_names
        self.total = total
        if airplane_types_num != -1:
            self.airplane_types_num = airplane_types_num
            self.basic_traffic_data = self.traffic_only_top_n(
                airplane_types_num, self.basic_traffic_data
            )
        self.columns = columns
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.n_traj = len(self.basic_traffic_data)
        self.seq_len = seq_len
        self.num_channels = len(columns)
        self.one_hot = OneHotEncoder(sparse_output=False)
        if len(file_names) != 0:
            self.traffic_ascend = self.is_ascedending_per_traffic()

    def traffic_only_top_n(self, top_num: int, tra: Traffic) -> Traffic:
        top_10 = self.get_top_10_planes()
        bound = min(top_num, len(top_10))
        top_n = top_10[:bound]
        top_n_typecodes = [keys for keys, _ in top_n]

        def simple(flight: Flight) -> Flight:
            return flight.assign(
                simple=lambda _: flight.typecode in top_n_typecodes
            )

        temp_traf = tra.iterate_lazy().pipe(simple).eval(desc="")
        final_traf = temp_traf.query("simple")
        print("Final traffic size : ", len(final_traf))
        return final_traf

    def dataloader_traffic_converter(
        self,
        data: DataLoader,
        num_epoch: int,
        landing: bool = False,
        labels: list[str] = [],
    ) -> Traffic:
        list_traff = []
        i = 0
        for batch in data:
            if i == num_epoch:
                break
            i += 1
            list_traff.append(batch[0])
        total_tensor = torch.cat(list_traff, dim=0)
        return (
            self.output_converter(total_tensor, landing=landing)
            if len(labels) == 0
            else self.output_converter_several_datasets(total_tensor, labels)
        )

    def output_converter(
        self, x_recon: torch.Tensor, landing: bool = False
    ) -> Traffic:
        mean_point = (
            self.get_mean_end_point()
            if not landing
            else self.get_mean_start_point()
        )
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
            forward=landing,
        )
        return Traffic(x_df)

    def is_ascedending_per_traffic(self) -> list[bool]:
        traffics = self.return_traff_per_dataset()
        return [is_ascending(traff) for traff in traffics]

    def get_mean_points(self) -> list[tuple]:
        all_means = []
        traffics = self.return_traff_per_dataset()
        is_ascend = self.traffic_ascend
        assert len(traffics) == len(is_ascend)
        for i, traff in enumerate(traffics):
            mean_point = (
                get_mean_start_point(traff)
                if is_ascend[i]
                else get_mean_end_point(traff)
            )
            all_means.append(mean_point)
        return all_means

    def output_converter_several_datasets(
        self, x_recon: torch.Tensor, labels: list[str]
    ) -> Traffic:
        mean_points = self.get_mean_points()

        coordinates = [
            {"latitude": mean_point[0], "longitude": mean_point[1]}
            for mean_point in mean_points
        ]

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
        print(len(labels), len(x_df), len(x_df) // self.seq_len)
        x_df["label"] = [
            labels[i // self.seq_len] for i in range(self.seq_len * len(labels))
        ]
        # print( x_df["label"])

        grouped_x_df = {
            group: group_df for group, group_df in x_df.groupby("label")
        }

        is_ascend = self.traffic_ascend
        all_labels = self.return_unique_labels_datasets()
        all_dfs = []
        print(all_labels)
        for i, l in enumerate(all_labels):
            print(l)
            df = grouped_x_df.get(l, pd.DataFrame()).reset_index(drop=True)
            if len(df) == 0:
                print("la")
                continue
            coord = coordinates[i]
            df = compute_latlon_from_trackgs(
                df,
                n_samples=len(df) // self.seq_len,
                n_obs=self.seq_len,
                coordinates=coord,
                forward=is_ascend[i],
            )

            all_dfs.append(df)

        f_df = pd.concat(all_dfs)

        return Traffic(f_df)

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

    def clean_data_several_datasets(self) -> np.ndarray:
        flights_og_list = [
            self.basic_traffic_data.data.iloc[
                i * self.seq_len : (i + 1) * self.seq_len
            ]
            for i in range(len(self.basic_traffic_data))
        ]
        # checked

        flights = []
        for data in flights_og_list:
            data = data[self.columns]
            np_data = data.to_numpy()
            flights.append(np_data)

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

    def return_labels_datasets(self) -> np.ndarray:
        """
        Returns labels that represents the dataset of origin of the data
        """
        data_set_len = len(self.basic_traffic_data)
        number_of_files = len(self.file_names)
        fligh_per_dataset = data_set_len // number_of_files
        print(fligh_per_dataset)

        labels = [f"d{i // fligh_per_dataset}" for i in range(data_set_len)]
        labels_array = np.array(labels).reshape(-1, 1)
        return self.one_hot.fit_transform(labels_array)

    def return_unique_labels_datasets(self) -> list[str]:
        return [f"d{i}" for i in range(len(self.file_names))]

    def return_traff_per_dataset(self) -> list[Traffic]:
        data_set_len = len(self.basic_traffic_data)
        number_of_files = len(self.file_names)
        fligh_per_dataset = data_set_len // number_of_files
        data = self.basic_traffic_data

        traff_list = [
            Traffic(
                data.data.iloc[
                    i * fligh_per_dataset * self.seq_len : (i + 1)
                    * fligh_per_dataset
                    * self.seq_len
                ]
            )
            for i in range(number_of_files)
        ]
        return traff_list

    ### Function to get the mean end point of the trajectories
    def get_mean_end_point(self) -> tuple:
        return get_mean_end_point(self.basic_traffic_data)

    def get_mean_start_point(self) -> tuple:
        return get_mean_start_point(self.basic_traffic_data)

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

    def reordered_labels(self, labels: np.ndarray,num_flight: int = None) -> np.ndarray:
        """
        Reorder the labels to follow the order of the list of datasets.
        to use when labels are related to the dataset of origin of the traffics
        Labels must be an array of strings of shape (n_flight,1)
        """
        if not num_flight:
            num_flight = len(labels)
        labels_order = self.return_unique_labels_datasets()
        labels_count = dict(Counter(labels.flatten().tolist()[:num_flight]))
        reordered_labels = [
            label for label in labels_order for _ in range(labels_count[label])
        ]
        return np.array(reordered_labels).reshape(-1,1)


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


def compute_vertical_rate_flight(flight: Flight) -> Flight:
    data = flight.data
    delta_time = data["timedelta"].diff()
    delta_altitude = data["altitude"].diff()
    data["vertical_rate"] = delta_altitude / delta_time * 60
    data.loc[data.index[0], "vertical_rate"] = data["vertical_rate"].iloc[1]
    return Flight(data)


def compute_vertical_rate(traffic: Traffic) -> Traffic:
    if "vertical_rate" in traffic.data.columns:
        return traffic
    return traffic.iterate_lazy().pipe(compute_vertical_rate_flight).eval()
