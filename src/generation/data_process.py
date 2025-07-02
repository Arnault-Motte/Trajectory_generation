import datetime
from collections import Counter
import pickle

import torch
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
from numpy._typing._array_like import NDArray
from traffic.algorithms.generation import compute_latlon_from_trackgs
from traffic.core import Flight, Traffic


### Function to get the mean end point of the trajectories
def get_mean_end_point(traff: Traffic) -> tuple:
    """
    Returns the mean end point of all the trajectories in the traffic
    """
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
    """
    Returns the mean of the start point of all the trajectories in the traffic
    """
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
    """
    Merges traffic.
    If total is false balances the two traffic, by undersampling all traffics to the size of the smallest one
    """
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
    """used to chech if a traffic contains take off"""
    vr_mean = traff.data["vertical_rate"].mean()
    return vr_mean > 0


def traffic_only_chosen(
    traff: Traffic, chosen_typecodes: list[str], seq_len: int
) -> tuple[Traffic, list[str]]:
    """Returns the selected typecodes in order and the filtered traffic containing only this datasets"""
    data = traff.data

    typecodes = [flight.typecode for flight in traff]
    # data["typecode"] = [
    #     typecode for typecode in typecodes for _ in range(seq_len)
    # ]
    num_typecodes = Counter(typecodes)
    ordered_typecodes = [
        typecode
        for typecode, _ in num_typecodes.most_common()
        if typecode in chosen_typecodes
    ]
    filtered_data = data[
        data["typecode"].isin(chosen_typecodes)
    ]  ##change by query
    new_traff = Traffic(filtered_data)

    return new_traff, ordered_typecodes


def add_time_deltas(f: Flight) -> Flight:
    """Adds time deltas to a flight"""
    df = f.data
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return f.assign(
        timedelta=(
            pd.to_datetime(df["timestamp"])
            - pd.to_datetime(df["timestamp"].iloc[0])
        ).dt.total_seconds()
    )


def compute_time_delta(t: Traffic) -> Traffic:
    """
    Adds a time delta columns to the dataset of every flight in the traffic
    """
    return t.pipe(add_time_deltas).eval(
        desc="computing time deltas", max_workers=4
    )


class Data_cleaner:
    """Object that prepare the take off or landing dataset to be used by the model"""

    def __init__(
        self,
        file_name: str = None,  ##file to be loaded
        columns: list[str] = [
            "track",
            "altitude",
            "groundspeed",
            "timedelta",
        ],  # columns to be selected to be used by the model
        seq_len: int = 200,  # the number of point in the file flights
        airplane_types_num: int = -1,  # number of airplanes models to consider, if different than 10 uses the 10 most commun aircraft moels trajectories
        chosen_typecodes: list[
            str
        ] = [],  # list of typecodes to consider, left it to [] if airplane_types_num is different from -1
        total: bool = False,
        traff: Traffic = None,  # traffic to load, to use if file name is none
        min_max: MinMaxScaler = None,  # put the scaler you want to use, if None will create the min max scaler itself
        diff: bool = False,
        one_hot: OneHotEncoder = None,
        no_data: bool = False,
        mean_point: tuple[float, float] = None,
    ):
        # Check if we have any filename provided

        if no_data:
            self.one_hot = one_hot
            self.scaler = min_max
            self.mean_point = mean_point
            self.seq_len = seq_len
            self.num_channels = 4
            self.columns = columns
            return

        if not file_name and traff is None:
            raise ValueError("No traffic given")

        # defininig the traffic
        if traff is not None:
            self.basic_traffic_data = traff
        elif file_name:
            self.basic_traffic_data = Traffic.from_file(file_name)

        if "timedelta" not in self.basic_traffic_data.data.columns:
            self.basic_traffic_data = compute_time_delta(
                self.basic_traffic_data
            )

        if "typecode" not in self.basic_traffic_data.data.columns:
            self.basic_traffic_data = self.basic_traffic_data.aircraft_data()

        self.total = total

        # If we want to study only the top airplane_types_num most common typecodes
        if airplane_types_num != -1 and len(chosen_typecodes) == 0:
            self.airplane_types_num = airplane_types_num
            self.basic_traffic_data = self.traffic_only_top_n(
                airplane_types_num, self.basic_traffic_data
            )

        chosen_typecodes_temp = chosen_typecodes

        # if we want to study specific aircrafts trajectories
        if len(chosen_typecodes) != 0:
            self.basic_traffic_data, chosen_typecodes_temp = (
                traffic_only_chosen(
                    self.basic_traffic_data, chosen_typecodes, seq_len=seq_len
                )
            )

        self.chosen_types = (
            chosen_typecodes_temp
            if len(chosen_typecodes) != 0
            else [label for label, _ in self.get_top_n_typecodes()]
        )
        print(self.chosen_types)

        self.columns = columns
        # define scaler
        self.fit_scale = True
        if not min_max:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.fit_scale = False
        else:
            self.scaler = min_max

        self.n_traj = len(self.basic_traffic_data)
        self.seq_len = seq_len
        self.num_channels = len(columns)
        self.one_hot = OneHotEncoder(sparse_output=False)

        if "vertical_rate" not in self.basic_traffic_data.data.columns:
            self.basic_traffic_data =  self.comp_vertical_rates()
        self.mean_point = None
        if is_ascending(self.basic_traffic_data):
            self.mean_point = self.get_mean_start_point()
        else:
            self.mean_point = self.get_mean_end_point()

    def save_scalers(self,path:str)->None:
        """
        Saves the scaler, the mean end point, and the one hot encoder as a tuple in the designated pickle file.
        """
        scalers = (self.scaler,self.one_hot,self.mean_point,self.columns)
        with open(path,"wb") as f:
            pickle.dump(scalers,file = f)
    
    def load_scalers(self,path:str)->None:
        """
        Laods the scaler, the mean end point, and the one hot encoder as a tuple from the designated pickle file.
        """
        with open(path, "rb") as f:
            self.scaler,self.one_hot,self.mean_point,self.columns = pickle.load(f)
        return


    def get_typecodes_labels(self) -> list[str]:
        """
        Returns the typecodes in the order of interpretation
        """
        return self.chosen_types

    def comp_vertical_rates(self) -> Traffic:
        """
        Computes the vertical rates of the underlying traffic.
        If a column vertical_rate already existed it overwrites it
        """
        if (
            "vertical_rate" in self.basic_traffic_data.data.columns
        ):  # deleting the existing vertical rate column
            self.basic_traffic_data = self.basic_traffic_data.drop(
                columns=["vertical_rate"]
            )
        self.basic_traffic_data = compute_vertical_rate(self.basic_traffic_data)
        return self.basic_traffic_data

    def return_typecode_array(self, typecode: str):
        """Returns the trajectories linked to the typecode in entry as a scaled tensor"""
        traffic = return_traff_per_typecode(
            self.basic_traffic_data, [typecode]
        )[0]
        cleaned = self.clean_data_specific(traffic, fit_scale=False)
        data = torch.tensor(cleaned, dtype=torch.float32)
        return data

    def traffic_only_top_n(self, top_num: int, tra: Traffic) -> Traffic:
        """Returns the traffic formed of only the n most commun typecodes"""
        top_10 = self.get_top_n_typecodes(top_num)
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

    def traffic_per_label(self) -> list[Traffic]:
        """
        Return a traffic per defined label (typecode)
        """
        x_df = self.basic_traffic_data.data
        if "typecode" not in x_df.columns:
            raise ValueError(
                "Airplane data must be set before calling the function"
            )
        grouped_x_df = {
            group: group_df for group, group_df in x_df.groupby("typecode")
        }
        label_list_order = self.get_typecodes_labels()
        return [Traffic(grouped_x_df[label]) for label in label_list_order]

    def dataloader_traffic_converter(
        self,
        data: DataLoader,
        num_epoch: int,
        landing: bool = False,
        labels: list[str] = [],
    ) -> Traffic:
        """Convert a dataloader to a traffic"""
        list_traff = []
        i = 0
        for batch in data:
            if i == num_epoch:
                break
            i += 1
            list_traff.append(batch[0])
        total_tensor = torch.cat(list_traff, dim=0)
        return self.output_converter(total_tensor, landing=landing)

    def output_converter(
        self,
        x_recon: torch.Tensor,
        landing: bool = False,
        lat_lon: bool = True,
        mean_point: tuple[float, float] = None,
        seq_len: int = None,
    ) -> Traffic:
        """Converts the output of the model into an readable traffic"""
        if not seq_len:
            seq_len = self.seq_len
        if not mean_point:
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
            print("No altitude recieved, possible error")
            altitude = [10000 for _ in range(len(x_df))]
            x_df["altitude"] = altitude

        if "timedelta" not in self.columns:
            print("No time delta recieved, possible error")
            time_delta = [2 * (i // self.seq_len) for i in range(len(x_df))]
            x_df["timedelta"] = time_delta

        x_df["timestamp"] = datetime.datetime(2020, 5, 17) + x_df[  # noqa: DTZ001
            "timedelta"
        ].apply(lambda x: datetime.timedelta(seconds=x))
        x_df["icao24"] = [str(i // self.seq_len) for i in range(len(x_df))]
        x_df["callsign"] = x_df["icao24"]

        if lat_lon:
            x_df = compute_latlon_from_trackgs(
                x_df,
                n_samples=batch_size,
                n_obs=seq_len,
                coordinates=coordinates,
                forward=landing,
            )
        return Traffic(x_df)

    def clean_data_specific(
        self, data_to_clean: Traffic, fit_scale: bool = True
    ) -> np.ndarray:
        """turns the the entered traffic into a scaled np array"""
        if data_to_clean is None:
            return np.array([])
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
        trajectories_scaled = (
            self.scaler.fit_transform(trajectories_reshaped)
            if fit_scale
            else self.scaler.transform(trajectories_reshaped)
        )

        # Reshape back to the original dimensions
        trajectories_scaled = trajectories_scaled.reshape(original_shape)

        return trajectories_scaled

    def clean_data(self) -> np.ndarray:
        """turns the underlying traffic into a scaled np array"""
        return self.clean_data_specific(
            self.basic_traffic_data, fit_scale=not self.fit_scale
        )

    def return_labels(self) -> np.ndarray:
        """Returns all the labels in one hot encoded labels"""
        labels = np.array(
            [flight.typecode for flight in self.basic_traffic_data]
        ).reshape(-1, 1)
        return self.one_hot.fit_transform(labels)
        # return np.array([flight.typecode for flight in self.basic_traffic_data])

    ### Function to get the mean end point of the trajectories
    def get_mean_end_point(self) -> tuple:
        if not self.mean_point:
            self.mean_point = get_mean_end_point(self.basic_traffic_data)
        return self.mean_point

    def get_mean_start_point(self) -> tuple:
        if not self.mean_point:
            self.mean_point =get_mean_start_point(self.basic_traffic_data)
        return self.mean_point

    ### Function to unscale the data
    def unscale(self, traj: Traffic) -> np.ndarray:
        """allows to unscale the data passes in a traffic in entry"""
        original_shape = traj.shape
        traj_reshaped = traj.reshape(-1, original_shape[-1])
        traj_unscaled = self.scaler.inverse_transform(traj_reshaped)
        traj_unscaled = traj_unscaled.reshape(original_shape)
        return traj_unscaled

    def get_top_n_typecodes(self, n_type: int = 10) -> list[tuple[str, int]]:
        """returs the n_type most commun typecodes in the dataset"""
        data = self.basic_traffic_data
        # start_time = [flight.start for flight in data]
        df = data.data

        flight_counts = df["typecode"].value_counts()
        top_10_planes = flight_counts.head(n_type).items()
        return list(top_10_planes)

    def transform_back_labels_tensor(
        self, label_tensor: torch.Tensor
    ) -> np.ndarray:
        """transform a tensor of labels back into a nd array os strings corresponding to every label"""
        array = label_tensor.cpu().numpy()
        inverse = self.one_hot.inverse_transform(array)
        return inverse

    def return_flight_id_for_label(self, label: str) -> list[str]:
        """returns all the flights ids, linked to a specific aircraft"""
        labels = [
            f.flight_id for f in self.basic_traffic_data if f.typecode == label
        ]
        return labels


def return_traff_per_typecode(
    traff: Traffic, typecodes: list[str], max_worker: int = 2
) -> list[Traffic]:
    """
    Returns a list of traff, where each traff
    contains only aircraft of the corresponding label
    in typecodes
    """
    if "typecode" not in traff.data.columns:
        traff = traff.aircraft_data()
    # print(traff.data.columns)
    traffs_l = []
    # print(typecodes)
    for typecode in tqdm(typecodes, desc="test"):
        traff_i = traff.query(f'typecode == "{typecode}"')
        traffs_l.append(traff_i)
    return traffs_l


def compute_vertical_rate_flight(flight: Flight) -> Flight:
    """Computes the vertical rate of a flight"""
    data = flight.data
    delta_time = data["timedelta"].diff()
    delta_altitude = data["altitude"].diff()
    data["vertical_rate"] = delta_altitude / delta_time * 60
    data.loc[data.index[0], "vertical_rate"] = data["vertical_rate"].iloc[1]
    return Flight(data)


def compute_vertical_rate(traffic: Traffic) -> Traffic:
    """
    Computes all the vertical rates in the data
    """
    if "vertical_rate" in traffic.data.columns:
        return traffic
    return (
        traffic.iterate_lazy()
        .pipe(compute_vertical_rate_flight)
        .eval(desc="computing vertical rates", max_workers=4)
    )


def jason_shanon(vr_rates_1: np.ndarray, vr_rates_2: np.ndarray) -> float:
    """ "
    Compte jason shanon divergence for two different vertical rates distribution sets

    """
    bins = np.histogram_bin_edges(
        np.concatenate([vr_rates_1, vr_rates_2]), bins="auto"
    )

    p, _ = np.histogram(vr_rates_1, bins=bins, density=True)
    q, _ = np.histogram(vr_rates_2, bins=bins, density=True)

    # Avoid zero probabilities
    p += 1e-10
    q += 1e-10

    # Compute KL Divergence
    js_div = jensenshannon(p, q)  # KL(P || Q)
    return js_div


def save_smaller_traffic(
    traff: Traffic, save_path: str, sizes: list[int] = [200, 500]
) -> None:
    """
    Creat subsamples of traff of sizes sizes.
    Saves them in save path which must be a path to a pkl file.
    """

    n = len(traff)

    for nf in sizes:
        if nf > n:
            print(f"{nf} is bigger than the size of the traff")
            continue
        path = save_path.split(".")[0] + f"_{nf}.pkl"
        print(path)
        sampled = traff.sample(nf)
        sampled.to_pickle(path)

    return


def combine_save(traffs: list[Traffic], path: str) -> None:
    """Combines the traffs in the list and saves them to path"""
    final_traff: Traffic = sum(traffs)
    final_traff.to_pickle(path)
    return


def clean_data(
    traffic: Traffic, scaler: MinMaxScaler, columns: list
) -> np.ndarray:
    flights = []
    for flight in traffic:
        data = flight.data
        data = data[columns]
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
    trajectories_scaled = scaler.transform(trajectories_reshaped)

    # Reshape back to the original dimensions
    trajectories_scaled = trajectories_scaled.reshape(original_shape)

    return trajectories_scaled


def split_flight_padd(
    f: Flight, split: float, columns_to_zero: list[str], beggining: bool = True
) -> Flight:
    """
    Zeros out split % of the dataset
    """
    data = f.data
    zero_rows_num = int(split * len(data))
    if beggining:
        data.iloc[:zero_rows_num, data.columns.get_indexer(columns_to_zero)] = 0
    else:
        data.iloc[zero_rows_num:, data.columns.get_indexer(columns_to_zero)] = 0
    return Flight(data)


def split_trajectories_label(
    traffic: Traffic,
    split: float = 0.5,
    columns_to_zero: list[str] = [
        "track",
        "altitude",
        "timedelta",
        "groundspeed",
    ],
) -> tuple[Traffic, Traffic]:
    T1 = traffic.pipe(
        split_flight_padd,
        split=split,
        beggining=True,
        columns_to_zero=columns_to_zero,
    ).eval(max_workers=6, desc="")
    labels = traffic.pipe(
        split_flight_padd,
        split=split,
        beggining=False,
        columns_to_zero=columns_to_zero,
    ).eval(max_workers=6, desc="")
    return T1, labels


def filter_missing_values(
    t: Traffic,
    columns: list[str],
    threshold: float = 0.05,
    max_wrokers: int = 1,
) -> Traffic:
    """
    Function to use to deal with missing values in a Traffic.
    The function removes the flights from traffic, that have more than threshold % of rows with a Nan values.
    The function only considers the rows stated in columns

    For the flights that have less threshold of their rows with Nan values,
    the rows are deleted. Then the flight is resampled to its original len.

    """

    t = t.pipe(
        filter_missing_values_flight, columns=columns, threshold=threshold
    ).eval(desc="Filtering missing values", max_workers=max_wrokers)
    return t


def filter_missing_values_flight(
    f: Flight, columns: list[str], threshold: float = 0.05
) -> Flight | None:
    """
    Same as filter_missing_values but for a single flight
    """
    df = f.data

    count = df[columns].isna().any(axis=1).sum() / len(
        df
    )  # frequency of nan values

    if count > threshold:
        return None  # to many nan
    else:
        n = len(df)  # number of point to resample
        df_clean = df.dropna(subset=columns)  # df wiht no na
        new_f = Flight(df_clean).resample(n)
        return new_f


def return_labels(traffic:Traffic,one_hot:OneHotEncoder) -> np.ndarray:
        """Returns all the labels in one hot encoded labels"""
        labels = np.array(
            [flight.typecode for flight in traffic]
        ).reshape(-1, 1)
        return one_hot.transform(labels)