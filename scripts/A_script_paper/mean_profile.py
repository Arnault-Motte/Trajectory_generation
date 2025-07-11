import sys  # noqa: I001
import os

from tqdm import tqdm

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.generation.data_process import (
    Data_cleaner,
    return_labels,
    compute_time_delta,
    compute_vertical_rate,
)
from data_orly.src.generation.generation import Generator, ONNX_Generator
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_ONNX import CVAE_ONNX
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.test_display import (
    plot_traffic,
    vertical_rate_profile_2,
)
from data_orly.src.simulation import Simulator
from traffic.core import Traffic, Flight
from pitot import aero
import altair as alt
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

import statistics


def max_alt_diff(traff: Traffic) -> float:
    return statistics.median(
        [flight.data["altitude"].diff().median() for flight in traff]
    )


def max_alt_diff_per_traff(traff: Traffic) -> float:
    return [
        flight.data[flight.data["altitude"] < 20000]["altitude"]
        .diff()
        .abs()
        .max()
        for flight in traff
    ]


def return_centroid(traff: Traffic) -> int:
    ids = [f.flight_id for f in traff]
    raw_features = [
        f.data[f.data["altitude"] < 20000][["CAS", "altitude"]]
        .astype(float)
        .values
        for f in traff
    ]

    min_len = min(arr.shape[0] for arr in raw_features)
    raw_features = [arr[:min_len] for arr in raw_features]

    stacked = np.stack(raw_features)

    flat = stacked.reshape(len(stacked), -1)

    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8

    normalized = (flat - mean) / std

    distances = squareform(pdist(normalized))

    centroid_index = distances.mean(axis=1).argmin()

    centroid_id = ids[centroid_index]
    return centroid_id


# def return_centroid(traff:Traffic) -> None:
#     ids = list(f.flight_id for f in traff)
#     index = ids[
#         squareform(
#             pdist(
#                 np.stack(
#                     list(f.data[["CAS", "altitude"]].astype(float).values.ravel() for f in traff)
#                 )
#             )
#         )
#         .mean(axis=1)
#         .argmin()
#     ]  # Not sure about this centroid search, "copied" and adapted from traffic's git repo
#     return index


def get_mean_profile(traffic: Traffic) -> tuple[pd.DataFrame, Flight]:
    """
    Improved version using altair
    """
    print("max_flight_diff_1 : ", max_alt_diff(traffic))
    print(
        traffic.data.to_csv(
            f"traffic_temp{traffic.data['altitude'].iloc[0]}.csv"
        )
    )
    # traffic = traffic.filter().eval(desc="flitering")
    traffic = traffic.resample("2s").eval(desc="")

    # traffic = traffic.filter().eval(desc="flitering")
    print("max_flight_diff_2 : ", max_alt_diff(traffic))
    max_diff_alt = max_alt_diff_per_traff(traffic)
    print(
        "number of fucked traff: ",
        len([elem for elem in max_diff_alt if elem > 2000]),
    )

    traffic = traffic.drop(columns=["timedelta"])
    traffic = compute_time_delta(traffic)
    if "vertical_rate" in traffic:
        traffic = traffic.drop("vertical_rate")

    traffic = compute_vertical_rate(traffic=traffic)

    cas = aero.tas2cas(traffic.data["groundspeed"], h=traffic.data["altitude"])
    traffic = traffic.assign(CAS=cas)

    min_f_len: int = min(len(f) for f in traffic)

    t_temp = traffic.query(f"timedelta < {min_f_len * 2}")
    print("max: ", min(len(f) for f in t_temp))
    print("min length traff: ", min_f_len)
    id_centroid = return_centroid(t_temp)
    print(
        "flight speed over 300: ",
        len([1 for f in traffic if (f.data["CAS"] > 300).any()]),
    )

    # removing flights with impossible altitudes (habborant )
    print(
        "Flights with one alt under 2000",
        sum((f.data["altitude"] < 1500).any() for f in traffic),
        f" {traffic[0].typecode}",
    )

    def f_good(f: Flight, threshold: int = 1500) -> Flight:
        if (f.data["altitude"] < threshold).any() == 0:  # and (
            #     f.data[f.data["altitude"] < 20000]["altitude"].diff().max() < 2000
            # ):
            return f
        return None

    traffic = traffic.pipe(f_good).eval(desc="removing outliers")
    id_centroid = return_centroid(t_temp)

    alts = []
    x_list = []
    v_rates = []
    for flight in traffic:
        alts.append(flight.data["altitude"].to_list())
        x_list.append(flight.data["CAS"].to_list())
        v_rates.append(flight.data["vertical_rate"].to_list())

    mean_profile = pd.DataFrame()

    max_len = max(len(arr) for arr in alts)

    array_alts = np.array(
        [
            np.pad(
                np.array(arr),
                (0, max_len - len(arr)),
                constant_values=np.nan,
            )
            for arr in alts
        ]
    )

    nan_counts = np.isnan(array_alts).sum(axis=0)

    print(nan_counts)

    threshold = 100

    mask = nan_counts <= len(traffic) - threshold  # we want to have at least threshold vals, before computing the mean

    array_alts = array_alts[:, mask]

    mean_profile["altitude"] = np.nanmean(
        array_alts,
        axis=0,
    )

    cas_array = np.array(
        [
            np.pad(
                np.array(arr),
                (0, max_len - len(arr)),
                constant_values=np.nan,
            )
            for arr in x_list
        ]
    )

    cas_array = cas_array[:, mask]

    mean_profile["CAS"] = np.nanmean(
        cas_array,
        axis=0,
    )

    vertical_rate = np.array(
        [
            np.pad(
                np.array(arr),
                (0, max_len - len(arr)),
                constant_values=np.nan,
            )
            for arr in v_rates
        ]
    )

    mean_profile["vertical_rate"] = np.nanmean(
        vertical_rate[:, mask],
        axis=0,
    )

    mean_profile = mean_profile[mean_profile["altitude"] < 20000]
    mean_profile = mean_profile[mean_profile["CAS"] < 300]
    mean_profile = mean_profile  # .dropna()
    print(mean_profile["altitude"].diff().max())
    print(mean_profile["altitude"].diff().tolist())
    print(
        traffic[id_centroid]
        .data[traffic[id_centroid].data["altitude"] > 14900][
            ["CAS", "altitude", "groundspeed"]
        ]
        .head(20)
    )
    return mean_profile, traffic[id_centroid]


def mean_profile_per_typecodes(
    traff: Traffic, label: str, typecodes: list[str], n_flights: int = 1000
) -> pd.DataFrame:
    mean_prof_total = pd.DataFrame(
        columns=["typecode", "generated", "CAS", "altitude"]
    )
    for typecode in typecodes:
        # print(len(traff),print(traff.data['typecode']))
        # print([f.typecode for f in traff.sample(10)])
        # print()
        t = traff.query(f"typecode == '{typecode}'")
        t = t.sample(min(n_flights, len(t)))
        mean_prof, centroid = get_mean_profile(t)
        mean_prof["generated"] = label
        mean_prof["typecode"] = typecode

        centroid_data = centroid.data[["altitude", "CAS", "vertical_rate"]]
        centroid_data["generated"] = label + "_centroid"
        centroid_data["typecode"] = typecode
        centroid_data = centroid_data[centroid_data["altitude"] < 20000]

        mean_prof_total = pd.concat(
            [mean_prof_total, mean_prof, centroid_data], axis=0
        )
    return mean_prof_total


def profile_gen_and_not_gen(
    generator: ONNX_Generator,
    traffic_og: Traffic,
    typecodes: list[str],
    n_flights: int = 1000,
    batch_size: int = 500,
) -> pd.DataFrame:
    ##generated dataframe
    traffics = generator.generate_n_flight_per_labels(
        typecodes, n_flights, batch_size=batch_size
    )
    tc = [
        typecodes[i // (n_flights * 200)]
        for i in range(n_flights * len(typecodes) * 200)
    ]
    icao24 = [i // 200 for i in range(n_flights * len(typecodes) * 200)]
    callsign = icao24

    f_traf: Traffic = sum(traffics)
    f_traf = f_traf.assign(typecode=tc)
    f_traf = f_traf.assign(icao24=icao24)
    f_traf = f_traf.assign(callsign=callsign)
    f_traf = f_traf.assign_id().eval(desc="assigning ids", max_workers=4)

    mean_prof = mean_profile_per_typecodes(
        f_traf, label="generated", typecodes=typecodes, n_flights=n_flights
    )

    mean_prof_og = mean_profile_per_typecodes(
        traffic_og, label="flown", typecodes=typecodes, n_flights=n_flights
    )

    return pd.concat([mean_prof, mean_prof_og], axis=0)


def plot_mean_profile(mean_profile: pd.DataFrame, path: str) -> None:
    mean_profile["typecode"] = mean_profile["typecode"].astype(str)
    mean_profile["generated"] = mean_profile["generated"].astype(str)

    # Create the Altair chart
    chart = (
        alt.Chart(mean_profile)
        .mark_line()
        .encode(
            x=alt.X("CAS:Q", title="CAS"),
            y=alt.Y("altitude:Q", title="Altitude"),
            color=alt.Color("generated:N", title="Generated Label"),
            strokeDash=alt.condition(
                "datum.generated == 'generated' || datum.generated == 'flown'",
                alt.value([1, 0]),  # solid line
                alt.value([4, 4]),  # dotted line
            ),
            opacity=alt.condition(
                "datum.generated == 'generated' || datum.generated == 'flown'",
                alt.value(1.0),  # full opacity for "generated" or "flown"
                alt.value(0.4),  # lower opacity for others
            ),
        )
        .facet(row=alt.Row("typecode:N", title="Typecode"), columns=2)
    )

    chart.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="My script with arguments")

    parser.add_argument(
        "--onnx_dir",
        type=str,
        default="",
        help="Path of the directory for the onnx file to load model",
    )
    parser.add_argument(
        "--typecodes",
        nargs="+",
        type=str,
        default=[],
        help="typecodes for which to consider the altitude profiles",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to the dataset to compute the vamp",
    )

    parser.add_argument(
        "--plot_path",
        type=str,
        default="",
        help="Path to the plot",
    )

    args = parser.parse_args()

    print(args.typecodes)

    data_cleaner = Data_cleaner(
        no_data=True
    )  # saves scalers, and scales and modifies data, allows to interpret the model outputs
    data_cleaner.load_scalers(args.onnx_dir + "/scalers.pkl")
    cvae_onnx = CVAE_ONNX(args.onnx_dir)  # the CVAE
    onnx_gen = ONNX_Generator(
        cvae_onnx, data_cleaner
    )  # used to gen from the CVAE
    traff = Traffic.from_file(args.data)
    if "typecode" not in traff.data.columns:
        traff = traff.aircraft_data()
    mean_prof = profile_gen_and_not_gen(
        generator=onnx_gen,
        traffic_og=traff,
        typecodes=args.typecodes,
        n_flights=2000,
    )
    mean_prof.to_parquet(args.plot_path.split(".")[0] + "_mean_profiles.pkl")
    mean_prof.to_csv(args.plot_path.split(".")[0] + "_mean_profiles.csv")
    plot_mean_profile(mean_prof, args.plot_path)


if __name__ == "__main__":
    main()
