import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

from data_orly.src.simulation import Simulator
from data_orly.src.generation.test_display import plot_traffic
from data_orly.src.generation.evaluation import compute_distances
from data_orly.src.generation.data_process import Data_cleaner
from traffic.core import Traffic
import pickle
import argparse


def main() -> None:
    file_path_og = (
        "/data/data/arnault/data/new_data/TO_LFPO_7_B_train_no_nan.pkl"
    )
    file_path_reb = ""

    t = Traffic.from_file(file_path_og)
    t = t.aircraft_data()
    print(len(t))
    # icao = str(
    #     t.data[t.data["timedelta"] == t.data["timedelta"].max()]["icao24"].iloc[
    #         0
    #     ]
    # )
    # print(icao)
    print((t.data["timedelta"].iloc[199::200] > 1800).sum())
        # Step 1: Get the indices of interest
    indices = t.data.iloc[199::200].index

    # Step 2: Create the boolean mask on the sliced data
    mask = t.data.loc[indices, "timedelta"] > 1800

    # Step 3: Apply the mask to the DataFrame
    selected_rows = t.data.loc[indices[mask]]["flight_id"]

    print(selected_rows.head(5))
    print((t.data["timedelta"].iloc[199::200] > 1200).sum())
    indices = t.data.iloc[199::200].index

    # Step 2: Create the boolean mask on the sliced data
    mask = t.data.loc[indices, "timedelta"] > 1200

    # Step 3: Apply the mask to the DataFrame
    selected_rows = t.data.loc[indices[mask]]["flight_id"]
    
    print(selected_rows.head(5))
    print((t.data["timedelta"].iloc[199::200] > 600).sum())

    indices = t.data.iloc[199::200].index

    # Step 2: Create the boolean mask on the sliced data
    mask = (t.data.loc[indices, "timedelta"] > 600) & (t.data.loc[indices, "timedelta"] < 1200)

    # Step 3: Apply the mask to the DataFrame
    selected_rows = t.data.loc[indices[mask]]["flight_id"]

    print(selected_rows.head(5))

    plot_traffic(t.sample(200),plot_path='test_traff_thres.png')
    # print(t[icao])
    # print(t[icao].resample(150).data.head())
    # print(t[icao].data["groundspeed"].mean())
    # print(t[icao].data["typecode"].iloc[0])

    # print(t.data[dist_airp][0::200])


main()
