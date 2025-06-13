import sys  # noqa: I001
import os


current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.generation.data_process import (
    Data_cleaner,
    return_traff_per_typecode,
    filter_missing_values,
    compute_time_delta,
)

from traffic.core import Traffic


def main() -> None:
    """
    Adds time deltas from timestamp and filters the Nan values of the entered traffic
    """
    praser = argparse.ArgumentParser(
        description="Cleaning the data of the entered traffic"
    )

    praser.add_argument(
        "--data", type=str, default="", help="Path of the data to clean"
    )

    praser.add_argument(
        "--data_clean", type=str, default="", help="Path of the data cleaned"
    )

    columns = [
        "track",
        "groundspeed",
        "timedelta",
        "altitude",
        "latitude",
        "longitude",
    ]

    args = praser.parse_args()

    og_t = Traffic.from_file(args.data)

    t = compute_time_delta(og_t)  # computing time deltas

    t = filter_missing_values(  # filtering out the missing values
        t, columns, max_wrokers=1
    )

    t.to_pickle(args.data_clean)
    print("|-- Results --|")
    print("Original traffic size: ", len(og_t))
    print("Filtered traffic size: ", len(t))
    print("%% of deleted: ", 1- len(t)/len(og_t))
    print("Nan values remaining: ",t.data[columns].isna().any(axis=1).sum())
    print('--------------')

    return

if __name__ == "__main__":
    main()
