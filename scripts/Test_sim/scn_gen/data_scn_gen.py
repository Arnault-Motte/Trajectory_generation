import sys  # noqa: I001
import os

from tqdm import tqdm

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.generation.data_process import Data_cleaner
from data_orly.src.generation.generation import Generator
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.test_display import plot_traffic
from data_orly.src.simulation import Simulator
from traffic.core import Traffic


def main() -> None:
    parser = argparse.ArgumentParser(description="My script with arguments")

    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path of the data",
    )
    parser.add_argument(
        "--data_s",
        type=str,
        default="",
        help="Path of the sampled data",
    )
    parser.add_argument(
        "--scn_file",
        type=str,
        default="",
        help="Name of the scene file",
    )

    parser.add_argument(
        "--nf",
        type=int,
        default=3000,
        help="Number of flights to be sampled for each label",
    )
    args = parser.parse_args()

    data_clean = Data_cleaner(file_name=args.data, airplane_types_num=10)
    data_clean.return_labels()  # computing the labels

    labels = data_clean.get_typecodes_labels()
    total_f_traff = []
    for label in tqdm(labels):
        d = data_clean.basic_traffic_data.query(f'typecode == "{label}"')
        if (
            len(d) > args.nf
        ):  # sampling a smaller number of traejctory (else it's hell)
            d = d.sample(args.nf)
        total_f_traff.append(d)

    t_f: Traffic = sum(total_f_traff)
    t_f.to_pickle(args.data_s)

    s = Simulator(t_f)
    s.scenario_create(
        args.scn_file,
        log_f_name=args.scn_file.split(".") + ".txt",
        load_file="navpoint_temp.pkl",
        typecode="0",
    )

    return


if __name__ == "__main__":
    main()
