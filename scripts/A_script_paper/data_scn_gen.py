import sys  # noqa: I001
import os

from tqdm import tqdm

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch
from src.data_process import Data_cleaner
from src.generation import Generator
from src.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from src.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from src.simulation import Simulator
from src.test_display import plot_traffic
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

    parser.add_argument(
        "--typecodes",
        type=str,
        default=[],
        nargs="+",
        help="typecode of the data to select",
    )

    parser.add_argument(
        "--navpoint_path",
        type=str,
        default="",
        help="if the navpoints have already been created you can reuse them by passing their saved csv path",
    )
    args = parser.parse_args()


    


    # data_clean.return_labels()  # computing the labels
    t_f = Traffic.from_file(args.data)
    t_f = t_f.aircraft_data()
    if len(args.typecodes) != 0:
        t_f = t_f.query(f'typecode in {args.typecodes}') 
    
    labels = args.typecodes
    total_f_traff = []
    for label in tqdm(labels):
        d = t_f.query(f'typecode == "{label}"')
        if (
            len(d) > args.nf
        ):  # sampling a smaller number of traejctory (else it's hell)
            d = d.sample(args.nf)
        total_f_traff.append(d)

    t_f: Traffic = sum(total_f_traff)
    if args.data_s != "":
        t_f.to_pickle(args.data_s)

    s = Simulator(t_f)
    s.scenario_create(
        args.scn_file,
        log_f_name=args.scn_file.split(".")[0] + ".txt",
        load_file=args.navpoint_path,
        typecode="0",
    )

    return


if __name__ == "__main__":
    main()
