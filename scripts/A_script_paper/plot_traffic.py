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
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_ONNX import CVAE_ONNX
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
from data_orly.src.generation.test_display import plot_distribution_typecode
from traffic.core import Traffic
from data_orly.src.generation.generation import ONNX_Generator
from tqdm import tqdm

from data_orly.src.generation.test_display import plot_traffic


def main() -> None:
    parser = argparse.ArgumentParser(description="Training a CVAE")

    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to acess the data",
    )

    parser.add_argument(
        "--onnx_dir",
        type=str,
        default="",
        help="Path to acess the data",
    )

    parser.add_argument(
        "--plot_path",
        type=str,
        default="",
        help="Path to acess the data",
    )

    parser.add_argument(
        "--typecodes",
        nargs="+",
        type=str,
        default=[],
        help="typecodes for which to consider the altitude profiles",
    )

    args = parser.parse_args()

    if args.data == "":
        data_cleaner = Data_cleaner(
            no_data=True
        )  # saves scalers, and scales and modifies data, allows to interpret the model outputs
        data_cleaner.load_scalers(args.onnx_dir + "/scalers.pkl")
        cvae_onnx = CVAE_ONNX(args.onnx_dir)  # the CVAE
        onnx_gen = ONNX_Generator(
            cvae_onnx, data_cleaner
        )  # used to gen from the CVAE

        traffs = onnx_gen.generate_n_flight_per_labels(args.typecodes, 2000) #to modify to not have several time the same _id

        f_traf: Traffic = sum(traffs)
        
        plot_traffic(f_traf,args.plot_path)
    else:
        traff = Traffic.from_file(args.data)
        plot_traffic(traff,args.plot_path)


if __name__ == "__main__":
    main()
