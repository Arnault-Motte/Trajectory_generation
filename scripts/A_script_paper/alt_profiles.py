import sys  # noqa: I001
import os

from tqdm import tqdm

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.generation.data_process import Data_cleaner, return_labels
from data_orly.src.generation.generation import Generator, ONNX_Generator
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_ONNX import CVAE_ONNX
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.test_display import (
    plot_traffic,
    vertical_rate_profile_2,
)
from data_orly.src.simulation import Simulator
from traffic.core import Traffic


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
    parser.add_argument(
        "--x_col",
        type=str,
        default="timedelta",
        help="Which columns you wish as x_col",
    )

    args = parser.parse_args()

    print(args.typecodes)

    if args.data == "":
        data_cleaner = Data_cleaner(
            no_data=True
        )  # saves scalers, and scales and modifies data, allows to interpret the model outputs
        data_cleaner.load_scalers(args.onnx_dir + "/scalers.pkl")
        cvae_onnx = CVAE_ONNX(args.onnx_dir)  # the CVAE
        onnx_gen = ONNX_Generator(
            cvae_onnx, data_cleaner
        )  # used to gen from the CVAE

        traffs = onnx_gen.generate_n_flight_per_labels(args.typecodes, 10,batch_size=1)

        for i, typecode in tqdm(enumerate(args.typecodes)):
            gen_data = traffs[i]
            vertical_rate_profile_2(
                gen_data,
                args.plot_path.split(".")[0] + f"_{typecode}.png",
                distance=False,
                x_col=args.x_col,
            )
    elif args.onnx_dir == "":
        traff = Traffic.from_file(args.data)
        if "typecode" not in traff.data.columns:
            traff = traff.aircraft_data()
        for i, typecode in tqdm(enumerate(args.typecodes)):
            data_hist = traff.query(f'typecode == "{typecode}"').sample(
                min(10, len(traff))
            )
            vertical_rate_profile_2(
                data_hist,
                args.plot_path.split(".")[0] + f"_{typecode}.png",
                distance=False,
                x_col=args.x_col,
            )
    else:
        raise ValueError(
            "Choose between generation and using historical data, you can't do both"
        )


if __name__ == "__main__":
    main()
