import sys  # noqa: I001
import os

from tqdm import tqdm

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch
from src.data_process import Data_cleaner, return_labels
from src.generation import Generator, ONNX_Generator
from src.models.CVAE_ONNX import CVAE_ONNX
from src.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from src.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from src.simulation import Simulator
from src.test_display import (
    plot_traffic,
    vertical_rate_profile_2,
)
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

    parser.add_argument(
        "--vamp",
        type=int,
        default=0,
        help="The number of the vamprior to generate for",
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


        for i, typecode in tqdm(enumerate(args.typecodes)):

            gen_data = onnx_gen.generate_flight_for_label_vamp(typecode,vamp=args.vamp,n_points=100,batch_size=100)
            vertical_rate_profile_2(
                gen_data,
                args.plot_path.split(".")[0] + f"_time_{typecode}_{args.vamp}.png",
                x_col="timedelta",
            )
            vertical_rate_profile_2(
                gen_data,
                args.plot_path.split(".")[0] + f"_speed_{typecode}_{args.vamp}.png",
                x_col="CAS",
            )
            plot_traffic(gen_data,plot_path=args.plot_path.split(".")[0] + f"_traffic_time_{typecode}_{args.vamp}.png")
    elif args.onnx_dir == "":
        pass
    else:
        raise ValueError(
            "Choose between generation and using historical data, you can't do both"
        )


if __name__ == "__main__":
    main()
