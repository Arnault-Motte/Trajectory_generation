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
from data_orly.src.generation.models.VAE_ONNX import VAE_ONNX
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.test_display import (
    plot_traffic,
    vertical_rate_profile_2,
)
from data_orly.src.simulation import Simulator
from traffic.core import Traffic
from data_orly.src.generation.data_process import compute_vertical_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="My script with arguments")

    parser.add_argument(
        "--onnx_dir",
        type=str,
        default="",
        help="Path of the directory for the onnx file to load model",
    )

    parser.add_argument(
        "--plot_dir",
        type=str,
        default="",
        help="Path to the plot",
    )


    args = parser.parse_args()

    data_cleaner = Data_cleaner(
            no_data=True
    )  # saves scalers, and scales and modifies data, allows to interpret the model outputs
    data_cleaner.load_scalers(args.onnx_dir + "/scalers.pkl")
    vae_onnx = VAE_ONNX(args.onnx_dir)  # the CVAE
    onnx_gen = ONNX_Generator(
            vae_onnx, data_cleaner
    )  # used to gen from the CVAE

    traff = onnx_gen.generate_n_flight(100,10)
    plot_path = args.plot_dir
    if plot_path == "":
            plot_path = "data_orly/figures/paper/" + args.onnx_dir.split('/')[-1]
            os.makedirs(plot_path, exist_ok=True)
        
    plot_path = plot_path + "/traffic.png"
            
    plot_traffic(traff,plot_path)
       


if __name__ == "__main__":
    main()
