import os
import pickle
import sys  # noqa: I001

from tqdm import tqdm

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse
import statistics

import altair as alt
import numpy as np
import pandas as pd
import torch
from pitot import aero
from scipy.spatial.distance import pdist, squareform
from src.data_process import (
    Data_cleaner,
    compute_time_delta,
    compute_vertical_rate,
    return_labels,
)
from src.generation import Generator, ONNX_Generator
from src.models.CVAE_ONNX import CVAE_ONNX
from src.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from src.models.VAE_ONNX import VAE_ONNX
from src.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from src.simulation import Simulator
from src.test_display import (
    latent_space_CVAE,
    latent_space_VAE,
    plot_traffic,
    vertical_rate_profile_2,
)
from traffic.core import Flight, Traffic


def main() -> None:
    parser = argparse.ArgumentParser(description="My script with arguments")

    parser.add_argument(
        "--onnx_cvae_dir",
        type=str,
        default="",
        help="Path of the directory for the onnx file to load the cvae model",
    )

    parser.add_argument(
        "--onnx_vae_dirs",
        nargs="+",
        type=str,
        default=[],
        help="Paths of the directories for the vaes onnx file to load model",
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
        "--spec",
        type=int,
        default=0,
        help="true if you want to use the model spec as label",
    )

    parser.add_argument(
        "--cond_pseudo",
        type=int,
        default=0,
        help="true if the pseudo inputs of the vae are cond",
    )

    parser.add_argument(
        "--profile_path",
        type=str,
        default="",
        help="path to the profile .csv",
    )

    args = parser.parse_args()
    
    traff = Traffic.from_file(args.data)
    if "typecode" not in traff.data.columns:
        traff = traff.aircraft_data()
    if args.onnx_cvae_dir !="":
        model = CVAE_ONNX(args.onnx_cvae_dir,condition_pseudo=args.cond_pseudo)
        data_clean = Data_cleaner(no_data=True)
        data_clean.load_scalers(args.onnx_cvae_dir + "/scalers.pkl")

        latent_space_CVAE(
            model=model,
            data_cleaner=data_clean,
            typecodes=args.typecodes,
            plt_path=args.plot_path,
            num_point=500,
            traff=traff,
        )

    if args.onnx_vae_dirs != []:
        latent_space_VAE(
            typecodes=args.typecodes,
            num_point=500,
            models=args.onnx_vae_dirs,
            plt_path=args.plot_path.split(".")[0] + "vae.png",
            traff=traff,
        )


if __name__ == "__main__":
    main()
