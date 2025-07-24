import sys  # noqa: I001
import os

from tqdm import tqdm

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch

from src.data_process import (
    Data_cleaner,
    return_labels,
    compute_time_delta,
    compute_vertical_rate,
)
from src.generation import Generator, ONNX_Generator
from src.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from src.models.CVAE_ONNX import CVAE_ONNX
from src.models.VAE_ONNX import VAE_ONNX

from src.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from src.test_display import (
    plot_traffic,
    vertical_rate_profile_2,
)
from src.data_process import compute_vertical_rate
from src.simulation import Simulator
from traffic.core import Traffic, Flight
from pitot import aero
import altair as alt
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.evaluation import compute_e_dist

import statistics


def add_dist_to_file(path:str,dist:float,model:VAE_ONNX|CVAE_ONNX,typecode:str)->None:
    first_line = "Model_type, typecode, typecodes, e_dist \n"
    line = f"{model.__class__.__name__}, {typecode}, {dist}"

    add_line_to_file(path, line, first_line)


def add_line_to_file(
    file_path: str, line: str, first_line: str = "This is the first line.\n"
) -> None:
    # Check if the file is empty
    is_empty = not os.path.exists(file_path) or os.stat(file_path).st_size == 0

    # Open the file in append mode
    with open(file_path, "a") as file:
        if is_empty:
            # Write the special first line if the file is empty
            file.write(first_line)
        # Add the new line
        file.write(line + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="My script with arguments")

    parser.add_argument(
        "--typecodes",
        nargs="+",
        type=str,
        default=[],
        help="typecodes for which to consider the altitude profiles",
    )
    parser.add_argument(
        "--data_og",
        type=str,
        default="",
        help="Path to the dataset to compute the vamp",
    )

    parser.add_argument(
        "--CVAE_ONNX",
        type=str,
        default="",
        help="Path to the ONNX CVAE",
    )
    parser.add_argument(
        "--VAEs_ONNX",
        nargs="+",
        type=str,
        default="",
        help="Paths to the ONNX VAEs",
    )

    parser.add_argument(
        "--dist_path",
        type=str,
        default="",
        help="Path to the plot",
    )

    parser.add_argument(
        "--n_f",
        type=int,
        default=3000,
        help="Number of flight to generate per trial",
    )

    parser.add_argument(
        "--cond_pseudo",
        type=int,
        default=0,
        help="1 if the CVAE uses conditioned pseudo inputs",
    )


    args = parser.parse_args()

    print(args.typecodes)
    # data cleaner def

    traff = Traffic.from_file(args.data_og)
    if "typecode" not in traff.data.columns:
        traff = traff.aircraft_data()
    if args.CVAE_ONNX != "":
        data_cleaner = Data_cleaner(
            no_data=True
        )  # saves scalers, and scales and modifies data, allows to interpret the model outputs
        data_cleaner.load_scalers(args.CVAE_ONNX + "/scalers.pkl")
        og_scaler = data_cleaner.scaler #used to rescal the CVAE and VAE data to the same range, otherwise we cannot compare the two e_dist values

        # generating with CVAE
        cvae_onnx = CVAE_ONNX(args.CVAE_ONNX,condition_pseudo=args.cond_pseudo)  # the CVAE
        onnx_gen = ONNX_Generator(
            cvae_onnx, data_cleaner
        )  # used to gen from the CVAE

        for typecode in tqdm(args.typecodes,desc="E-distances for the CVAE"):
            e_dist = compute_e_dist(
                onnx_gen,
                traff,
                number_of_trial=100,
                n_t=args.n_f,
                label=typecode,
                scaler=og_scaler,
            )
            add_dist_to_file(args.dist_path,e_dist,cvae_onnx,typecode)

    if args.VAEs_ONNX != []:
        for vae_path,typecode in tqdm(zip(args.VAEs_ONNX,args.typecodes),desc="E-distances for the VAEs"):
            data_cleaner = Data_cleaner(
                no_data=True
            )  # saves scalers, and scales and modifies data, allows to interpret the model outputs
            data_cleaner.load_scalers(vae_path + "/scalers.pkl")
            vae_onnx = VAE_ONNX(vae_path)
            onnx_gen = ONNX_Generator(
                vae_onnx, data_cleaner
            )  # used to gen from the CVAE

            e_dist = compute_e_dist(
                onnx_gen,
                traff,
                number_of_trial=100,
                n_t=args.n_f,
                label="",#Gives the info to gen from a VAE and not a CVAE
                scaler=og_scaler,
            )
            add_dist_to_file(args.dist_path,e_dist,vae_onnx,typecode)



if __name__ == "__main__":
    main()
