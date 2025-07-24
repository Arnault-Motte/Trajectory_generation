import sys  # noqa: I001
import os

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
from src.evaluation import compute_e_dist
from src.generation import Generator, ONNX_Generator
from src.models.CVAE_ONNX import CVAE_ONNX
from src.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from src.models.VAE_ONNX import VAE_ONNX
from src.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from src.simulation import Simulator
from src.test_display import (
    plot_traffic,
    vertical_rate_profile_2,
)
from traffic.core import Flight, Traffic


def add_dist_to_file(
    path: str,
    loss: float,
    kl_loss: float,
    mse: float,
    typecode: str,
    model: VAE_ONNX | CVAE_ONNX,
) -> None:
    first_line = "Model_type, typecode, typecodes, total loss, kl_loss, MSE \n"
    line = f"{model.__class__.__name__}, {typecode}, {loss}, {kl_loss}, {mse}"

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


def traff_data_to_tensors(
    traff: Traffic, typecode: str, data_clean: Data_cleaner, n_f: int
) -> tuple[torch.Tensor, torch.Tensor]:
    t_typecode = traff.query(f"typecode == '{typecode}'")
    t_typecode = t_typecode.sample(min(len(t_typecode), n_f))
    data = data_clean.clean_data_specific(t_typecode, fit_scale=False)
    mask = (data > 1) | (
        data < -1
    )  # delete trajectories that are out of the scope of the min max scaler
    rows_with_condition = np.any(mask, axis=(1, 2))
    count = np.sum(rows_with_condition)
    print("wierd values : ", count)
    data = data[~rows_with_condition]
    test_data = torch.tensor(data, dtype=torch.float32)

    typecodes = data_clean.one_hot.transform(np.full((len(t_typecode), 1), typecode))
    typecodes_tensor = torch.Tensor(typecodes)
    return test_data, typecodes_tensor


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
        default=[],
        help="Paths to the ONNX VAEs",
    )

    parser.add_argument(
        "--loss_path",
        type=str,
        default="",
        help="Path to the loss file",
    )

    parser.add_argument(
        "--n_f",
        type=int,
        default=3000,
        help="Number of flight to generate",
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
        # generating with CVAE
        cvae_onnx = CVAE_ONNX(args.CVAE_ONNX)  # the CVAE

        for typecode in args.typecodes:
            test_data, typecodes_tensor = traff_data_to_tensors(
                traff, typecode, data_cleaner, args.n_f
            )
            loss, kl_div, total_mse = cvae_onnx.compute_loss(
                test_data, typecodes_tensor
            )
            add_dist_to_file(
                args.loss_path, loss, kl_div, total_mse, typecode, cvae_onnx
            )

    if args.VAEs_ONNX != []:
        for vae_path, typecode in tqdm(
            zip(args.VAEs_ONNX, args.typecodes), desc="E-distances for the VAEs"
        ):
            data_cleaner = Data_cleaner(
                no_data=True
            )  # saves scalers, and scales and modifies data, allows to interpret the model outputs
            data_cleaner.load_scalers(vae_path + "/scalers.pkl")
            vae_onnx = VAE_ONNX(vae_path)

            test_data, _ = traff_data_to_tensors(
                traff, typecode, data_cleaner, args.n_f
            )

            loss, kl_div, total_mse = vae_onnx.compute_loss(test_data)

            add_dist_to_file(
                args.loss_path, loss, kl_div, total_mse, typecode, vae_onnx
            )


if __name__ == "__main__":
    main()
