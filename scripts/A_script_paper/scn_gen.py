import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.generation.data_process import Data_cleaner
from data_orly.src.generation.generation import Generator, ONNX_Generator
from data_orly.src.generation.models.CVAE_ONNX import CVAE_ONNX
from data_orly.src.generation.models.VAE_ONNX import VAE_ONNX
from data_orly.src.generation.test_display import plot_traffic
from data_orly.src.simulation import Simulator
from traffic.core import Traffic


def scenario_create(t: Traffic, typecode: str, scn_file: str, model) -> None:
    s = Simulator(t)
    folder = scn_file + "/" if scn_file != "" else ""
    name_path = folder + str(model.__class__.__name__) + "_" + str(typecode)
    traff_file = name_path + ".pkl"
    t.to_pickle(traff_file)
    path = name_path + ".scn"
    second_path = ""
    s.scenario_create(path, name_path, load_file=second_path, typecode=typecode)


def main() -> None:
    parser = argparse.ArgumentParser(description="My script with arguments")

    parser.add_argument(
        "--scn_file",
        type=str,
        default="",
        help="path to the created_scn_files",
    )

    parser.add_argument(
        "--cvae_onnx",
        type=str,
        default="",
        help="path to the cvae onnx",
    )

    parser.add_argument(
        "--vae_onnx",
        nargs="+",
        type=str,
        default="",
        help="path to the vaes onnx",
    )

    parser.add_argument(
        "--n_f",
        type=int,
        default=2000,
        help="Number of flights to be generated",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Number of flights to be generated",
    )

    parser.add_argument(
        "--typecodes",
        nargs="+",
        type=str,
        default=[],
        help="typecodes of interest",
    )

    parser.add_argument(
        "--cond_pseudo",
        type=int,
        default=0,
        help="set to true if the cvae uses conditioned pseudo inputs",
    )

    args = parser.parse_args()

    if args.vae_onnx != []:
        for model_path, typecode in zip(args.vae_onnx, args.typecodes):
            model = VAE_ONNX(model_path)
            data_cleaner = Data_cleaner(no_data=True)
            data_cleaner.load_scalers(model_path + "/scalers.pkl")
            onnx_gen = ONNX_Generator(
                model, data_cleaner
            )  # used to gen from the VAE
            traff = onnx_gen.generate_n_flight(
                args.n_f, batch_size=args.batch_size
            )
            if traff.flight_ids is None:
                traff = traff.assign_id().eval()
            scenario_create(traff, typecode, args.scn_file, model)

    if args.cvae_onnx != "":
        model = CVAE_ONNX(args.cvae_onnx,condition_pseudo=args.cond_pseudo)
        data_cleaner = Data_cleaner(no_data=True)
        data_cleaner.load_scalers(args.cvae_onnx + "/scalers.pkl")
        onnx_gen = ONNX_Generator(
            model, data_cleaner
        )  # used to gen from the CVAE
        traffs = onnx_gen.generate_n_flight_per_labels(
            args.typecodes, n_points=args.n_f, batch_size=args.batch_size
        )
        for traff, typecode in zip(traffs, args.typecodes):
            if traff.flight_ids is None:
                traff = traff.assign_id().eval()
            scenario_create(traff, typecode, args.scn_file, model)

    print("end")


if __name__ == "__main__":
    main()
