import sys  # noqa: I001
import os


CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

from data_orly.src.generation.evaluation import Evaluator
from data_orly.src.generation.generation import Generator
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.data_process import Data_cleaner
from traffic.core import Traffic
import torch
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="files for the test")

    parser.add_argument("--model", type=str, default="", help="model file path")
    parser.add_argument(
        "--save_file",
        type=str,
        default="",
        help="File to save the results",
    )
    parser.add_argument(
        "--typecode",
        type=str,
        default="",
        help="Typecode to be generated if cond",
    )
    parser.add_argument(
        "--cond",
        type=int,
        default=0,
        help="Is the loaded model a conditional model",
    )
    parser.add_argument(
        "--cuda", type=int, default=0, help="Number of the gpu to run the code"
    )
    parser.add_argument(
        "--typecodes",
        type=str,
        nargs="+",
        default="",
        help="Typecodes of the dataset to be considered",
    )
    parser.add_argument(
        "--traff_data",
        type=str,
        default="",
        help="Path to the traffic file used to train the models",
    )

    parser.add_argument(
        "--vertical_rate",
        type=int,
        default=0,
        help="True if you want to use the vertical rate as a column",
    )

    parser.add_argument(
        "--n_trial",
        type=int,
        default=100,
        help="Number of trial to compute e_dist",
    )
    parser.add_argument(
        "--n_gen",
        type=int,
        default=3000,
        help="Number of trajectory to generate",
    )

    args = parser.parse_args()
    columns = ["track", "groundspeed", "timedelta"]
    columns += ["altitude"] if not args.vertical_rate else ["vertical_rate"]
    print(columns)
    print(args.typecodes)

    data_cleaner = Data_cleaner(
        args.traff_data,
        columns=columns,
        chosen_typecodes=args.typecodes,
    )
    print("Data len : ", len(data_cleaner.basic_traffic_data))
    data = data_cleaner.clean_data()
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )  # noqa: F405
    print(device)
    seq_len = 200
    in_channels = 4
    output_channels = 64
    latent_dim = 64
    pooling_factor = 10
    stride = 1
    number_of_block = 4
    kernel_size = 16
    dilatation = 2
    dropout = 0.2
    pseudo_input_num = 800  # *1.5
    patience = 30
    min_delta = -100
    labels_latent = 16

    labels = data_cleaner.return_labels()
    labels_dim = labels.shape[1]

    cond = bool(args.cond)
    if not cond:
        model = VAE_TCN_Vamp(
            in_channels,
            output_channels,
            latent_dim,
            kernel_size,
            stride,
            dilatation,
            dropout,
            number_of_block,
            pooling_factor,
            pooling_factor,
            seq_len,
            pseudo_input_num=pseudo_input_num,
            early_stopping=True,
            patience=patience,
            min_delta=min_delta,
            init_std=0.3,
        ).to(device)
    else:
        model = CVAE_TCN_Vamp(
            in_channels,
            output_channels,
            latent_dim,
            kernel_size,
            stride,
            dilatation,
            dropout,
            number_of_block,
            pooling_factor,
            pooling_factor,
            label_dim=labels_dim,
            label_latent=labels_latent,
            seq_len=seq_len,
            pseudo_input_num=pseudo_input_num,
            early_stopping=True,
            patience=patience,
            min_delta=min_delta,
            temp_save="best_model_1000.pth",
            conditioned_prior=True,
            num_worker=6,
        ).to(device)

    label = args.typecode  # label is defaulting to "" if it is not defined
    model.load_model(args.model)

    gen = Generator(model, data_cleaner)
    ev = Evaluator(gen, data_cleaner.basic_traffic_data)

    dist = ev.compute_e_dist(
        label=label,
        number_of_trial=args.n_trial,
        n_t=args.n_gen,
        scaler=data_cleaner.scaler,
    )

    first_line = "Model_Path, Model_type, typecode, typecodes, e_dist \n"
    line = f"{args.model}, {model.__class__.__name__}, {args.typecode}, {args.typecodes}, {dist}"

    add_line_to_file(args.save_file, line, first_line)


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


if __name__ == "__main__":
    main()
