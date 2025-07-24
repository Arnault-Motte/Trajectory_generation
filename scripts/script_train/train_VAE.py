import sys  # noqa: I001
import os

current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import argparse

import torch
from src.data_process import Data_cleaner
from src.models.VAE_TCN_VampPrior import *  # noqa: F403
from traffic.core import Traffic


def main() -> int:
    praser = argparse.ArgumentParser(description="Training a VAE")

    praser.add_argument("--data", type=str, default="", help="Path of the data")

    praser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path and name of the file where to save the weights of the model",
    )
    praser.add_argument(
        "--typecodes",
        type=str,
        nargs="+",
        default="",
        help="Typcodes to train the model",
    )
    praser.add_argument(
        "--vrate", type=int, default=0, help="Use vertical rate"
    )

    praser.add_argument(
        "--cuda", type=int, default=0, help="Index of the GPU to be used"
    )
    praser.add_argument("--scale", type=float, default=1, help="inital scale")
    praser.add_argument("--l_dim", type=int, default=64, help="inital scale")
    praser.add_argument(
        "--pseudo_in", type=int, default=800, help="pseudo inputs num"
    )

    praser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Determines the %% of the data associated to the typecode that will be used for training",
    )

    praser.add_argument(
        "--save_sampled",
        type=str,
        default="",
        help="If you setted sample, you can use this parameter to save the new dataset. The path of the save file.",
    )

    args = praser.parse_args()

    # if any(value == "" for value in vars(args).values()):
    #     raise ValueError(
    #         "At least one argument has not been defined. Only temp can be ignored."
    #     )

    vr_rate = bool(args.vrate)

    print(sys.path)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )  # noqa: F405
    print(device)
    ## Getting the data
    ##Setting the chosen columns
    columns = ["track", "groundspeed", "timedelta"]
    columns += ["vertical_rate"] if vr_rate else ["altitude"]

    print(f"chosen columns : {columns}")

    if 0 < args.sample < 1.0:
        t = Traffic.from_file(args.data)
        t = t.sample(int(len(t) * args.sample))

        if args.save_sample != "":
            t.to_pickle(args.save_sample)

        data_cleaner = Data_cleaner(
            traff=t,
            columns=columns,
            chosen_typecodes=args.typecodes,
        )
    else:
        data_cleaner = Data_cleaner(
            args.data,
            columns=columns,
            chosen_typecodes=args.typecodes,
        )

    data = data_cleaner.clean_data()
    labels = data_cleaner.return_labels()
    print(labels, labels.shape)

    ##Getting the model
    seq_len = 200
    in_channels = len(data_cleaner.columns)
    output_channels = 64
    latent_dim = args.l_dim
    pooling_factor = 10
    stride = 1
    number_of_block = 4
    kernel_size = 16
    dilatation = 2
    dropout = 0.2
    pseudo_input_num = args.pseudo_in
    patience = 30
    min_delta = -100
    print(in_channels)

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
        temp_name=f"best_model{args.cuda}.pth",
        init_std=args.scale,
    ).to(device)

    print("n_traj =", len(data_cleaner.basic_traffic_data))
    ## Training the model
    model.fit(data, epochs=1000, lr=1e-3, batch_size=500)
    model.save_model(args.weights)
    path = "models_paper/" + args.weights.split("/")[-1].split(".")[0]
    model.save_model_ONNX(
        path
    )
    data_cleaner.save_scalers(path +"/scalers.pkl")
    return 0


if __name__ == "__main__":
    main()
