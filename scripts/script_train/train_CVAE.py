import sys  # noqa: I001
import os

current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import torch

from data_orly.src.generation.data_process import (
    Data_cleaner,
    return_traff_per_typecode,
)
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
import argparse
from traffic.core import Traffic


def main() -> int:
    praser = argparse.ArgumentParser(description="Training a CVAE")

    praser.add_argument("--data", type=str, default="", help="Path of the data")

    praser.add_argument(
        "--data_save",
        type=str,
        default="",
        help="Where to save the sampled data",
    )

    praser.add_argument(
        "--num_flights",
        nargs="+",
        type=int,
        default="",
        help="Num of flight for each chosen label",
    )
    praser.add_argument(
        "--temp",
        type=str,
        default="best_model.pth",
        help="Name of the temp file created during training",
    )
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

    praser.add_argument(
        "--scale", type=float, default=1, help="inital scale"
    )


    args = praser.parse_args()

    if any(value == "" for value in vars(args).values()):
        raise ValueError(
            "At least one argument has not been defined. Only temp can be ignored."
        )

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

    if len(args.num_flights) != len(args.typecodes) and args.num_flights != "":
        raise ValueError(
            "You must five as many values for num_flights as typecodes"
        )

    traff = Traffic.from_file(args.data)
    list_traff = return_traff_per_typecode(traff, args.typecodes)
    n_f = args.num_flights
    if args.num_flights != "":  # sampling some part
        combined_traff: Traffic = sum(
            [
                list_traff[i].sample(n_f[i]) if n_f[i] != -1 else list_traff[i]
                for i in range(len(list_traff))
            ]
        )
        combined_traff.to_pickle(args.data_save)
    else:
        combined_traff = traff
    
    print('yo')

    data_cleaner = Data_cleaner(
        traff=combined_traff,
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
    latent_dim = 64
    pooling_factor = 10
    stride = 1
    number_of_block = 4
    kernel_size = 16
    dilatation = 2
    dropout = 0.2
    pseudo_input_num = 800
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
        init_std=0.8,
    ).to(device)

    print("n_traj =", len(data_cleaner.basic_traffic_data))
    ## Training the model
    model.fit(data, epochs=1000, lr=1e-3, batch_size=500)
    model.save_model(args.weights)
    return 0


if __name__ == "__main__":
    main()
