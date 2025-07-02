import sys  # noqa: I001
import os


current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.generation.data_process import (
    Data_cleaner,
    compute_time_delta,
    filter_missing_values,
    return_traff_per_typecode,
)
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
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
        type=float,
        default=[],
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
        default=[],
        help="Typecodes to train the model",
    )
    praser.add_argument(
        "--vrate", type=int, default=0, help="Use vertical rate"
    )

    praser.add_argument(
        "--cuda", type=int, default=0, help="Index of the GPU to be used"
    )
    praser.add_argument(
        "--weights_data",
        type=int,
        default=0,
        help="True if you want to weight the data",
    )
    praser.add_argument(
        "--pseudo_in",
        type=int,
        default=800,
        help="True if you want to weight the data",
    )
    praser.add_argument(
        "--l_dim",
        type=int,
        default=64,
        help="True if you want to weight the data",
    )

    praser.add_argument(
        "--max_num_flights",
        type=int,
        default=-1,
        help="Limits the number of flights to be considered",
    )
    praser.add_argument(
        "--balanced",
        type=int,
        default=0,
        help="1 if we want to take num flight corresponding to teh pourcentage of the last label num_flights",
    )

    praser.add_argument(
        "--cond_pseudo",
        type=int,
        default=0,
        help="1 if the pseudo inputs must be conditioned",
    )

    praser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="The batch size for training",
    )

    praser.add_argument("--scale", type=float, default=1, help="inital scale")

    args = praser.parse_args()

    # if any(value == "" for value in vars(args).values()):
    #     raise ValueError(
    #         "At least one argument has not been defined. Only temp can be ignored."
    #     )
    vr_rate = bool(args.vrate)

    print("STARTING----------------------------------------------------")

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

    if len(args.num_flights) != len(args.typecodes) and len(args.num_flights) != 0:
        raise ValueError(
            "You must five as many values for num_flights as typecodes"
        )

    limit_nf = (
        args.max_num_flights
    )  # in case we want to limit the number of flights to be considered
    if limit_nf == -1:
        limit_nf = float("inf")

    print(limit_nf)

    traff = Traffic.from_file(args.data)
    print(len(traff))
    print("Nan values remaining: ",traff.data[columns].isna().any(axis=1).sum())

    if len(args.typecodes) != 0: #limiting the number of labels
        list_traff = return_traff_per_typecode(traff, args.typecodes)
        n_f = args.num_flights

    if len(args.num_flights) != 0:  # sampling some part
        if 1.0 >= n_f[0] > 0.0:  # in case we want %,
            if (
                not args.balanced
            ):  # % of the number of trajectory for each trajectory
                n_f = [
                    int(n_f[i] * min(len(list_traff[i]), limit_nf))
                    for i in range(len(list_traff))
                ]
            else:  # len computed using the % of the last trajectory
                n_f = [
                    int(n_f[i] * min(len(list_traff[-1]), limit_nf))
                    for i in range(
                        len(list_traff)
                    )  # help having a balanced data
                ]

        # Sampling the according number of trajectories
        combined_traff: Traffic = sum(
            [
                list_traff[i].sample(n_f[i]) if n_f[i] != -1 else list_traff[i]
                for i in range(len(list_traff))
            ]
        )
        combined_traff.to_pickle(args.data_save)
    else:
        combined_traff = traff

    # combined_traff = combined_traff.sample(200)
    print("yo")
    print(len(combined_traff))
    print(combined_traff.data.columns)


    if len(args.typecodes) != 0:
        data_cleaner = Data_cleaner(
            traff=combined_traff,
            columns=columns,
            chosen_typecodes=args.typecodes,
        )
    else :
        data_cleaner = Data_cleaner(
            traff=combined_traff,
            columns=columns,
            airplane_types_num=10, #limiting to the 10 most commun typecodes
        )


    data = data_cleaner.clean_data()

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
    labels_latent = 16
    labels = data_cleaner.return_labels()
    labels_dim = labels.shape[1]

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
        temp_save=f"best_model{args.cuda}.pth",
        conditioned_prior=True,
        num_worker=6,
        init_std=args.scale,
        d_weight=bool(args.weights_data),
        pseudo_labels=True,
    ).to(device)

    print("n_traj =", len(data_cleaner.basic_traffic_data))
    ## Training the model
    model.fit(data, labels, epochs=1000, lr=1e-3, batch_size=args.batch_size,step_size=100)
    model.save_model(args.weights)
    scalers_path = args.weights.split('.')[0] + "_scalers.pkl"
    data_cleaner.save_scalers(scalers_path)
    return 0


if __name__ == "__main__":
    main()
