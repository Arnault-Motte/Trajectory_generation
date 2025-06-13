import sys  # noqa: I001
import os


current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import argparse

import matplotlib.pyplot as plt
import torch
from cartes.crs import Lambert93

import numpy as np
from data_orly.src.generation.data_process import(
    Data_cleaner,
    return_traff_per_typecode,
)
from data_orly.src.generation.models.CVAE_TCN_VampPrior import (
    CVAE_TCN_Vamp,
)
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
from traffic.core import Flight, Traffic
from traffic.data import airports


def display_flight_sep(
    traffic_s: Traffic, traffic_e: Traffic, save_path: str, split: float = 0.5
) -> None:
    # f_len = len(flight)
    # f1 = Flight(flight.data.iloc[: int(f_len * 0.5)])
    # f2 = Flight(flight.data.iloc[int(f_len * 0.5) :])

    f1: Traffic = traffic_s
    f2 = traffic_e

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
        f1.plot(ax, alpha=0.5, color="blue")
        f2.plot(ax, alpha=0.5, color="red")
        # flight.plot(ax, alpha=0.1, color="grey")

    plt.savefig(save_path)


def main() -> int:
    praser = argparse.ArgumentParser(description="Training a CVAE")

    praser.add_argument("--data", type=str, default="", help="Path of the data")

    praser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path and name of the file where to save the weights of the model",
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
        "--complexe_w",
        type=int,
        default=0,
        help="True if you want better prior weights attribution",
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
        "--labels_l_dim",
        type=int,
        default=64,
        help="True if you want to weight the data",
    )

    praser.add_argument("--scale", type=float, default=1, help="inital scale")

    args = praser.parse_args()

    if any(value == "" for value in vars(args).values()):
        raise ValueError(
            "At least one argument has not been defined. Only temp can be ignored."
        )

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

    data_cleaner = Data_cleaner(
        file_name=args.data,
        columns=columns,
    )
    data = data_cleaner.clean_data()

    def split_data(
        data: np.ndarray, split: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len = data.shape[1]
        n_flights = int(seq_len * split)

        predict = data.copy()
        label = data.copy()
        # predict[:, :n_flights, :] = -10
        # label[:, n_flights:, :] = -10
        predict = predict[:, n_flights:, :]
        label = label[:, :n_flights, :]

        return predict, label

    predict, label_traj = split_data(data)
    print(predict.shape, label_traj.shape)

    ##Getting the model
    seq_len = predict.shape[1]  # 200
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
    min_delta = -30
    print(in_channels)
    labels_latent = args.labels_l_dim

    labels_dim = label_traj[1, :, :].size
    print(labels_dim)
    labels = label_traj
    # labels = data_cleaner.return_labels()
    # labels_dim = labels.shape[1]

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
        trajectory_label=True,
        complexe_weight=args.complexe_w,
    ).to(device)

    print("n_traj =", len(data_cleaner.basic_traffic_data))
    ## Training the model
    model.load_model(args.weights)

    first_part = (
        torch.Tensor(label_traj[1]).unsqueeze(0).permute(0, 2, 1).to(device)
    )

    print(first_part.shape)
    tensor_res = model.sample(1, 1, first_part).to(device)  # 100
    predict_part = torch.Tensor(predict[1]).unsqueeze(0).to(device)
    res_2 = model.forward(
        predict_part,
        first_part if not args.complexe_w else first_part.permute(0, 2, 1),
    )[0].permute(0, 2, 1)
    print(res_2.shape)
    first_part = first_part.permute(0, 2, 1)
    result = torch.cat((first_part, tensor_res), dim=1).to(device)
    true_traj = torch.cat((first_part, predict_part), dim=1).to(device)
    resultat_2 = torch.cat((first_part, res_2), dim=1).to(device)

    print(result.shape)
    print(true_traj.shape)

    last_point_first = data_cleaner.basic_traffic_data[1].data.iloc[
        99
    ]  # last point of first part


    first_part_t = data_cleaner.output_converter(
        first_part,
        landing=False,
        mean_point=(
            last_point_first["latitude"],
            last_point_first["longitude"],
        ),
        seq_len=100,
    )

    predicted = data_cleaner.output_converter(
        tensor_res, landing=False, seq_len=100
    )
    reproduced = data_cleaner.output_converter(
        res_2,
        landing=False,
        seq_len=100,
    )
    true_traff = data_cleaner.output_converter(
        predict_part,
        landing=False,
        seq_len=100,
    )

    # result_traff = data_cleaner.output_converter(result, landing=False)
    # true_traff = data_cleaner.output_converter(true_traj, landing=False)
    # reproduced = data_cleaner.output_converter(resultat_2, landing=False)

    # display_flight_sep(result_traff[0], "test_pred_1.png")
    # display_flight_sep(true_traff[0], "test_pred_2.png")
    # display_flight_sep(reproduced[0], "test_pred_3_recons.png")

    display_flight_sep(first_part_t, predicted, "test_pred_1_better.png")
    display_flight_sep(first_part_t, true_traff, "test_pred_2_better.png")
    display_flight_sep(
        first_part_t, reproduced, "test_pred_3_recons_better.png"
    )

    print(predicted[0].data.iloc[98:102])
    print(true_traff[0].data.iloc[98:102])
    print(reproduced[0].data.iloc[98:102])

    return 0


if __name__ == "__main__":
    main()
