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
from data_orly.src.generation.data_process import (
    Data_cleaner,
    return_traff_per_typecode,
)
from data_orly.src.generation.test_display import plot_traffic
from data_orly.src.generation.models.CVAE_TCN_VampPrior import (
    CVAE_TCN_Vamp,
)
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
from traffic.core import Flight, Traffic
from traffic.data import airports
from data_orly.src.core.networks import get_data_loader

import random
import pandas as pd


# def from_f_return_f_label(f: Flight, mask: list[bool]) -> Traffic:
#     #print(mask)
#     mask_s = pd.Series(mask)
#     df = f.data
#     group_ids = (mask_s != mask_s.shift()).cumsum()
#     print(group_ids.tolist())
#     true_groups = mask_s[group_ids][mask_s].groupby(group_ids[mask_s])
#     print(len(true_groups))

#     # Extract the DataFrame chunks
#     chunks = [df.iloc[group.index] for _, group in true_groups]
#     print(len(chunks))
#     return sum([Flight(fd) for fd in chunks])


def from_f_return_f_label(f: Flight, mask: list[bool]) -> Traffic:
    mask_s = pd.Series(mask)
    df = f.data

    # Identify runs of values (True/False)
    group_ids = (mask_s != mask_s.shift()).cumsum()

    # Create a DataFrame to help with grouping
    temp_df = pd.DataFrame({"mask": mask_s, "group": group_ids})
    df["group"] = group_ids  # temporarily attach group info to df

    # Filter only the groups where mask is True
    true_groups = temp_df[temp_df["mask"]].groupby("group")

    # Extract the corresponding chunks from df
    chunks = [
        df[df["group"] == group_id].drop(columns="group")
        for group_id in true_groups.groups
    ]

    return sum([Flight(chunk) for chunk in chunks])


def return_traffic_data(
    dataset: pd.DataFrame, mask: torch.Tensor, n_points: int = 200
) -> list[Traffic]:
    n_f = len(dataset) // n_points
    print(len(dataset))

    flights = [
        Flight(dataset.iloc[n_points * i : n_points * (i + 1)])
        for i in range(n_f)
    ]  # every flight
    print("f_len =", [len(f) for f in flights])
    flights_parted = [
        from_f_return_f_label(f, mask[i].cpu().tolist())
        for i, f in enumerate(flights)
    ]
    print("f_len =", [len(f) for t in flights_parted for f in t])

    return flights_parted


def turn_sub_part_f(
    predict: torch.Tensor,
    mask_t: torch.Tensor,
    label_lat_lon: np.ndarray,
    data_cleaner: Data_cleaner,
    lat_lon: bool = False,
) -> list[Traffic]:
    """
    Allows to reconstruct the latitude and logitude corresponding to predict.
    Label_latllon containes all the lat lon of the points in the label as a np.ndarray
    """
    t_list = []
    for b, batch in enumerate(predict):
        i = 0
        print(mask_t.shape)
        mask = mask_t[b]
        current_b = []
        print(mask)
        while i < len(batch):
            before_start = None
            print(mask.shape)
            print(mask[i].shape)

            while mask[i]:  # while we have padding values
                i += 1
                if i == len(batch):
                    break
            if i == len(batch):
                break
            before_start = i  # if equal to zero then first points only belongs predict and not the label

            start = i
            while i < len(batch) and (not mask[i]):
                i += 1

            end = i
            section = predict[b][start:end]  # tensor

            if before_start != 0:
                first_point = label_lat_lon[b][before_start]
                traffic = data_cleaner.output_converter(
                    section.unsqueeze(0),
                    landing=False,
                    # mean_point=(first_point[0], first_point[1]),
                    seq_len=section.shape[0],
                    lat_lon=not lat_lon,  # do not generate new lat lon cordinate from track if we already have it
                )
            else:
                last_point = label_lat_lon[b][end - 1]  # last point (is known)
                traffic = data_cleaner.output_converter(
                    section.unsqueeze(0),
                    landing=False,
                    mean_point=(last_point[0], last_point[1]),
                    seq_len=section.shape[0],
                    lat_lon=not lat_lon,
                )
            current_b.append(traffic)

        t_list.append(sum(current_b))

    return t_list


def generate_trajectories(
    predict: np.ndarray,
    labels: np.ndarray,
    n_traj: int,
    model: CVAE_TCN_Vamp,
    data_clean: Data_cleaner,
    data_padded: pd.DataFrame,
    seed: int = None,
    seq_len: int = 200,
    padding: float = -1.1,
    generate: bool = True,
    lat_lon: bool = False,
    label_msk: bool = False,
    keep_label: bool = False,
) -> tuple[list[Traffic], list[Traffic], list[int]]:
    """
    Generates and cleans the trajectories, through our dataset
    """
    model.eval()
    if seed is not None:
        random.seed(seed)

    data_l, _ = get_data_loader(
        predict,
        labels,
        train_split=0.9,
        shuffle=False,
        padding=-1.1,
        batch_size=1,
        testing=True,
    )

    n = len(data_l.dataset)
    print("len ", n)
    indices = random.sample(range(n), n_traj)
    samples = []
    masks = []
    for i, batch in enumerate(data_l):
        if i in indices:
            pred = batch[0].to(next(model.parameters()).device)
            lab = batch[1].to(next(model.parameters()).device)

            mask = (
                (pred == padding).all(dim=2).to(next(model.parameters()).device)
            )
            label_mask = None
            if label_msk:
                label_mask = (
                    (lab != padding)
                    .all(dim=2)
                    .to(next(model.parameters()).device)
                )
            if generate:
                mu_p, lvar_p = model.pseudo_inputs_latent()
                prior_weight = model.get_prior_weight(
                    lab, mask if not label_msk else label_mask
                )
                distrib = create_mixture(
                    mu_p,
                    lvar_p,
                    vamp_weight=prior_weight.reshape(prior_weight.shape[-1]),
                )
                z = distrib.sample((1,))
                # labels = lab.permute(0, 2, 1)
                labels_i = lab
                print(mask.shape)
                print(labels_i.shape)
                # t_mask = mask.unsqueeze(0).permute(0,2,1)
                l_msk = label_mask if label_msk else mask
                if l_msk is not None:
                    l_msk = ~l_msk
                generated_data = model.decode(z, labels_i, l_msk).detach().cpu()
            else:
                generated_data = (
                    model.forward(pred, lab, mask, label_mask)[0].detach().cpu()
                )
            samples.append(generated_data)
            masks.append(mask)

    mask_t = torch.concat(masks, dim=0)
    dataframes = [
        data_padded[["latitude", "longitude"]].iloc[
            seq_len * i : seq_len * (i + 1)
        ]
        for i in indices
    ]
    real_data = pd.concat(
        [data_padded.iloc[seq_len * i : seq_len * (i + 1)] for i in indices],
        axis=0,
        ignore_index=True,
    )
    print("rd", len(real_data))
    lat_lon_label = np.stack([df.values for df in dataframes], axis=0)
    # print(lat_lon_label)
    # print(lat_lon_label.shape)
    print(pred.shape)
    print(pred.permute(0, 2, 1)[0][1])
    print(lab.permute(0, 2, 1)[0][1])
    print(generated_data[0][1])

    if keep_label:
        l_t = [
            data_clean.output_converter(s.permute(0, 2, 1), landing=False)
            for s in samples
        ]
        return l_t, None, indices
    pred = torch.concat(samples, dim=0).permute(0, 2, 1)
    print(pred.shape)
    pred_flights = turn_sub_part_f(
        pred,
        mask_t,
        data_cleaner=data_clean,
        label_lat_lon=lat_lon_label,
        lat_lon=lat_lon,
    )
    print("pred", len(pred_flights))
    label_list = return_traffic_data(real_data, mask_t)

    return pred_flights, label_list, indices


def cut_tensor(s: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """
    Returns a list containing all the sections to predict in the output
    """

    seg = []
    mask = mask.squeeze(-1)
    in_seg = False
    start = 0
    for j in tqdm(range(mask.shape[0]), desc="processing sample"):
        if mask[j] and not in_seg:
            start = j
            in_seg = True
        elif not mask[j] and in_seg:
            seg.append(s[start:j])
            in_seg = False
    if in_seg:
        seg.append(s[start:])

    return seg


def find_pred(
    samples: list[torch.Tensor], masks: list[torch.Tensor], pred: bool = True
) -> list[list[torch.Tensor]]:
    """
    find the prediction part of the trajectories in samples accoring to the masks.
    Can also be used to return the label part by setting pred to false and passing real data.
    """
    predictions = []
    for i, s in tqdm(enumerate(samples), desc="Processing samples"):
        # pred = s[~masks[i]]
        mask = masks[i] if not pred else ~masks[i]
        seg = cut_tensor(s, mask)
        predictions.append(seg)
    return predictions


def display_flight_sep(
    traffic_s: Traffic,
    traffic_e: Traffic,
    save_path: str,
    split: float = 0.5,
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

    praser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="1 if you want to use the padding vers",
    )

    praser.add_argument(
        "--test_conc_msk",
        type=int,
        default=0,
        help="1 if you want to directly use masking to combine the label in the econder requires test_conc to be true",
    )

    praser.add_argument(
        "--test_conc",
        type=int,
        default=0,
        help="1 if you want to directly concatenate the label in the encoder",
    )

    praser.add_argument(
        "--use_lat_lon",
        type=int,
        default=0,
        help="1 if you want to use the latitude and longitude instead of track and groundspeed",
    )

    praser.add_argument(
        "--last_fcl",
        type=int,
        default=0,
        help="",
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
    columns = (
        ["latitude", "longitude"]
        if args.use_lat_lon
        else ["track", "groundspeed"]
    )
    columns += ["timedelta"]
    # columns = ["track", "groundspeed", "timedelta"]
    columns += ["vertical_rate"] if vr_rate else ["altitude"]

    print(f"chosen columns : {columns}")

    data_cleaner = Data_cleaner(
        file_name=args.data,
        columns=columns,
    )
    data = data_cleaner.clean_data()

    basic_data = data_cleaner.basic_traffic_data.data

    def mask_basic_data(label: np.ndarray) -> np.ndarray:
        masks = np.all(label == -1.1, axis=-1)
        f_masks = masks.reshape(-1)
        return f_masks

    def split_data(
        data: np.ndarray, split: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len = data.shape[1]
        n_flights = int(seq_len * split)

        predict = data.copy()
        label = data.copy()
        predict[:, : n_flights - 1, :] = -1.1
        label[:, n_flights:, :] = -1.1
        label_points = basic_data.copy()
        mask = mask_basic_data(label)  # to getr which points are hidden
        label_points[mask] = None
        print(label_points.shape)
        # predict = predict[:, n_flights:, :]
        # label = label[:, :n_flights, :]

        return predict, label, label_points

    predict, label_traj, label_points = split_data(data)
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
        padding=-1.1 if args.padding else None,
        test_concat=args.test_conc,
        test_concat_mask=args.test_conc_msk,
        final_fcl= args.last_fcl
    ).to(device)

    print("n_traj =", len(data_cleaner.basic_traffic_data))
    ## Training the model
    model.load_model(args.weights)
    # print(model.label_mask)

    pred_flights, label_list, indices = generate_trajectories(
        predict=predict,
        labels=labels,
        n_traj=1,
        model=model,
        data_clean=data_cleaner,
        data_padded=label_points,
        seed=1,
        padding=-1.1,
        generate=True,
        lat_lon=args.use_lat_lon,
        label_msk=True,
    )

    if label_list is None:
        true_data = pd.concat(
            [
                data_cleaner.basic_traffic_data.data.iloc[
                    i * 200 : (i + 1) * 200
                ]
                for i in indices
            ],
            axis=0,
        )
        print(Traffic(true_data)[0].data)

        print([len(f) for f in pred_flights])
        (print(pred_flights[0].data))
        display_flight_sep(
            Traffic(true_data), sum(pred_flights), "test_temp.png"
        )
        return

    t_pred = sum(pred_flights)
    t_start = sum(label_list)

    print(len(t_pred))
    print(len(t_start))
    display_flight_sep(t_start, t_pred, "test_pred_not_full_no_last_fcl_2.png")

    print(t_pred.data.iloc[0])
    print(t_start.data.iloc[-1])
    true_data = pd.concat(
        [
            data_cleaner.basic_traffic_data.data.iloc[i * 200 : (i + 1) * 200]
            for i in indices
        ],
        axis=0,
    )
    print(indices)
    print(len(Traffic(true_data)))
    plot_traffic(Traffic(true_data), "test_pred_padded_1.png")

    # first_part = (
    #     torch.Tensor(label_traj[1]).unsqueeze(0).permute(0, 2, 1).to(device)
    # )

    # print(first_part.shape)
    # tensor_res = model.sample(1, 1, first_part).to(device)  # 100
    # predict_part = torch.Tensor(predict[1]).unsqueeze(0).to(device)
    # res_2 = model.forward(
    #     predict_part,
    #     first_part if not args.complexe_w else first_part.permute(0, 2, 1),
    # )[0].permute(0, 2, 1)
    # print(res_2.shape)
    # first_part = first_part.permute(0, 2, 1)
    # result = torch.cat((first_part, tensor_res), dim=1).to(device)
    # true_traj = torch.cat((first_part, predict_part), dim=1).to(device)
    # resultat_2 = torch.cat((first_part, res_2), dim=1).to(device)

    # print(result.shape)
    # print(true_traj.shape)

    # last_point_first = data_cleaner.basic_traffic_data[1].data.iloc[
    #     99
    # ]  # last point of first part

    # first_part_t = data_cleaner.output_converter(
    #     first_part,
    #     landing=False,
    #     mean_point=(
    #         last_point_first["latitude"],
    #         last_point_first["longitude"],
    #     ),
    #     seq_len=100,
    # )

    # predicted = data_cleaner.output_converter(
    #     tensor_res, landing=False, seq_len=100
    # )
    # reproduced = data_cleaner.output_converter(
    #     res_2,
    #     landing=False,
    #     seq_len=100,
    # )
    # true_traff = data_cleaner.output_converter(
    #     predict_part,
    #     landing=False,
    #     seq_len=100,
    # )

    # # result_traff = data_cleaner.output_converter(result, landing=False)
    # # true_traff = data_cleaner.output_converter(true_traj, landing=False)
    # # reproduced = data_cleaner.output_converter(resultat_2, landing=False)

    # # display_flight_sep(result_traff[0], "test_pred_1.png")
    # # display_flight_sep(true_traff[0], "test_pred_2.png")
    # # display_flight_sep(reproduced[0], "test_pred_3_recons.png")

    # display_flight_sep(first_part_t, predicted, "test_pred_1_better.png")
    # display_flight_sep(first_part_t, true_traff, "test_pred_2_better.png")
    # display_flight_sep(
    #     first_part_t, reproduced, "test_pred_3_recons_better.png"
    # )

    # print(predicted[0].data.iloc[98:102])
    # print(true_traff[0].data.iloc[98:102])
    # print(reproduced[0].data.iloc[98:102])

    return 0


if __name__ == "__main__":
    main()
