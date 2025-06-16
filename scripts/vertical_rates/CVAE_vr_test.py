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
    return_traff_per_typecode,
    filter_missing_values,
    compute_time_delta,
)
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
from data_orly.src.generation.test_display import plot_distribution_typecode
from traffic.core import Traffic
from tqdm import tqdm


def main() -> None:
    praser = argparse.ArgumentParser(description="Training a CVAE")

    praser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to acess the data",
    )

    praser.add_argument(
        "--plot_path",
        type=str,
        default="",
        help="Path to acess the data",
    )

    praser.add_argument(
        "--num_flights",
        nargs="+",
        type=float,
        default=[],
        help="Num of flight for each chosen label",
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

    data_cleaner = Data_cleaner(
        file_name=args.data, columns=columns, airplane_types_num=10
    )

    data = data_cleaner.clean_data()  # be good if i just saved the scaler
    data_cleaner.return_labels()
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
        condition_pseudo_inputs=bool(args.cond_pseudo),
    ).to(device)

    model.load_model(args.weights)

    gen_list = []

    # max_num_labels = 10
    # labels = data_cleaner.get_typecodes_labels()[:max_num_labels]
    labels = ['B738', 'A321', 'A21N', 'A333']
    max_num_labels = len(labels)
    outputs_tensors = []
    for label in tqdm(labels,desc = 'generating for each label'):
        print(label)
        label_ar = np.array(
            [label]
        ).reshape(-1, 1)

        label_ar = data_cleaner.one_hot.transform(label_ar)
        print(label_ar.shape)
        print(label_ar)
        label_tens = torch.Tensor(label_ar).to(next(model.parameters()).device)
        label_tens =  label_tens.expand(500,-1)
        print("l_t: ",label_tens.shape)
        generated = model.sample(
            num_samples=2000, batch_size=500, labels=label_tens
        )
        outputs_tensors.append(generated)
        # traff = data_cleaner.output_converter(generated, landing=True)
        # text_label = label # data_cleaner.one_hot.inverse_transform(label)
        # traff = traff.assign(typecode=text_label)
        # print(traff.data.columns)
        # gen_list.append(traff)

    out = torch.concat(outputs_tensors,dim = 0)
    converted = data_cleaner.output_converter(out)
    tc = [labels[i//(2000*seq_len)] for i in range(2000*len(labels)*seq_len)]
    d = converted.data
    d['typecode'] = tc
    converted = Traffic(d)
    total_traff = converted

    # total_traff = sum(gen_list)
    print(total_traff.data["typecode"].unique())
    path = args.plot_path
    plot_distribution_typecode(
        total_traff,
        path.split(".")[0] + "_1.png",
        path.split(".")[0] + "_2.png",
        hist=False,
    )


if __name__ == "__main__":
    main()
