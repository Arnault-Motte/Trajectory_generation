import sys  # noqa: I001
import os


current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.core.networks import get_data_loader
from data_orly.src.generation.data_process import (
    Data_cleaner,
    return_traff_per_typecode,
)
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
from traffic.core import Traffic


def main() -> int:
    praser = argparse.ArgumentParser(description="Computing loss")

    praser.add_argument("--data", type=str, default="", help="Path of the data")

    praser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path and name of the file where to save the weights of the model",
    )

    praser.add_argument(
        "--file",
        type=str,
        default="",
        help="Path and name of the file where to save the weights of the model",
    )

    praser.add_argument(
        "--loss_file",
        type=str,
        default="",
        help="Path and name of the file where to save the weights of the model",
    )
    praser.add_argument(
        "--typecode",
        type=str,
        default="",
        help="Path and name of the file where to save the weights of the model",
    )
    praser.add_argument(
        "--typecodes",
        type=str,
        nargs="+",
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
        "--cond", type=int, default=0, help="True if the model is a CVAE"
    )

    args = praser.parse_args()

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

    data_cleaner = Data_cleaner(
        file_name=args.file,
        columns=columns,
        chosen_typecodes=args.typecodes,
    )

    data = data_cleaner.clean_data()
    labels = data_cleaner.return_labels()
    
    
    label_data = data_cleaner.return_typecode_array(args.typecode).to(device)
    # labels_spec = torch.Tensor(
    #     [
    #         data_cleaner.one_hot.transform(args.typecode)
    #         for i in range(len(label_data))
    #     ]
    # ).to(device)
    labels_ar = np.array([args.typecode for _ in range(len(label_data))]).reshape(-1, 1)
    labels_ar = data_cleaner.one_hot.transform(labels_ar)
    #print(labels_ar)
    labels_spec = torch.Tensor(labels_ar).to(device)



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
    labels_latent = 16
    labels = data_cleaner.return_labels()
    labels_dim = labels.shape[1]

    if args.cond == 1:
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
            conditioned_prior=True,
            num_worker=6,
            init_std=1,
        ).to(device)
        model.load_model(args.model)
        print(label_data.shape)
        val_loss, val_kl, val_recons = model.compute_loss(
            label_data, labels_spec
        )
    else:
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
            init_std=1,
        ).to(device)
        model.load_model(args.model)
        val_loss, val_kl, val_recons = model.compute_loss(label_data)

    print("n_traj =", len(label_data))

    ## Training the model
    

    val_loss = val_loss
    val_kl = val_kl 
    val_recons = val_recons 
    print(
        f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f} "
    )

    with open(args.loss_file,"a") as file:
        if os.stat(args.loss_file).st_size == 0:
            file.write("Model file, Data file, Dataset Typecodes, Selected Typecode, Total loss, KL, log likelihood, MSE \n")
        file.write(f'{args.model}, {args.file}, {args.typecodes} ,{args.typecode}, {val_loss}, {val_kl}, {val_loss + val_kl}, {val_recons} \n')
    return 0


if __name__ == "__main__":
    main()
