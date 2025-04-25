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
    clean_data,
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
        "--origin",
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
    print("total ",label_data.shape)

    labels_ar = np.array([args.typecode for _ in range(len(label_data))]).reshape(-1, 1)
    labels_ar = data_cleaner.one_hot.transform(labels_ar)
    labels_spec = torch.Tensor(labels_ar).to(device)


     #prepare test data
    print(args.typecode)
    selected_id = list(data_cleaner.return_flight_id_for_label(args.typecode))
    print(len(selected_id))
    
    traff_test = Traffic.from_file(args.origin)


    if "typecode" not in traff_test.data.columns:
        traff_test = traff_test.aircraft_data()


    print(set([f.typecode for f in traff_test[selected_id]]))
    print(traff_test[0:20])
    traff_test = traff_test.query(f'flight_id not in {selected_id}') #retourne tout
    print(len(traff_test))

    # test_cleaner = Data_cleaner(
    #     traff=traff_test,
    #     columns=columns,
    #     chosen_typecodes=args.typecodes,
    # )

    #select only the typecodes of interest
    
    traff_test: None | Traffic = traff_test.query(f'typecode == "{args.typecode}"')


    #print(len(traff_test))
    #scale the data with the right scaler, otherwise it won't work


    if traff_test != None:
        print(len(traff_test))
        test_data_ar = clean_data(traff_test,data_cleaner.scaler,data_cleaner.columns)
        print(test_data_ar.shape)
        mask = (test_data_ar > 1) | (test_data_ar < -1)
        rows_with_condition = np.any(mask, axis=(1,2))
        count = np.sum(rows_with_condition)
        print("wierd values : ", count)
        good_rows = ~rows_with_condition
        test_data_ar = test_data_ar[good_rows]
        print(len(test_data_ar))
        # weird_rows = test_data_ar[rows_with_condition]  # shape (?, 200, 4)
        # print("Shape of weird rows:", weird_rows.shape)
        # test_data_ar = weird_rows
        test_data = torch.tensor(test_data_ar, dtype=torch.float32).to(device)

        
    else:
        test_data = torch.Tensor([]).to(device)


    if len(test_data) != 0:
        test_labels = np.array([args.typecode for _ in range(len(test_data))]).reshape(-1, 1)
        test_labels = data_cleaner.one_hot.transform(test_labels)
        test_labels_spec = torch.Tensor(test_labels).to(device)

    vram_used = torch.cuda.memory_allocated(device)
    print(f"Current VRAM usage: {vram_used / (1024 ** 2):.2f} MB")

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
        vram_used = torch.cuda.memory_allocated(device)
        print(f"Current VRAM usage: {vram_used / (1024 ** 2):.2f} MB")
        print(label_data.shape)
        val_loss, val_kl, val_recons = model.compute_loss(
            label_data, labels_spec
        )
        if (len(test_data) != 0):
            test_loss, test_kl, test_recons = model.compute_loss(
                test_data, test_labels_spec
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
        vram_used = torch.cuda.memory_allocated(device)
        print(f"Current VRAM usage: {vram_used / (1024 ** 2):.2f} MB")
        val_loss, val_kl, val_recons = model.compute_loss(label_data)
        if (len(test_data) != 0):
            test_loss, test_kl, test_recons = model.compute_loss(test_data)

    print("n_traj =", len(label_data))
    vram_used = torch.cuda.memory_allocated(device)
    print(f"Current VRAM usage: {vram_used / (1024 ** 2):.2f} MB")

    ## Training the model
    

    val_loss = val_loss
    val_kl = val_kl 
    val_recons = val_recons 
    print(
        f"Validation set, Loss: {val_loss:.4f},MSE: {val_recons:.4f}, KL: {val_kl:.4f} "
    )

    if (len(test_data) != 0):
        print(
        f"Test set, Loss: {test_loss:.4f},MSE: {test_recons:.4f}, KL: {test_kl:.4f} "
    )



    with open(args.loss_file,"a") as file:
        if os.stat(args.loss_file).st_size == 0:
            file.write("Model file, Data file, Dataset Typecodes, Selected Typecode, Total loss, KL, log likelihood, MSE, test \n")
        file.write(f'{args.model}, {args.file}, {args.typecodes} ,{args.typecode}, {val_loss}, {val_kl}, {val_loss + val_kl}, {val_recons}, False \n')
        if (len(test_data) != 0):
            file.write(f'{args.model}, {args.file}, {args.typecodes} ,{args.typecode}, {test_loss}, {test_kl}, {test_loss + test_kl}, {test_recons}, True \n')
    return 0


if __name__ == "__main__":
    main()
