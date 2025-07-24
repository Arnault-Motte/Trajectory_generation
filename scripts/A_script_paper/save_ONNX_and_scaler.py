import sys  # noqa: I001
import os

from tqdm import tqdm



CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch
from src.data_process import Data_cleaner, return_labels
from src.generation import Generator, ONNX_Generator
from src.models.CVAE_ONNX import CVAE_ONNX
from src.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from src.models.CVAE_TCN_VampPrior_old import CVAE_TCN_Vamp_old
from src.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from src.simulation import Simulator
from src.test_display import plot_traffic, vertical_rate_profile_2
from traffic.core import Traffic


def main() -> None:
    parser = argparse.ArgumentParser(description="My script with arguments")

    parser.add_argument(
        "--weight_file",
        type=str,
        default="/home/arnault/traffic/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_vr_direct_cond_A21N.pth",
        help="Path of the weight file",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="The directory to save the scaler and the ONNX file",
    )
    parser.add_argument(
        "--scene_file",
        type=str,
        default="",
        help="Name of the scene file",
    )

    parser.add_argument(
        "--nf", type=int, default=2000, help="Number of flights to be generated"
    )
    parser.add_argument(
        "--typecodes",
        nargs="+",
        type=str,
        default=[],
        help="typecodes of interest",
    )
    parser.add_argument("--typecode_to_gen", type=str, default="")
    parser.add_argument(
        "--cond", type=int, default=0, help="Conditional model or not"
    )

    parser.add_argument(
        "--vertical_rate",
        type=int,
        default=0,
        help="Model trained with vertical rate",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/takeoffs_LFPO_07.pkl",
        help="path to the data",
    )

    parser.add_argument(
        "--cuda",
        type=int,
        default="0",
        help="Index of the cuda GPU to use",
    )

    parser.add_argument(
        "--l_dim",
        type=int,
        default=64,
        help="latent dim of the model",
    )

    parser.add_argument(
        "--pseudo_in",
        type=int,
        default=800,
        help="pseudo inputs num",
    )

    parser.add_argument(
        "--scaler",
        type=int,
        default=0,
        help="equal to 1 if you want to use the already save scalers",
    )
    parser.add_argument(
        "--cond_pseudo",
        type=int,
        default=0,
        help="equal to 1 if the model used conditioned pseudo labels",
    )

    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )  # noqa: F405
    print(device)
    print(args.typecodes)
    conditional = bool(args.cond)

    columns = ["track", "groundspeed", "timedelta"]
    columns += ["vertical_rate"] if args.vertical_rate else ["altitude"]
    if not args.scaler:
        data_cleaner = Data_cleaner(
            args.data,
            columns=columns,
            chosen_typecodes=args.typecodes,
            airplane_types_num=10 if len(args.typecodes) ==0 else -1,
        )
        print(data_cleaner.chosen_types)
        data = data_cleaner.clean_data()
    else:
        data_cleaner = Data_cleaner(no_data=True, columns=columns)
        data_cleaner.load_scalers(args.weight_file.split(".")[0] + "_scalers.pkl")

    print("start")

    seq_len = 200
    # in_channels = len(data_cleaner.columns)
    in_channels = 4
    output_channels = 64
    latent_dim = args.l_dim
    pooling_factor = 10
    stride = 1
    number_of_block = 4
    kernel_size = 16
    dilatation = 2
    dropout = 0.2
    pseudo_input_num = args.pseudo_in  # *1.5
    patience = 30
    min_delta = -100
    labels_latent = 16
    if not args.scaler:
        labels = data_cleaner.return_labels() 
    # labels_dim = labels.shape[1]
    # print("l ",labels_dim)
    labels_dim = 10
    # print(labels.shape)
    if not conditional:
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
    else:
        if not args.cond_pseudo:
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
        else : 
            model = CVAE_TCN_Vamp_old(
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
                init_std=1,
                condition_pseudo_inputs=args.cond_pseudo
            ).to(device)

    

    model.load_model(weight_file_path=args.weight_file)
    model.save_model_ONNX(args.save_dir)
    data_cleaner.save_scalers(args.save_dir + "/scalers.pkl")

    print("model_saved")

   







if __name__ == "__main__":
    main()
