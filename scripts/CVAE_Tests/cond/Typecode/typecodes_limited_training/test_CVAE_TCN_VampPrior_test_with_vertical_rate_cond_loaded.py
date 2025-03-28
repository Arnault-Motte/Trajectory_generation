import sys  # noqa: I001
import os

current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import torch

from data_orly.src.generation.data_process import Data_cleaner
from data_orly.src.generation.generation import Generator
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
from data_orly.src.generation.test_display import Displayer


def main() -> int:
    print(sys.path)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data

    data_cleaner = Data_cleaner(
        "data_orly/data/takeoffs_LFPO_07.pkl",
        columns=["track", "vertical_rate", "groundspeed", "timedelta"],
        chosen_typecodes=["A320"],
    )
    displayer = Displayer(data_cleaner)
    data = data_cleaner.clean_data()
    labels = data_cleaner.return_labels()
    print(labels, labels.shape)
    labels_dim = labels.shape[1]

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
    pseudo_input_num = 800  # *1.5
    patience = 30
    min_delta = -100
    labels_latent = 16
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
        temp_name="best_model_A320.pth",
    ).to(device)

    print("n_traj =", len(data_cleaner.basic_traffic_data))
    ## Training the model
    model.load_model(
        "data_orly/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_vr_direct_cond_A320.pth"
    )
    ## Testing reconstuction on one batch

    x_recon, data_2 = model.reproduce_data(data, 500, 2)
    print(x_recon.shape, "\n")
    traffic_init = data_cleaner.dataloader_traffic_converter(data_2, 2)

    traffic_f = data_cleaner.output_converter(x_recon,landing=True)

    displayer.plot_compare_traffic(
        traffic_init,
        traffic_f,
        2000,
        plot_path="data_orly/figures/generation/rec_VAE_Vamp_w_vr_A320.png",
    )

    gen = Generator(None, data_cleaner, model)

    traff = gen.generate_n_flight(2000, 500)
    displayer.plot_traffic(
        traffic=traff,
        plot_path="data_orly/figures/generation/VAE_Vamp_w_vr_A320.png",
    )
    return 0


if __name__ == "__main__":
    main()
