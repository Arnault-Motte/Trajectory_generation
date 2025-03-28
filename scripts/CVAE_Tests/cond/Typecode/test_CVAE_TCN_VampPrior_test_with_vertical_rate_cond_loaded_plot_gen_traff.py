import sys  # noqa: I001
import os

current_path = os.getcwd()
sys.path.append(
    os.path.abspath(current_path)
)

print("Current working directory:", current_path)
print(os.path.dirname(__file__))

import torch

from data_orly.src.generation.data_process import Data_cleaner
from data_orly.src.generation.models.CVAE_TCN_VampPrior import *  # noqa: F403
from data_orly.src.generation.test_display import Displayer
from data_orly.src.generation.generation import Generator


def main() -> int:
    print(sys.path)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data

    data_cleaner = Data_cleaner(
        "data_orly/data/takeoffs_LFPO_07.pkl",
        columns=["track", "vertical_rate", "groundspeed", "timedelta"],
        airplane_types_num=10,
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
    pseudo_input_num = 800
    patience = 30
    min_delta = -100
    labels_latent = 16
    print(in_channels)

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
        temp_save="best_model_6.pth",
        conditioned_prior=True,
    ).to(device)

    ## Training the model
    # model.fit(data, labels, epochs=1000, lr=1e-3, batch_size=500)
    model.load_model(
        "data_orly/src/generation/models/saved_weights/CVAE_TCN_Vampprior_take_off_7_vr_direct_cond.pth"
    )

    gen = Generator(model,data_cleaner)

    top_10 = [label for label,_ in data_cleaner.get_top_10_planes()]
    gen_traffic = gen.generate_n_flight_per_labels(top_10,2000)

    displayer.plot_traff_labels("data_orly/figures/generation/CVAE_Vamp_w_vr.png",gen_traffic,top_10)

    return 0


if __name__ == "__main__":
    main()
