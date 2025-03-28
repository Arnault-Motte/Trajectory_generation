import sys  # noqa: I001
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)



import torch

from data_orly.src.generation.data_process import Data_cleaner, filter_outlier
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
from data_orly.src.generation.test_display import Displayer

# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# )


def main()->int:
    print(sys.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data

    data_cleaner = Data_cleaner("data_orly/data/landings_LFPO_06.pkl")
    displayer = Displayer(data_cleaner)
    data = data_cleaner.clean_data()
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
        early_stopping= True,
        patience=patience,
        min_delta=min_delta,
    ).to(device)

    ## Training the model
    model.load_model("data_orly/src/generation/models/saved_weights/VAE_TCN_Vampprior.pth")

    x_recons = model.sample(500,1).permute(0,2,1)
    print(x_recons.shape)
    traffic_res = data_cleaner.output_converter(x_recons)
    displayer.plot_traffic(traffic_res,plot_path="data_orly/figures/generation/VAE_TCN_Vamp.png" )

    return 0


if __name__ == "__main__":
    main()


