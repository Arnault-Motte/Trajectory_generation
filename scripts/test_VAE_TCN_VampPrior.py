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
    displayer = Displayer()
    data_cleaner = Data_cleaner("data_orly/data/landings_LFPO_06.pkl")
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
    dropout = 0
    pseudo_input_num = 800
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
        pseudo_input_num=pseudo_input_num
    ).to(device)

    ## Training the model
    model.fit(data, epochs=1000, lr=1e-3, batch_size=500)


    ## Testing reconstuction on one batch

    x_recon,data_2 =  model.reproduce_data(data, 500, 2)
    print(x_recon.shape, "\n")
    traffic_init = data_cleaner.dataloader_traffic_converter(data_2,2)

    traffic_f = data_cleaner.output_converter(x_recon)
    displayer.plot_compare_traffic(traffic_init, traffic_f,plot_path="data_orly/figures/VAE_TCN_vamp_Recons.png")  # noqa: E501
    traffic_f.data.to_pickle('data_orly/generated_traff/reproducted/VAE_TCN_Vamp_reproducted_traff.pkl')
    print(data_cleaner.first_n_flight_delta_time(traffic_f))
    model.save_model("data_orly/src/generation/models/saved_weights/VAE_TCN_Vampprior.pth")
    return 0


if __name__ == "__main__":
    main()


