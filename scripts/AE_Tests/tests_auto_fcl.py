import sys  # noqa: I001
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import torch

from data_orly.src.generation.data_process import Data_cleaner, filter_outlier
from data_orly.src.generation.models.AE_FCL import *  # noqa: F403
from data_orly.src.generation.test_display import Displayer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


def main()->int:
    print(sys.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data
    displayer = Displayer()
    data_cleaner = Data_cleaner("data_orly/landings_LFPO_06.pkl")
    data = data_cleaner.clean_data()

    #mean_end = data_cleaner.get_mean_end_point()
    # coordinates = {"latitude": mean_end[0], "longitude": mean_end[1]}
    # print(mean_end)

    ##Getting the model
    seq_len = 200
    num_channels = 4
    hidden_dim_1 = 218
    hidden_dim_2 = 128
    latent_dim = 64
    model = AE_FCL(  # noqa: F405
        hidden_dim_1, hidden_dim_2, latent_dim, seq_len, num_channels
    ).to(device)

    ## Training the model
    model.fit(data, epochs=1000, lr=1e-3, batch_size=500)

    ## Testing reconstuction on one batch

    x_recon =  model.reproduce_data(data, 500, 2)
    print(x_recon.shape, "\n")

    traffic_f = data_cleaner.output_converter(x_recon)
    displayer.plot_compare_traffic(data_cleaner.basic_traffic_data, traffic_f,plot_path="data_orly/figures/AE_FCL.png")  # noqa: E501
    filtered_traffic  = filter_outlier(traffic_f)
    displayer.plot_compare_traffic(data_cleaner.basic_traffic_data, filtered_traffic,plot_path="data_orly/AE_FCL_filtered.png")  # noqa: E501
    traffic_f.data.to_pickle('data_orly/generated_traff/reproducted/AE_FCL_reproducted_traff.pkl')
    return 0


if __name__ == "__main__":
    main()


