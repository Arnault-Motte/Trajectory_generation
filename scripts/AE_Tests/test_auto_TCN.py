import sys  # noqa: I001
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
)

from src.generation.data_process import Data_cleaner, filter_outlier
from src.generation.models.AE_TCN import *  # noqa: F403
from src.generation.test_display import Displayer

# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data
    displayer = Displayer()
    data_cleaner = Data_cleaner("data_orly/data/landings_LFPO_06.pkl")#,columns=["track","groundspeed","timedelta"])
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
    print(in_channels)

    model = AE_TCN(
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
    ).to(device)

    ## Training the model
    model.fit(data, epochs=1000, lr=1e-3, batch_size=500)

    ## Testing reconstuction on one batch

    x_recon,og_data = model.reproduce_data(data, 500, 2)
    print(x_recon.shape, "\n")

    traffic_f = data_cleaner.output_converter(x_recon)
    displayer.plot_compare_traffic(data_cleaner.basic_traffic_data, traffic_f,plot_path="data_orly/figures/AE_TCN.png")  # noqa: E501
    print(data_cleaner.first_n_flight_delta_time(traffic_f))
    filtered_traffic  = filter_outlier(traffic_f)
    displayer.plot_compare_traffic(data_cleaner.basic_traffic_data, filtered_traffic,plot_path="data_orly/figures/AE_TCN_filtered.png")  # noqa: E501
    traffic_f.data.to_pickle('data_orly/generated_traff/reproducted/AE_TCN_reproducted_traff.pkl')
    return 0


if __name__ == "__main__":
    main()
