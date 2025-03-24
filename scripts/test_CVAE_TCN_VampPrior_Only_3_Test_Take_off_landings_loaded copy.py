import sys  # noqa: I001
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


import torch

from data_orly.src.generation.data_process import Data_cleaner
from data_orly.src.generation.models.CVAE_TCN_VampPrior import *  # noqa: F403
from data_orly.src.generation.test_display import Displayer


def main() -> int:
    print(sys.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data

    data_cleaner = Data_cleaner(
        file_names=[
            "data_orly/data/takeoffs_LFPO_07.pkl",
            "data_orly/data/landings_LFPO_06.pkl",
        ]
    )
    displayer = Displayer(data_cleaner)
    data = data_cleaner.clean_data_several_datasets()
    labels = data_cleaner.return_labels_datasets()
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
    labels_latent = 4
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
        temp_save="best_model_2.pth",
    ).to(device)

    ## Training the model
    model.load_model("data_orly/src/generation/models/saved_weights/CVAE_TCN_Vampprior_take_off_and_landings.pth")


    ## Testing reconstuction on one batch

    x_recon, data_2 = model.reproduce_data(data, labels, 500, 2)
    print(x_recon.shape, "\n")
    traffic_init = data_cleaner.dataloader_traffic_converter(data_2, 2)
    ordered_labels = [li for _, label in data_2 for li in label.tolist()]
    ordered_labels= data_cleaner.one_hot.inverse_transform(ordered_labels)
    #print(ordered_labels)

    traffic_f = data_cleaner.output_converter(x_recon)
    displayer.plot_compare_traffic_hue(
        traffic_init,
        generated_traffic=traffic_f,
        labels_hue= ordered_labels,
        plot_path="data_orly/figures/recons/CVAE_TCN_vamp_Recons_take_off_landing.png",
    )
    traffic_f.data.to_pickle(
        "data_orly/generated_traff/reproducted/CAE_TCN_Vamp_reproducted_traff_take_off_and landings.pkl"
    )

    return 0


if __name__ == "__main__":
    main()
