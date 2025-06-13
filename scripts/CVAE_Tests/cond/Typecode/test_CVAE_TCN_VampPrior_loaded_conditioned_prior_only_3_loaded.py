import sys  # noqa: I001
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import torch

from data_orly.src.generation.data_process import Data_cleaner
from data_orly.src.generation.models.CVAE_TCN_VampPrior import *  # noqa: F403
from data_orly.src.generation.test_display import Displayer

# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# )


def main() -> int:
    print(sys.path)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data

    data_cleaner = Data_cleaner(
        "data_orly/data/takeoffs_LFPO_07.pkl", airplane_types_num=3
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
        conditioned_prior=True,
    ).to(device)

    ## Training the model
    model.load_model(
        "data_orly/src/generation/models/saved_weights/CVAE_TCN_Vampprior_take_off_7_3_airplane_types_cond.pth"
    )

    ## Testing reconstuction on one batch

    # x_recon,data_2 =  model.reproduce_data(data,labels, 500, 2)
    # print(x_recon.shape, "\n")
    # traffic_init = data_cleaner.dataloader_traffic_converter(data_2,2)

    # traffic_f = data_cleaner.output_converter(x_recon)
    # displayer.plot_compare_traffic(traffic_init, traffic_f,plot_path="data_orly/figures/CVAE_TCN_vamp_Recons.png")  # noqa: E501
    # traffic_f.data.to_pickle('data_orly/src/generation/models/saved_temp/CVAE_TCN_Vampprior_take_off_7.pth')

    # displayer.plot_latent_space_top10_labels(2000,model,'data_orly/figures/CVAE_TCN_vamp_Latent_space.png')
    # displayer.plot_distribution_typecode_label_generation("data_orly/figures/vertical_rates/CVAE_TCN_vamp_Take_off_7_condi_n1.png","data_orly/figures/vertical_rates/CVAE_TCN_vamp_Take_off_7_condi_n2.png",model,2000,True,(0,4000))
    # displayer.display_pseudo_inputs(model,"data_orly/figures/pseudo_inputs_traj/CVAE_TCN_vamp_TO_7_condi.png")
    # displayer.plot_vamp_generated(model,"data_orly/figures/pseudo_inputs_traj/CVAE_TCN_vamp_TO_7_chosen_condi.png",1,100)
    # displayer.plot_latent_spaces_labels(model,"data_orly/figures/latent_spaces/CVAE_TCN_Vamp_Cond_TO_7.png")
    displayer.plot_distribution_typecode_label_generation(
        "data_orly/figures/vertical_rates_recons/CVAE_TCN_vamp_Recons_take_off_7_3_airplane_types_1_cond_only3.png",
        "data_orly/figures/vertical_rates_recons/CVAE_TCN_vamp_Recons_take_off_7_3_airplane_types_2_cond_only3.png",
        model,
        hist=True,
        bounds=(0, 4000),
    )
    return 0


if __name__ == "__main__":
    main()
