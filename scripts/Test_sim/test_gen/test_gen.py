import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp, get_data_loader as get_data_loader1
from data_orly.src.simulation import Simulator
from data_orly.src.generation.test_display import plot_traffic
from traffic.core import Traffic
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp,get_data_loader
from data_orly.src.generation.generation import Generator
from data_orly.src.generation.data_process import Data_cleaner
import argparse
import torch


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)

    data_cleaner1 = Data_cleaner(
        "data_orly/data/takeoffs_LFPO_07.pkl",
        columns=["track", "vertical_rate", "groundspeed", "timedelta"],
        chosen_typecodes=["B738","A21N"],
    )
    data_cleaner = Data_cleaner(
        "data_orly/data/takeoffs_LFPO_07.pkl",
        columns=["track", "vertical_rate", "groundspeed", "timedelta"],
        chosen_typecodes=["B738"],
    )
    data = data_cleaner.clean_data()
    data1 = data_cleaner1.clean_data()
    print("start")

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


    labels1 = data_cleaner1.return_labels()
    labels_dim1 = labels1.shape[1]


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

    model1 = CVAE_TCN_Vamp(
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
            label_dim=labels_dim1,
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
    
    model1.load_model("/home/arnault/traffic/data_orly/src/generation/models/saved_weights/CVAE_TCN_Vampprior_take_off_7_vr_direct_cond_B738_A21N.pth")
    model.load_model("/home/arnault/traffic/data_orly/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_vr_direct_cond_B738.pth")


    _, data_loader_val = get_data_loader(
            data, batch_size=500
        )
    _, data_loader_val1 = get_data_loader1(
            data1, batch_size=500, num_worker=6,labels = labels1
        )
    print([ val// len(data_loader_val) for val in model.compute_val_loss(data_loader_val)])
    print([ val// len(data_loader_val1) for val in model1.compute_val_loss(data_loader_val1)])

main()