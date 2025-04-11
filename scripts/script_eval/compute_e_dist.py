import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(
    os.path.abspath(CURRENT_PATH)
)

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

from data_orly.src.generation.evaluation import Evaluator
from data_orly.src.generation.generation import Generator
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.data_process import Data_cleaner
from traffic.core import Traffic
import torch

def main()->None:


    data_cleaner = Data_cleaner(
        "data_orly/data/takeoffs_LFPO_07.pkl",
        columns=["track", "altitude", "groundspeed", "timedelta"],
        chosen_typecodes=["B738"],
    )
    data = data_cleaner.clean_data()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    seq_len = 200
    in_channels = 4
    output_channels = 64
    latent_dim = 64
    pooling_factor = 10
    stride = 1
    number_of_block = 4
    kernel_size = 16
    dilatation = 2
    dropout = 0.2
    pseudo_input_num = 800 #*1.5
    patience = 30
    min_delta = -100
    labels_latent = 16

    model= VAE_TCN_Vamp(
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
        init_std=0.3,
    ).to(device)

    model.load_model('data_orly/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_alt_cond_B738.pth')

    gen = Generator(model,data_cleaner)
    ev = Evaluator(gen,data_cleaner.basic_traffic_data)

    print(ev.compute_e_dist())

if __name__ == "__main__":
    main()
