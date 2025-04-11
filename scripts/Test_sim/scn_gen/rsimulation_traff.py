import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(
    os.path.abspath(CURRENT_PATH)
)

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

from data_orly.src.simulation import Simulator
from data_orly.src.generation.test_display import plot_traffic
from traffic.core import Traffic
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.generation import Generator
from data_orly.src.generation.data_process import Data_cleaner
import argparse
import torch


def main():

    parser = argparse.ArgumentParser(description="files for the test")

    parser.add_argument("--weight_file", type=str, default="weight.pth", help="Path of the weight file")
    parser.add_argument("--scene_file", type=str, default="sim.scn", help="Path of the scene file")
    parser.add_argument("--traffic_file", type=str, default="traff.pkl", help="Path of the traffic file")

    args = parser.parse_args()
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)

    data_cleaner = Data_cleaner(
        "data_orly/data/takeoffs_LFPO_07.pkl",
        columns=["track", "vertical_rate", "groundspeed", "timedelta"],
        chosen_typecodes=["A21N"],
    )
    
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


    model.load_model(weight_file_path=args.weight_file)

    gen = Generator(None,data_cleaner,model_v= model)

    t = gen.generate_n_flight(2000)

    t = Traffic.from_file("data_orly/data/takeoffs_LFPO_07.pkl")
    s = Simulator(t)
    path = "/home/arnault/traffic/data_orly/sim_logs/MY_LOG_LFPO7TRAF_20250407_09-14-10.log"
    flight_ids = "/home/arnault/traffic/data_orly/scn/LFPO_7_tst_denied_flight.pkl"
    f_traff = s.read_csv_log_file(path,flight_ids_paths=flight_ids)
    print(f_traff)
    path_plot = "/home/arnault/traffic/data_orly/figures/random/test.png"
    plot_traffic(f_traff,path_plot)
    f_traff.to_pickle('/home/arnault/traffic/data_orly/results/simulated_traff/simulated_lfpo7.pkl')