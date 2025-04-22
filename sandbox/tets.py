import sys  # noqa: I001
import os



current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import matplotlib.pyplot as plt
from cartes.crs import Lambert93
from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd
from data_orly.src.generation.test_display import plot_traffic
from traffic.core import Flight, Traffic

# traff = Traffic.from_file("data_orly/src/generation/saved_traff/VAE_TCN_Vampprior_take_off_7_alt_cond_A21N_150_A21N__800_2000.pkl")
# plot_traffic(traff,'data_orly/traff_test_1.png')

# traff = Traffic.from_file("data_orly/src/generation/saved_traff/CVAE_both_A321_A21N_200_A321_A21N_A21N_800_2000.pkl")
# plot_traffic(traff,'data_orly/traff_test_2.png')

traff = Traffic.from_file("data_orly/src/generation/saved_traff/CVAE_both_A321_A21N_200_A321_A21N_A321_800_2000.pkl")
plot_traffic(traff,'data_orly/traff_test_3.png')