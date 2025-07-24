import sys  # noqa: I001
import os



current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartes.crs import Lambert93
from src.test_display import plot_traffic
from tqdm.autonotebook import tqdm
from traffic.core import Flight, Traffic

# traff = Traffic.from_file("src/generation/saved_traff/VAE_TCN_Vampprior_take_off_7_alt_cond_A21N_150_A21N__800_2000.pkl")
# plot_traffic(traff,'traff_test_1.png')
file = "/data/data/arnault/data/final_data/LFPO_all_A320_010.pkl"
traff = Traffic.from_file(file)
print(len(traff))
# traff = traff.aircraft_data()
# print(traff.)
print(traff.data["typecode"].iloc[0::200].value_counts())
# plot_traffic(traff,'A321_151_both.png')

# traff = Traffic.from_file("src/generation/saved_traff/VAE_A321_151_A321__800_2000.pkl")
# plot_traffic(traff,'A321_151.png')