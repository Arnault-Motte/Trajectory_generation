import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

import argparse
import pickle

from data_orly.src.generation.test_display import vertical_rate_profile,vertical_rate_profile_2
from traffic.core import Traffic

traff = Traffic.from_file(filename="/data/data/arnault/data/takeoffs_LFPO_07.pkl")

traff = traff.sample(1000)

vertical_rate_profile_2(traff,'data_orly/figures/alt_profile/old_data.png',distance=True)