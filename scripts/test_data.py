import sys  # noqa: I001
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import datetime

import pandas as pd
from data_orly.src.generation.data_process import Data_cleaner
from data_orly.src.generation.models.AE_FCL import *  # noqa: F403
from data_orly.src.generation.test_display import Displayer
from traffic.algorithms.generation import compute_latlon_from_trackgs
from traffic.core import Traffic


def main() -> int:
    print(sys.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data

    data_cleaner = Data_cleaner("data_orly/data/takeoffs_LFPO_07.pkl",airplane_types_num=3)
    displayer = Displayer(data_clean=data_cleaner)
    
    displayer.plot_distribution_typecode('/home/arnault/traffic/data_orly/figures/airplane_models_distrib_stacked_take_off_7_3.png','/home/arnault/traffic/data_orly/figures/airplane_models_distrib_take_off_7_3.png',hist=True)


    return 0

if __name__ == "__main__":
    main()