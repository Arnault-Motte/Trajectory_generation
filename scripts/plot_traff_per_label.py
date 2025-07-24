import sys  # noqa: I001
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import datetime

import pandas as pd
from src.data_process import Data_cleaner
from src.models.AE_FCL import *  # noqa: F403
from src.test_display import display_traffic_per_typecode
from traffic.algorithms.generation import compute_latlon_from_trackgs
from traffic.core import Traffic

def main() -> int:
    print(sys.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: F405
    print(device)
    ## Getting the data

    data_cleaner = Data_cleaner("/data/data/arnault/data/final_data/TO_LFPO_final.pkl")

    display_traffic_per_typecode(data_cleaner,"data_orly/figures/og_data/traffic_per_typecode_n_data.png")


    return 0

if __name__ == "__main__":
    main()