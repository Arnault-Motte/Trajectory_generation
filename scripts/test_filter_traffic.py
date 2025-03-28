import sys  # noqa: I001
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from data_orly.src.generation.data_process import (
    Data_cleaner,
    filter_outlier,
    filter_smoothness,
)
from data_orly.src.generation.models.AE_FCL import *  # noqa: F403
from data_orly.src.generation.test_display import Displayer


def main() -> int:
    ## Getting the data
    displayer = Displayer()
    data_cleaner = Data_cleaner(
        "data_orly/generated_traff/reproducted/VAE_FCL_reproducted_traff.pkl"
    )
    traff = data_cleaner.basic_traffic_data
    new_traffic = filter_smoothness(traff,n_segment=150)
    print(len(new_traffic))

    displayer.plot_compare_traffic(
        new_traffic, traff, plot_path="data_orly/figures/test_filter.png"
    )


    return 0


if __name__ == "__main__":
    main()
