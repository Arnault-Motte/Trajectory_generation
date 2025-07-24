import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

import argparse
import pickle

from src.data_process import Data_cleaner
from src.evaluation import compute_distances
from src.simulation import Simulator
from src.test_display import plot_DTW_SSPD
from traffic.core import Traffic


def select_dist(traff: str, dis: dict, chosen_labels: list) -> list[dict]:
    data = Data_cleaner(file_name=traff)
    list_dist = []
    for label in chosen_labels:
        flight_ids = data.return_flight_id_for_label(label)
        print(flight_ids[:20])
        print(len(flight_ids))
        dic_dis = {
            dist: {key: 0.0 for key in dis[dist].keys()}
            for dist in flight_ids
            if dist in dis.keys()
        }
        for f_id in flight_ids:
            if f_id in dis.keys():
                for key in list(dis.values())[0].keys():
                    dic_dis[f_id][key] = dis[f_id][key]
        list_dist.append(dic_dis)
    return list_dist


def main() -> None:
    parser = argparse.ArgumentParser(description="DTW SSPD")

    parser.add_argument(
        "--og_traff",
        type=str,
        default="data/takeoffs_LFPO_07.pkl",
        help="Path of the of traffic file",
    )

    parser.add_argument(
        "--true_dist",
        type=str,
        default="results/distances/d_TO_7_Real",
        help="Path of the distance file for the whole dataset",
    )

    parser.add_argument(
        "--typecode",
        type=str,
        default="",
        help="Typecode to be considered",
    )

    parser.add_argument(
        "--file_paths",
        type=str,
        nargs="+",
        default=[],
        help="List of distance files to be ploted",
    )

    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=[],
        help="Label for each distance",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="File to save the fig",
    )

    args = parser.parse_args()

    all_true_dist = args.true_dist
    og_traff_path = args.og_traff
    chosen_labels = [args.typecode]
    file_paths = args.file_paths
    labels = ["REFERENCE"] + args.labels

    with open(all_true_dist, "rb") as f:
        og_distance = pickle.load(f)

    og_dists = select_dist(
        og_traff_path, og_distance, chosen_labels=chosen_labels
    )
    # print(og_dists)
    print(og_dists[0][next(iter(og_dists[0]))].keys())
    distances_list = [og_dists[0]]
    for file_name in file_paths:
        with open(file_name, "rb") as f:
            distance = pickle.load(f)
            distances_list.append(distance)
    plot_DTW_SSPD(
        distances_list,
        labels,
        path=args.save_path,
    )

if __name__ == "__main__":
    main()
