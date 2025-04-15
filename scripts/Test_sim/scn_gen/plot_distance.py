import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

from data_orly.src.simulation import Simulator
from data_orly.src.generation.test_display import plot_DTW_SSPD
from data_orly.src.generation.evaluation import compute_distances
from traffic.core import Traffic
import pickle
from data_orly.src.generation.data_process import Data_cleaner


def select_dist(traff: str, dis: dict, chosen_labels: list) -> list[dict]:
    data = Data_cleaner(file_name=traff)
    list_dist = []
    for label in chosen_labels:
        flight_ids = data.return_flight_id_for_label(label)
        print(len(flight_ids))
        dic_dis = {dist: {key:0.0 for key in dis[dist].keys()} for dist in flight_ids if dist in dis.keys()}
        for f_id in flight_ids:
            if f_id in dis.keys():
                for key in list(dis.values())[0].keys():
                    dic_dis[f_id][key] = dis[f_id][key]
        list_dist.append(dic_dis)
    return list_dist


with open(
    "data_orly/results/distances/d_TO_7_Real", "rb"
) as f:
    og_distance = pickle.load(f)

og_traff_path = "data_orly/data/takeoffs_LFPO_07.pkl"
chosen_labels = ["A21N", "B738"]

og_dists = select_dist(og_traff_path, og_distance, chosen_labels=chosen_labels)

print(og_dists[0][next(iter(og_dists[0]))].keys())
# A21N
distances_list = [og_dists[0]]
labels = ["REFERENCE", "VAE", "CVAE"]
file_paths = [
    "data_orly/results/distances/A21N_solo.pkl",
    "data_orly/results/distances/A21N_duo.pkl",
]

for file_name in file_paths:
    with open(
        file_name, "rb"
    ) as f:
        distance = pickle.load(f)
        distances_list.append(distance)

plot_DTW_SSPD(
    distances_list,
    labels,
    path="/home/arnault/traffic/data_orly/figures/distances/ref_A21N.png",
)

# B738
distances_list = [og_dists[1]]
labels = ["REFERENCE", "VAE", "CVAE"]


file_paths = [
    "data_orly/results/distances/B738_solo.pkl",
    "data_orly/results/distances/B738_duo.pkl",
]

for file_name in file_paths:
    with open(
        file_name, "rb"
    ) as f:
        distance = pickle.load(f)
        distances_list.append(distance)

plot_DTW_SSPD(
    distances_list,
    labels,
    path="/home/arnault/traffic/data_orly/figures/distances/ref_B738.png",
)
