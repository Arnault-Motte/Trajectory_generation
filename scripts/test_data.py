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
    displayer = Displayer()
    data_cleaner = Data_cleaner("data_orly/landings_LFPO_06.pkl")
    print("data og \n",data_cleaner.basic_traffic_data.data[data_cleaner.columns + ["latitude","longitude"]].head())
    data = data_cleaner.clean_data()
    print("data cleaned \n",data)

    mean_end = data_cleaner.get_mean_end_point()
    coordinates = {"latitude": mean_end[0], "longitude": mean_end[1]}
    print("point moyen",mean_end)


    # print(x_recon.shape)
    # x_recon = x_recon.cpu().detach().numpy().reshape(-1, 4)
   # 

    x_recon = data_cleaner.unscale(data)
    x_recon = x_recon.reshape(-1, 4)
    print(x_recon.shape)
    x_df = pd.DataFrame(x_recon, columns=data_cleaner.columns)

    x_df["timestamp"] = datetime.datetime(2020, 5, 17) + x_df[  # noqa: DTZ001
        "timedelta"
    ].apply(lambda x: datetime.timedelta(seconds=x))
    x_df["icao24"] = [str(i // 200) for i in range(len(x_df))]
    print("x_df \n",x_df.head())
    x_df = x_df[:500*200]
    print("num flight ", data_cleaner.n_traj)
    x_df = compute_latlon_from_trackgs(
        x_df,
        n_samples=len(x_df)//200,
        n_obs=200,
        coordinates=coordinates,
        forward=False,
    )
    traffic_f = Traffic(x_df)
    print("final_df \n",traffic_f.data.head())
    displayer.plot_compare_traffic(data_cleaner.basic_traffic_data, traffic_f,n_trajectories=1)
    return 0

if __name__ == "__main__":
    main()