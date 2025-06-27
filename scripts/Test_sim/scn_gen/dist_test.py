import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

from data_orly.src.simulation import Simulator
from data_orly.src.generation.test_display import plot_traffic
from data_orly.src.generation.evaluation import compute_distances
from data_orly.src.generation.data_process import Data_cleaner
from traffic.core import Traffic
import pickle
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--og_traff",
        type=str,
        default="data_orly/data/takeoffs_LFPO_07.pkl",
        help="Path of the of traffic file",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="/home/arnault/traffic/data_orly/sim_logs/MY_LOG_LFPO7TRAF_20250407_10-19-14.log",
        help="Path of the log file",
    )
    parser.add_argument(
        "--ignored",
        type=str,
        default="/home/arnault/traffic/data_orly/scn/LFPO_7_tst_denied_flight.pkl",
        help="Path of the file containing all the fligth ids to ignore",
    )
    parser.add_argument(
        "--saved_name",
        type=str,
        default="d_TO_7_Real_more_points.pkl",
        help="name of the saved file",
    )

    parser.add_argument(
        "--typecodes",
        type=str,
        nargs="+",
        default=[],
        help="typecodes to be considered in the og traff",
    )

    parser.add_argument(
        "--vertical_rate",
        type=int,
        default=0,
        help="does the column contains teh vertical rate",
    )


    args = parser.parse_args()

    columns = ["track","groundspeed","timedelta"]
    columns += ["vertical_rate"] if args.vertical_rate else ["altitude"]
    print(args.typecodes)
#
    #data_clean = Data_cleaner(args.og_traff,chosen_typecodes=args.typecodes,columns=columns,aircraft_data=True)



    t = Traffic.from_file(args.og_traff)
    print(t.data.head(4))
    print(t.data["flight_id"].value_counts())
    

    


    print(len(t),t.data.head(4))
    s = Simulator(t)
    f_traff = s.read_csv_log_file(args.log_file, flight_ids_paths=args.ignored)
    print([f.flight_id for f in f_traff[:10]])

    distances = compute_distances(t, f_traff, 50)
    print(len(distances))
    with open(
        "data_orly/results/distances/" + args.saved_name,
        "wb",
    ) as file:
        pickle.dump(distances, file)

if __name__ == "__main__":
    main()