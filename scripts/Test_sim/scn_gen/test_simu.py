import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

print(CURRENT_PATH)


from data_orly.src.simulation import Simulator, launch_simulation
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="files for the test")

    parser.add_argument("--scn", type=str, default="", help="scn file path")
    parser.add_argument("--logs", type=str, default="", help="log file path")
    parser.add_argument(
        "--log_t_step",
        type=int,
        default=3,
        help="Timedeltas of the data to be logged",
    )

    args = parser.parse_args()

    log_file = args.logs
    scn_file = args.scn
    if log_file == "":
        log_file = (
            "/home/arnault/traffic/data_orly/sim_logs/"
            + scn_file.split("/")[-1].split(".")[0]
            + ".log"
        )

    launch_simulation(scn_file, log_file, args.log_t_step)
