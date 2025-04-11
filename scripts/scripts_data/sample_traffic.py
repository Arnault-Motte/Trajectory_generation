import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))

print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


from traffic.core import Traffic
from data_orly.src.generation.data_process import (
    save_smaller_traffic,
    Data_cleaner,
)
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="files for the test")

    parser.add_argument(
        "--traff_file", type=str, default="", help="original traffic file"
    )
    parser.add_argument(
        "--samp_traff",
        type=str,
        default="",
        help="Final traffic files",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default="",
        help="Defines the number of flight in each sampled traff",
    )
    parser.add_argument(
        "--typecodes",
        type=str,
        nargs="+",
        default="",
        help="typecodes to be considered",
    )

    args = parser.parse_args()
    print("la")
    # if path is not given we create it
    f_path = args.samp_traff
    og_path = args.traff_file
    if f_path == "":
        name = f"{og_path.split('/')[-1].split('.')[0]}"
        f_path = f"data_orly/data/sampled_data/{name}"
        for label in args.typecodes:
            f_path += f"_{label}"
        f_path += ".pkl"

    print(f_path)
    print(args.typecodes)
    col = Traffic.from_file(args.traff_file).data.columns
    clean = Data_cleaner(
        file_name=args.traff_file, chosen_typecodes=args.typecodes, columns=col
    )
    print(len(clean.basic_traffic_data))

    save_smaller_traffic(clean.basic_traffic_data, f_path, args.sizes)

    return


if __name__ == "__main__":
    main()
