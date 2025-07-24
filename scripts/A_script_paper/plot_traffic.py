import sys  # noqa: I001
import os

from traffic.core.flight import Flight


current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.generation.data_process import (
    Data_cleaner,
    return_traff_per_typecode,
    filter_missing_values,
    compute_time_delta,
)
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.VAE_ONNX import VAE_ONNX
from data_orly.src.generation.models.CVAE_ONNX import CVAE_ONNX
from data_orly.src.generation.models.VAE_TCN_VampPrior import *  # noqa: F403
from data_orly.src.generation.test_display import plot_distribution_typecode
from traffic.core import Traffic
from data_orly.src.generation.generation import ONNX_Generator
from tqdm import tqdm

from data_orly.src.generation.test_display import plot_traffic


def gen_per_vae(
    list_vae: list[str], typecodes: list[str], n: int, batch_size: int = 500
) -> Traffic:
    """
    Generates the traffic composed of n_flights for each vae model path in the list.
    list_vae is a list of the onnx dir for each vae.
    typecodes is the list of the typecodes associated with each vae.
    """
    traffs = []
    for vae_path in list_vae:
        data_cleaner = Data_cleaner(no_data=True)
        data_cleaner.load_scalers(vae_path + "/scalers.pkl")
        model = VAE_ONNX(vae_path)
        gen = ONNX_Generator(model, data_cleaner)
        t = gen.generate_n_flight(n, batch_size, lat_long=True)
        traffs.append(t)

    vae_traffs: Traffic = sum(traffs)

    tc = [typecodes[i // (n * 200)] for i in range(n * len(typecodes) * 200)]
    icao24 = [i // 200 for i in range(n * len(typecodes) * 200)]
    callsign = icao24

    vae_traffs = vae_traffs.assign(typecode=tc)
    vae_traffs = vae_traffs.assign(icao24=icao24)
    vae_traffs = vae_traffs.assign(callsign=callsign)
    vae_traffs = vae_traffs.assign_id().eval(
        desc="assigning ids", max_workers=4
    )

    return vae_traffs


import matplotlib.pyplot as plt
from cartes.crs import Lambert93, PlateCarree


def display_traffic_per_typecode(traff: Traffic, path: str) -> None:
    all_traffic = []
    labels = Counter(
        list(
            traff.data[traff.data["typecode"].notnull()]["typecode"].iloc[
                0::200
            ]
        )
    ).most_common(10)
    print(labels)
    for typecode, num_f in labels:
        n_t = traff.query(f"typecode == '{typecode}'")
        all_traffic.append((n_t, num_f))

    all_traffic.sort(key=lambda x: x[1], reverse=True)
    print([l for _, l in all_traffic])

    with plt.style.context("traffic"):
        fig, axes = plt.subplots(
            2,
            5,
            subplot_kw=dict(projection=Lambert93()),
            figsize=(15, 8),
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        for ax, (traff,count), label in tqdm(
            zip(axes, all_traffic, labels), desc="Plotting"
        ):
            traff.plot(ax, alpha=0.5, color="orange")
            ax.set_title(f"Typecode: {label},\n N = {len(traff)}")

        plt.tight_layout()
        plt.savefig(path)
        plt.show()
        


def main() -> None:
    parser = argparse.ArgumentParser(description="Training a CVAE")

    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to acess the data",
    )

    parser.add_argument(
        "--onnx_dir",
        type=str,
        default="",
        help="Path to acess the data",
    )

    parser.add_argument(
        "--onnx_vaes",
        nargs="+",
        type=str,
        default=[],
        help="Path to acess the data",
    )

    parser.add_argument(
        "--plot_path",
        type=str,
        default="",
        help="Path to acess the data",
    )

    parser.add_argument(
        "--typecodes",
        nargs="+",
        type=str,
        default=[],
        help="typecodes for which to consider the altitude profiles",
    )

    parser.add_argument(
        "--plot_per_typecode",
        type=int,
        default=0,
        help="equal 1 if you want to plot a traff per typecode,else plots a signle traff containing all typecodes",
    )

    args = parser.parse_args()

    def f_good(f: Flight, threshold: int = 1500) -> Flight:
        if (f.data["altitude"] < threshold).any() == 0:  # and (
            #     f.data[f.data["altitude"] < 20000]["altitude"].diff().max() < 2000
            # ):
            return f
        return None

    n = 1000
    if args.data == "":
        if args.onnx_dir != "":
            data_cleaner = Data_cleaner(
                no_data=True
            )  # saves scalers, and scales and modifies data, allows to interpret the model outputs
            data_cleaner.load_scalers(args.onnx_dir + "/scalers.pkl")
            cvae_onnx = CVAE_ONNX(args.onnx_dir)  # the CVAE
            onnx_gen = ONNX_Generator(
                cvae_onnx, data_cleaner
            )  # used to gen from the CVAE

            traffs = onnx_gen.generate_n_flight_per_labels(
                args.typecodes, n
            )  # to modify to not have several time the same _id

            f_traf: Traffic = sum(traffs)

            tc = [
                args.typecodes[i // (n * 200)]
                for i in range(n * len(args.typecodes) * 200)
            ]
            icao24 = [i // 200 for i in range(n * len(args.typecodes) * 200)]
            callsign = icao24

            f_traf = f_traf.assign(typecode=tc)
            f_traf = f_traf.assign(icao24=icao24)
            f_traf = f_traf.assign(callsign=callsign)
            

            f_traf = f_traf.assign_id().eval(
                desc="assigning ids", max_workers=4
            )
            filtered = f_traf.pipe(f_good).eval(desc="removing outliers")
            if not args.plot_per_typecode:
                plot_traffic(f_traf, args.plot_path.split(".")[0] + "cvae.png")
                plot_traffic(
                    filtered, args.plot_path.split(".")[0] + "cvae_filtered.png"
                )
            else:
                display_traffic_per_typecode(
                    f_traf, args.plot_path.split(".")[0] + "cvae.png"
                )
                display_traffic_per_typecode(
                    filtered, args.plot_path.split(".")[0] + "cvae_filtered.png"
                )

        if len(args.onnx_vaes) != 0:
            traff = gen_per_vae(args.onnx_vaes, args.typecodes, n)

            filtered = traff.pipe(f_good).eval(desc="removing outliers")
            if not args.plot_per_typecode:
                plot_traffic(traff, args.plot_path.split(".")[0] + "vae.png")
                plot_traffic(
                    filtered, args.plot_path.split(".")[0] + "vae_filtered.png"
                )
            else:
                display_traffic_per_typecode(
                    traff, args.plot_path.split(".")[0] + "vae.png"
                )
                display_traffic_per_typecode(
                    filtered, args.plot_path.split(".")[0] + "vae_filtered.png"
                )
            print(len(traff))
    else:
        traff = Traffic.from_file(args.data)
        plot_traffic(traff, args.plot_path)


if __name__ == "__main__":
    main()
