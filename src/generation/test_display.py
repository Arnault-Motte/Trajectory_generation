import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from cartes.atlas import france
from cartes.crs import Lambert93, PlateCarree
from cartes.osm import Nominatim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.distributions import Independent, Normal
from tqdm import tqdm
import altair as alt
import pandas as pd

import numpy as np
from data_orly.src.generation.data_process import (
    Data_cleaner,
    compute_vertical_rate,
    jason_shanon,
)
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_TCN_VampPrior import (
    get_data_loader as get_data_loader_labels,
)
from data_orly.src.generation.models.VAE_TCN_VampPrior import (
    VAE_TCN_Vamp,
    get_data_loader,
)
from traffic.core import Flight, Traffic
from pitot import aero


###TEST for seeing the reconstruction of the autoencoder
def plot_traffic(
    traffic: Traffic,
    plot_path: str,
    background: bool = True,
) -> None:
    """Plots the traffic in entry"""
    if background:
        # background elements
        paris_area = france.data.query("ID_1 == 1000")
        seine_river = Nominatim.search(
            "Seine river, France"
        ).shape.intersection(paris_area.union_all().buffer(0.1))

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
        traffic.plot(ax, alpha=0.2, color="blue")
        plt.savefig(plot_path, dpi=400)
        plt.show()


def plot_DTW_SSPD(distances: list[dict], labels: list[str], path: str) -> None:
    """Plots the DTW and SSPD distances passed in entry.
    These distances are stored each set of distances is associated to a label in labels (VAE,CVAE etc...)
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    # DTW

    for label, dist in zip(labels, distances):
        sorted_d = np.sort([d["dtw"] for d in dist.values()])
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        axes[0].plot(sorted_d, cdf, label=label)

    axes[0].set_title("DTW")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Cumulative probability")
    axes[0].legend(title="Generation method")
    axes[0].grid(True)
    axes[0].set_xlim(0, 1200000)

    for label, dist in zip(labels, distances):
        sorted_d = np.sort([d["sspd"] for d in dist.values()])
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        axes[1].plot(sorted_d, cdf, label=label)

    axes[1].set_title("SSPD")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].legend(title="Generation method")
    axes[1].grid(True)
    axes[1].set_xlim(0, 16000)

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_distribution_typecode(
    data: Traffic, path1: str, path2: str, hist: bool = False
) -> None:
    """Plots the distribution of the vertical rate for every typecode in the dataset.
    They are all shown on a single graph saved in path1, and on singular graphs in path2 (.png)"""
    if hist:
        path1 = path1.split(".")[0] + "_hist.png"
        path2 = path2.split(".")[0] + "_hist.png"

    data = compute_vertical_rate(data)
    typecodes = data.data["typecode"].unique()  # list all typecodes
    map_typecode_vrate = {}
    for typecode in typecodes:
        d = data.query(f"typecode == '{typecode}'")
        print(typecode, "----------------------------------")
        print(len(d))
        l_og = [f.vertical_rate_mean for f in d]
        print("og len: ", len(l_og))
        # print(l_og[:10])
        map_typecode_vrate[typecode] = [el for el in l_og if 4000 > el > 0]
        print("Len after bounding: ", len(map_typecode_vrate[typecode]))

    for typecode, vertical_rates in map_typecode_vrate.items():
        print(vertical_rates)
        if not hist:
            sns.kdeplot(
                vertical_rates, label=typecode, linewidth=2, clip=(0, 4000)
            )
            plt.ylabel("Density")
        else:
            sns.histplot(
                vertical_rates,
                label=typecode,
                bins=30,
                kde=True,
                clip=(0, 4000),
            )
            plt.ylabel("Count")

    plt.xlabel("Vertical Rate")
    plt.title("Vertical Rate Distribution by Aircraft Typecode")
    plt.legend(title="Typecode")
    plt.savefig(path1)
    plt.show

    # Define number of rows and columns
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))  # Adjust figure size

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate over each typecode and corresponding subplot
    for ax, (typecode, vertical_rates) in zip(axes, map_typecode_vrate.items()):
        if not hist:
            sns.kdeplot(vertical_rates, ax=ax, linewidth=2, clip=(0, 4000))
            ax.set_ylabel("Density")
        else:
            sns.histplot(
                vertical_rates,
                ax=ax,
                label=typecode,
                bins=30,
                kde=True,
                clip=(0, 4000),
            )
            ax.set_ylabel("Count")
        ax.set_title(f"Typecode: {typecode}")
        ax.set_xlabel("Vertical Rate")

    # Remove any empty subplots (if less than 10 typecodes)
    for i in range(len(map_typecode_vrate), rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(path2)
    plt.show()


def plot_compare_traffic(
    traffic: Traffic,
    generated_traffic: Traffic,
    n_trajectories: int = None,
    background: bool = True,
    plot_path: str = "data_orly/plot.png",
) -> int:
    """Plots two traffic side by side"""
    if not n_trajectories:
        n_trajectories = len(generated_traffic)

    if background:
        # background elements
        paris_area = france.data.query("ID_1 == 1000")
        seine_river = Nominatim.search(
            "Seine river, France"
        ).shape.intersection(paris_area.union_all().buffer(0.1))

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=Lambert93()))
        traffic[: min(n_trajectories, len(traffic))].plot(ax[0], alpha=0.7)
        generated_traffic[: min(n_trajectories, len(generated_traffic))].plot(
            ax[1], alpha=0.7
        )
        plt.savefig(plot_path)
    return 1


def plot_latent_space(
    data_clean: np.ndarray, num_point: int, model: VAE_TCN_Vamp, plt_path: str
) -> int:
    """
    Plots the latent space of the VAE model in entry
    """
    data = data_clean
    train, _ = get_data_loader(data, 500, 0.8)
    list_tensor = [batch[0] for batch in train]
    all_tensor = torch.concat(list_tensor).to(next(model.parameters()).device)

    all_tensor = all_tensor.permute(0, 2, 1)

    ## getting mu and sigma
    mu, log_var = model.encoder(all_tensor)
    scale = (log_var / 2).exp()
    # get the actual distribution
    distribution = Independent(Normal(mu, scale), 1)

    posterior_samples = distribution.rsample()

    prior_sample = model.sample_from_prior(num_point).squeeze(1)
    print(f"posterior shape {posterior_samples.shape}")
    print(f"prior shape : {prior_sample.shape}")

    total = np.concat(
        (
            posterior_samples.cpu().detach().numpy(),
            prior_sample.cpu().detach().numpy(),
        ),
        axis=0,
    )

    pca = PCA(n_components=2).fit(total[:-num_point])
    embeded = pca.transform(total)

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, figsize=(15, 10))

        ax.scatter(
            embeded[:-num_point, 0],
            embeded[:-num_point, 1],
            s=4,
            c="grey",
            label="True",
        )
        ax.scatter(
            embeded[-num_point:, 0],
            embeded[-num_point:, 1],
            s=4,
            c="blue",
            label="Generated",
        )
        ax.title.set_text("Latent Space")
        ax.legend()

        plt.savefig(plt_path)

    return 0


def plot_latent_space_pseudo_inputs_selected(
    data_clean: np.ndarray, num_point: int, model: VAE_TCN_Vamp, plt_path: str
) -> int:
    """
    Plots the latent space of the VAE model in entry for the selected vampprior component.
    The plot is saved in plt path. The pseudo input is selected by its num  (num_point).
    Data_clean is the real data to compare the latent spaces.
    """
    data = data_clean
    train, _ = get_data_loader(data, 500, 0.8)
    list_tensor = [batch[0] for batch in train]
    all_tensor = torch.concat(list_tensor).to(next(model.parameters()).device)

    all_tensor = all_tensor.permute(0, 2, 1)
    pseudo_latent, _ = model.pseudo_inputs_latent()
    pseudo_latent = pseudo_latent.cpu().detach().numpy()

    ## getting mu and sigma
    mu, log_var = model.encoder(all_tensor)
    scale = (log_var / 2).exp()
    # get the actual distribution
    distribution = Independent(Normal(mu, scale), 1)

    posterior_samples = distribution.rsample()

    prior_sample = model.sample_from_prior(num_point).squeeze(1)
    print(f"posterior shape {posterior_samples.shape}")
    print(f"prior shape : {prior_sample.shape}")

    total = np.concat(
        (
            posterior_samples.cpu().detach().numpy(),
            prior_sample.cpu().detach().numpy(),
        ),
        axis=0,
    )

    pca = PCA(n_components=2).fit(total[:-num_point])
    embeded = pca.transform(total)
    embedded_psuedo = pca.transform(pseudo_latent)

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, figsize=(15, 10))

        ax.scatter(
            embeded[:-num_point, 0],
            embeded[:-num_point, 1],
            s=4,
            c="grey",
            label="True",
        )
        ax.scatter(
            embeded[-num_point:, 0],
            embeded[-num_point:, 1],
            s=4,
            c="blue",
            label="Generated",
        )

        ax.scatter(
            embedded_psuedo[:, 0],
            embedded_psuedo[:, 1],
            s=4,
            c="orange",
            label="Pseudo Inputs",
        )
        ax.title.set_text("Latent Space")
        ax.legend()

        plt.savefig(plt_path)

    return 0


def plot_latent_space_for_label(
    data_cleaner: Data_cleaner,
    label: str,
    num_point: int,
    model: CVAE_TCN_Vamp,
    plt_path: str,
) -> int:
    """Plots the latent space associated to the selected label of the CVAE"""
    labels_array = np.full(num_point, label).reshape(-1, 1)
    transformed_label = data_cleaner.one_hot.transform(labels_array)

    traff_data = data_cleaner.basic_traffic_data.data
    traffic_curated = Traffic(traff_data[traff_data["typecode"] == label])

    data = data_cleaner.clean_data_specific(traffic_curated)[
        : min(num_point, len(traffic_curated))
    ]

    labels_true = np.array(
        [label for _ in traffic_curated[: min(num_point, len(traffic_curated))]]
    ).reshape(-1, 1)
    transformed_true_labels = self.data_clean.one_hot.transform(labels_true)

    train, _ = get_data_loader_labels(data, transformed_true_labels, 500, 0.8)

    list_tensor = [batch[0] for batch in train]
    all_tensor = torch.concat(list_tensor).to(next(model.parameters()).device)

    list_labels = [labels for _, labels in train]
    all_labels = torch.concat(list_labels).to(next(model.parameters()).device)
    all_tensor = all_tensor.permute(0, 2, 1)

    ## getting mu and sigma
    mu, log_var = model.encode(all_tensor, all_labels)
    scale = (log_var / 2).exp()
    # get the actual distribution
    distribution = Independent(Normal(mu, scale), 1)

    posterior_samples = distribution.rsample()

    prior_sample = model.sample_from_prior(num_point).squeeze(1)
    print(f"posterior shape {posterior_samples.shape}")
    print(f"prior shape : {prior_sample.shape}")

    total = np.concat(
        (
            posterior_samples.cpu().detach().numpy(),
            prior_sample.cpu().detach().numpy(),
        ),
        axis=0,
    )

    pca = PCA(n_components=2).fit(total[:-num_point])
    embeded = pca.transform(total)

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, figsize=(15, 10))

        ax.scatter(
            embeded[:-num_point, 0],
            embeded[:-num_point, 1],
            s=4,
            c="grey",
            label="True",
        )
        ax.scatter(
            embeded[-num_point:, 0],
            embeded[-num_point:, 1],
            s=4,
            c="blue",
            label="Generated",
        )
        ax.title.set_text("Latent Space")
        ax.legend()

        plt.savefig(plt_path)

    return 0


def plot_latent_space_top10_labels(
    data_cleaner: Data_cleaner,
    num_point: int,
    model: CVAE_TCN_Vamp,
    plt_path: str,
) -> int:
    """Plots the latent spaces for the top 10 most commun labels in the data in data cleaner"""
    top10 = data_cleaner.get_top_n_typecodes()
    for label, _ in top10:
        path = plt_path.split(".")[0] + "_" + label + ".png"
        plot_latent_space_for_label(data_cleaner, label, num_point, model, path)
    return 0


def return_label_shanon(
    self, type_code_vrate_gen: dict, type_code_vrate: dict
) -> dict:
    dic_res = {}
    for label, _ in self.type_code_vrate.items():
        dic_res[label] = jason_shanon(
            self.type_code_vrate[label], self.type_code_vrate_gen[label]
        )
    return dic_res


def set_v_rates_per_typecode(
    data_clean: Data_cleaner, chosen_labels: list[str]
) -> None:
    data = data_clean.basic_traffic_data
    map_typecode_vrate = {
        code: [
            flight.vertical_rate_mean
            for flight in data.query(f"typecode in ['{code}']")
            if 0 < flight.vertical_rate_mean < 4000
        ]
        for code in chosen_labels
    }
    type_code_vrate = map_typecode_vrate
    return type_code_vrate


def generated_data_v_rates_to_be_displayed(
    data_clean,
    model: CVAE_TCN_Vamp,
    n_points: int,
    bounds: tuple[float, float],
    labels_to_gen: list[str],
    batch_size: int = 500,
) -> None:
    """
    Generates n_points synthetic trajectory per existing label in the data using the model in entry.
    Then computes the mean Vertical rate for each generated trajectory and saves them in self.
    Use bounds to bound the  Vertical rates between two values (filtering out the extrems).
    Ex bounds = (0,4000)
    """
    map_typecode_vrate = dict()
    print("|--Generating data--|")
    for label in tqdm(labels_to_gen, desc="Generating"):
        with tqdm(total=4, desc=f"Processing {label}") as pbar:
            # print("|--Encoding--|")
            labels_array = np.full(batch_size, label).reshape(-1, 1)
            transformed_label = data_clean.one_hot.transform(labels_array)
            pbar.update(1)

            # print("|--Sampling--|")
            labels_final = torch.Tensor(transformed_label).to(
                next(model.parameters()).device
            )
            sampled = model.sample(n_points, batch_size, labels_final)
            pbar.update(1)

            # print("|--Converting--|")
            traf = data_clean.output_converter(sampled)
            pbar.update(1)

            # print("|--Vertical Rate Computation--|")
            traf = compute_vertical_rate(traf)
            print(traf[0].data["vertical_rate"])
            print(f"rate mean : {traf[0].vertical_rate_mean}")
            map_typecode_vrate[label] = [
                flight.vertical_rate_mean
                for flight in traf
                if bounds[0] <= flight.vertical_rate_mean <= bounds[1]
            ]
            pbar.update(1)
    type_code_vrate_gen = map_typecode_vrate
    return type_code_vrate_gen


def display_pseudo_inputs(
    data_clean: Data_cleaner,
    model: CVAE_TCN_Vamp | VAE_TCN_Vamp,
    path: str,
    k: int = 1,
    landings: bool = False,
) -> None:
    pseudo_in_rec = model.get_pseudo_inputs_recons()[::k, ...]
    traf = data_clean.output_converter(pseudo_in_rec, landing=landings)
    plot_traffic(traf, plot_path=path)


def plot_list_traff(
    l_traf: list[Traffic], plot_path: str, background: bool = False
) -> None:
    color_list = ["#f58518", "#4c78a8", "#54a24b"]  # orange, blue , green
    if background:
        # background elements
        paris_area = france.data.query("ID_1 == 1000")
        seine_river = Nominatim.search(
            "Seine river, France"
        ).shape.intersection(paris_area.union_all().buffer(0.1))

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
        i = 0
        for traffic in l_traf:
            traffic.plot(ax, alpha=0.3, color=color_list[i % len(color_list)])
            i += 1
        plt.savefig(plot_path)


def plot_vamp_generated(
    data_clean: Data_cleaner,
    model: CVAE_TCN_Vamp | VAE_TCN_Vamp,
    path: str,
    index: int,
    num_traj: int,
    landings: bool = False,
) -> None:
    used_model = model
    pseudo_in_rec = used_model.get_pseudo_inputs_recons()
    chosen_pseudo = pseudo_in_rec[index]

    gen_traj = used_model.generate_from_specific_vamp_prior(index, num_traj)
    pseudo_traff = data_clean.output_converter(
        chosen_pseudo.unsqueeze(0), landing=landings
    )
    gen_taff = data_clean.output_converter(gen_traj, landing=landings)

    plot_list_traff([pseudo_traff, gen_taff], path)


def display_traffic_per_typecode(
    data_clean: Data_cleaner, plt_path: str, background: bool = False
) -> None:
    all_traffic = data_clean.traffic_per_label()
    labels = data_clean.get_typecodes_labels()
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

        for ax, traff, label in tqdm(
            zip(axes, all_traffic, labels), desc="Plotting"
        ):
            traff.plot(ax, alpha=0.5, color="orange")
            ax.set_title(f"Typecode: {label},\n N = {len(traff)}")

        plt.tight_layout()
        plt.savefig(plt_path)
        plt.show()


def plot_traff_labels(
    plt_path: str, traffic_list: list[Traffic], labels: list[str]
) -> None:
    all_traffic = traffic_list
    labels = labels
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

        for ax, traff, label in tqdm(
            zip(axes, all_traffic, labels), desc="Plotting"
        ):
            traff.plot(ax, alpha=0.5, color="orange")
            ax.set_title(f"Typecode: {label},\n N = {len(traff)}")

        plt.tight_layout()
        plt.savefig(plt_path)
        plt.show()


def plot_compare_traffic_hue(
    traffic: Traffic,
    generated_traffic: Traffic,
    labels_hue: np.ndarray,
    n_trajectories: int = None,
    background: bool = True,
    plot_path: str = "data_orly/plot.png",
) -> int:
    if not n_trajectories:
        n_trajectories = len(generated_traffic)

    if background:
        # background elements
        paris_area = france.data.query("ID_1 == 1000")
        seine_river = Nominatim.search(
            "Seine river, France"
        ).shape.intersection(paris_area.union_all().buffer(0.1))
    labels_hue = labels_hue.flatten()
    print("flatten ", labels_hue)
    unique_labels = np.unique(labels_hue)
    print("unique labels ", unique_labels)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(unique_labels))]
    label_to_color = {
        lab: colors[i % len(unique_labels)]
        for i, lab in enumerate(unique_labels)
    }
    print(label_to_color)
    seq_len = len(traffic[0])
    flights_og_traffic = [
        Flight(traffic.data.iloc[i * seq_len : (i + 1) * seq_len])
        for i in range(min(len(traffic), n_trajectories))
    ]
    flights_gen_traffic = [
        Flight(generated_traffic.data.iloc[i * seq_len : (i + 1) * seq_len])
        for i in range(min(len(generated_traffic), n_trajectories))
    ]

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=Lambert93()))

        for i, flight in enumerate(flights_og_traffic):
            lab = labels_hue[
                i
            ]  # assuming the labels_hue order corresponds to traffic order
            color = label_to_color[lab]
            flight.plot(ax[0], color=color, alpha=0.7)

        for i, flight in enumerate(flights_gen_traffic):
            lab = labels_hue[i]
            color = label_to_color[lab]
            flight.plot(ax[1], color=color, alpha=0.7)

        import matplotlib.lines as mlines

        handles = [
            mlines.Line2D(
                [],
                [],
                color=label_to_color[lab],
                marker="o",
                linestyle="",
                label=str(lab),
            )
            for lab in unique_labels
        ]
        ax[0].legend(handles=handles, title="Labels")
        ax[1].legend(handles=handles, title="Labels")

        plt.savefig(plot_path)
    return 1


def vertical_rate_profile(
    traffic: Traffic, path: str, distance: bool = True
) -> None:
    """
    Plot the landing/vertical profile for every plane in a traffic.
    """

    if "timedelta" not in traffic.data.columns:
        raise ValueError("traffic must have set time deltas.")

    traffic = traffic.resample("2s").eval(desc="resmpling", max_workers=4)

    min_t, max_t = (
        traffic.data["timedelta"].max(),
        traffic.data["timedelta"].min(),
    )

    max_south = traffic.data["latitude"].min()
    bound = (max_south - 0.1, max_south + 0.1)

    def bound_f(f: Flight):
        if bound[0] < f.data["latitude"].iloc[-1] < bound[1]:
            return f
        return None

    traff_new = traffic.pipe(bound_f).eval(desc="l")

    plt.figure(figsize=(10, 6))
    import matplotlib.cm as cm

    colors = cm.viridis(np.linspace(0, 1, len(traffic)))
    max_len = max([len(f) for f in traff_new])

    mean_altitude_array = np.zeros(max_len, float)

    for i, f in enumerate(traff_new):
        plt.plot(
            f.data["timedelta"].to_list()
            if not distance
            else f.data["distance"].to_list(),
            f.data["altitude"].to_list(),
            color=colors[i],
            alpha=0.3,
        )
        # altitude_array = np.zeros(max_len, float)
        # altitude_temp_array = f.data["altitude"].to_numpy()
        # # altitude_array[0 : len(altitude_temp_array)] = altitude_temp_array

        # mean_altitude_array = mean_altitude_array + altitude_array

    # mean_altitude_array = mean_altitude_array / len(traff_new)
    # timestamp = [2 * i for i in range(max_len)]

    # plt.plot(timestamp, mean_altitude_array.tolist(), color="red", alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (ft)")
    plt.title("Altitude profile")
    plt.savefig(path, dpi=600)
    plt.tight_layout()
    plt.show()
    plot_traffic(traffic, path.split(".")[0] + "_traff.png", background=False)
    plot_traffic(
        traff_new, path.split(".")[0] + "_traff_limited.png", background=False
    )

    return


def vertical_rate_profile_2(
    traffic: Traffic,
    path: str,
    distance: bool = True,
    y_col: str = "altitude",
    x_col: str = "timedelta",
) -> None:
    """
    Improved version using altair
    """
    
    traffic = Traffic(traffic.data[[x_col if x_col!="CAS" else 'groundspeed',y_col,"callsign","icao24",'timestamp']])
    # traffic = traffic.resample("2s").eval(desc="")
    alts = []

    if x_col == "CAS":
        cas = aero.tas2cas(traffic.data["groundspeed"], h= traffic.data["altitude"])
        traffic = traffic.assign(CAS = cas)


    for flight in traffic:
        alts.append(flight.data[y_col].to_list())


    mean_flight = pd.DataFrame()
    
    max_len = max(len(arr) for arr in alts)
    mean_flight[y_col] = np.nanmean(
        np.array(
            [
                np.pad(
                    np.array(arr),
                    (0, max_len - len(arr)),
                    constant_values=np.nan,
                )
                for arr in alts
            ]
        ),
        axis=0,
    )
    mean_flight["timedelta"] = [2*i for i in range(max_len)]
    mean_flight = Flight(mean_flight)

    
    print(traffic.data.isna().sum().sum())
    print(' ')
    flights = [
        f.chart()
        .encode(
            x=alt.X(
                x_col,
                title="Time elapsed (s)"
                if x_col == 'timedelta'   
                else str(x_col),
                scale=alt.Scale(domain=(100, 600), clamp=True) if x_col == 'CAS' else None
            ),
            y=alt.Y(
                y_col,
                title=None,
                scale=alt.Scale(domain=(0, 20000), clamp=True),
            ),
            color=alt.value("#4c78a8"),
            opacity=alt.value(0.2),
        )
        .transform_filter("datum.altitude < 20000")
        for f in traffic
    ]

    if x_col == "timedelta":
        flights.append(
            mean_flight.chart()
            .encode(
                x=alt.X(
                    x_col,
                    title="Time elapsed (s)"
                    if x_col == 'timedelta'   
                    else str(x_col),
                    
                ),
                y=alt.Y(
                    y_col,
                    title=None,
                    scale=alt.Scale(domain=(0, 20000), clamp=True),
                ),
                color=alt.value("red"),
                opacity=alt.value(1),
            )
            .transform_filter("datum.altitude < 20000")
        )

    chart = alt.layer(*flights).properties(
        title="Departure altitude (in ft)", width=500, height=170
    )
    chart.save(path)
