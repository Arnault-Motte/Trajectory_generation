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

###TEST for seeing the reconstruction of the autoencoder
def plot_traffic(
        traffic: Traffic,
        plot_path: str,
        background: bool = True,
    ) -> None:
        if background:
            # background elements
            paris_area = france.data.query("ID_1 == 1000")
            seine_river = Nominatim.search(
                "Seine river, France"
            ).shape.intersection(paris_area.union_all().buffer(0.1))

        with plt.style.context("traffic"):
            fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
            traffic.plot(ax, alpha=0.2, color="#4c78a8")
            plt.savefig(plot_path)
            plt.show()

def plot_DTW_SSPD(distances : list[dict],labels:list[str],path:str)->None:
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (10,10))
    #DTW
 
    for label,dist in zip(labels,distances):
        sorted_d = np.sort([d["dtw"] for d in dist.values()])
        cdf = np.arange(1,len(sorted_d)+1) / len(sorted_d)
        axes[0].plot(sorted_d,cdf,label = label)

    axes[0].set_title("DTW")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Cumulative probability")
    axes[0].legend(title = "Generation method")
    axes[0].grid(True)
    axes[0].set_xlim(0,1200000)
    

    for label,dist in zip(labels,distances):
        sorted_d = np.sort([d["sspd"] for d in dist.values()])
        cdf = np.arange(1,len(sorted_d)+1) / len(sorted_d)
        axes[1].plot(sorted_d,cdf,label = label)

    axes[1].set_title("SSPD")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].legend(title = "Generation method")
    axes[1].grid(True)
    axes[1].set_xlim(0,16000)

    plt.tight_layout()
    plt.savefig(path)
    plt.show()



class Displayer:
    def __init__(self, data_clean: Data_cleaner) -> None:
        self.data_clean = data_clean
        self.type_code_vrate = {}
        self.type_code_vrate_gen = {}

    def return_label_shanon(self) -> dict:
        dic_res = {}
        for label, _ in self.type_code_vrate.items():
            dic_res[label] = jason_shanon(
                self.type_code_vrate[label], self.type_code_vrate_gen[label]
            )
        return dic_res

    def plot_traffic(
        self,
        traffic: Traffic,
        plot_path: str,
        background: bool = True,
    ) -> None:
        if background:
            # background elements
            paris_area = france.data.query("ID_1 == 1000")
            seine_river = Nominatim.search(
                "Seine river, France"
            ).shape.intersection(paris_area.union_all().buffer(0.1))

        with plt.style.context("traffic"):
            fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
            traffic.plot(ax, alpha=0.2, color="#4c78a8")
            plt.savefig(plot_path)
            plt.show()

    def plot_compare_traffic(
        self,
        traffic: Traffic,
        generated_traffic: Traffic,
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

        with plt.style.context("traffic"):
            fig, ax = plt.subplots(
                1, 2, subplot_kw=dict(projection=Lambert93())
            )
            traffic[: min(n_trajectories, len(traffic))].plot(ax[0], alpha=0.7)
            generated_traffic[
                : min(n_trajectories, len(generated_traffic))
            ].plot(ax[1], alpha=0.7)
            plt.savefig(plot_path)
        return 1

    def plot_latent_space(
        self, num_point: int, model: VAE_TCN_Vamp, plt_path: str
    ) -> int:
        data = self.data_clean.clean_data()
        train, _ = get_data_loader(data, 500, 0.8)
        list_tensor = [batch[0] for batch in train]
        all_tensor = torch.concat(list_tensor).to(
            next(model.parameters()).device
        )

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
        self, num_point: int, model: VAE_TCN_Vamp, plt_path: str
    ) -> int:
        data = self.data_clean.clean_data()
        train, _ = get_data_loader(data, 500, 0.8)
        list_tensor = [batch[0] for batch in train]
        all_tensor = torch.concat(list_tensor).to(
            next(model.parameters()).device
        )

        all_tensor = all_tensor.permute(0, 2, 1)
        pseudo_latent,_ = model.pseudo_inputs_latent()
        pseudo_latent  = pseudo_latent.cpu().detach().numpy()

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
        self, label: str, num_point: int, model: CVAE_TCN_Vamp, plt_path: str
    ) -> int:
        labels_array = np.full(num_point, label).reshape(-1, 1)
        transformed_label = self.data_clean.one_hot.transform(labels_array)

        traff_data = self.data_clean.basic_traffic_data.data
        traffic_curated = Traffic(traff_data[traff_data["typecode"] == label])

        data = self.data_clean.clean_data_specific(traffic_curated)[
            : min(num_point, len(traffic_curated))
        ]

        labels_true = np.array(
            [
                label
                for _ in traffic_curated[: min(num_point, len(traffic_curated))]
            ]
        ).reshape(-1, 1)
        transformed_true_labels = self.data_clean.one_hot.transform(labels_true)

        train, _ = get_data_loader_labels(
            data, transformed_true_labels, 500, 0.8
        )

        list_tensor = [batch[0] for batch in train]
        all_tensor = torch.concat(list_tensor).to(
            next(model.parameters()).device
        )

        list_labels = [labels for _, labels in train]
        all_labels = torch.concat(list_labels).to(
            next(model.parameters()).device
        )
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
        self, num_point: int, model: CVAE_TCN_Vamp, plt_path: str
    ) -> int:
        top10 = self.data_clean.get_top_10_planes()
        for label, _ in top10:
            path = plt_path.split(".")[0] + "_" + label + ".png"
            self.plot_latent_space_for_label(label, num_point, model, path)
        return 0

    def set_v_rates_per_typecode(self, chosen_labels: list[str]) -> None:
        data = self.data_clean.basic_traffic_data
        map_typecode_vrate = {
            code: [
                flight.vertical_rate_mean
                for flight in data.query(f"typecode in ['{code}']")
            ]
            for code in chosen_labels
        }
        self.type_code_vrate = map_typecode_vrate

    def plot_distribution_typecode(
        self, path1: str, path2: str, hist: bool = False
    ) -> None:
        data = self.data_clean.basic_traffic_data
        if hist:
            path1 = path1.split(".")[0] + "_hist.png"
            path2 = path2.split(".")[0] + "_hist.png"

        labels_list = self.data_clean.get_typecodes_labels()
        self.set_v_rates_per_typecode(labels_list)
        map_typecode_vrate = self.type_code_vrate

        for typecode, vertical_rates in map_typecode_vrate.items():
            if not hist:
                sns.kdeplot(vertical_rates, label=typecode, linewidth=2)
                plt.ylabel("Density")
            else:
                sns.histplot(vertical_rates, label=typecode, bins=30, kde=True)
                plt.ylabel("Count")

        plt.xlabel("Vertical Rate")
        plt.title("Vertical Rate Distribution by Aircraft Typecode")
        plt.legend(title="Typecode")
        plt.savefig(path1)
        plt.show

        # Define number of rows and columns
        rows, cols = 2, 5
        fig, axes = plt.subplots(
            rows, cols, figsize=(15, 8)
        )  # Adjust figure size

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Iterate over each typecode and corresponding subplot
        for ax, (typecode, vertical_rates) in zip(
            axes, map_typecode_vrate.items()
        ):
            if not hist:
                sns.kdeplot(vertical_rates, ax=ax, linewidth=2)
                ax.set_ylabel("Density")
            else:
                sns.histplot(
                    vertical_rates, ax=ax, label=typecode, bins=30, kde=True
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

    def generated_data_v_rates_to_be_displayed(
        self,
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
                transformed_label = self.data_clean.one_hot.transform(
                    labels_array
                )
                pbar.update(1)

                # print("|--Sampling--|")
                labels_final = torch.Tensor(transformed_label).to(
                    next(model.parameters()).device
                )
                sampled = model.sample(n_points, batch_size, labels_final)
                pbar.update(1)

                # print("|--Converting--|")
                traf = self.data_clean.output_converter(sampled)
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
        self.type_code_vrate_gen = map_typecode_vrate

    def plot_distribution_typecode_label_generation(
        self,
        path1: str,
        path2: str,
        model: CVAE_TCN_Vamp,
        number_of_point: int = 2000,
        hist: bool = False,
        bounds: tuple[float, float] = (-2000, 2000),
        batch_size: int = 500,
    ) -> None:
        plt.clf()
        if hist:
            path1 = path1.split(".")[0] + "_hist.png"
            path2 = path2.split(".")[0] + "_hist.png"

        labels_list = self.data_clean.get_typecodes_labels()
        self.generated_data_v_rates_to_be_displayed(
            model, number_of_point, bounds, labels_list, batch_size
        )
        map_typecode_vrate = self.type_code_vrate_gen
        print("|--Data Generated--|")
        fig, ax = plt.subplots() 
        for typecode, vertical_rates in map_typecode_vrate.items():
            if not hist:
                sns.kdeplot(vertical_rates, label=typecode, ax=ax, linewidth=2)
                plt.ylabel("Density")
            else:
                sns.histplot(vertical_rates, label=typecode,ax=ax, bins=30, kde=True)
                plt.ylabel("Count")

        # plt.xlim(, 0)
        plt.xlabel("Vertical Rate")
        plt.title("Vertical Rate Distribution by Aircraft Typecode")
        plt.legend(title="Typecode")
        plt.savefig(path1)
        plt.show()

        # Define number of rows and columns
        rows, cols = 2, 5
        fig, axes = plt.subplots(
            rows, cols, figsize=(15, 8)
        )  # Adjust figure size

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Iterate over each typecode and corresponding subplot
        for ax, (typecode, vertical_rates) in zip(
            axes, map_typecode_vrate.items()
        ):
            if not hist:
                sns.kdeplot(vertical_rates, ax=ax, linewidth=2)
                ax.set_ylabel("Density")
            else:
                sns.histplot(
                    vertical_rates, ax=ax, label=typecode, bins=30, kde=True
                )
                ax.set_ylabel("Count")
            ax.set_title(f"Typecode: {typecode}")
            ax.set_xlabel("Vertical Rate")

        # Remove any empty subplots (if less than 10 typecodes)
        for i in range(len(map_typecode_vrate), rows * cols):
            fig.delaxes(axes[i])

        # plt.xlim(-2000, 0)
        plt.tight_layout()
        plt.savefig(path2)
        plt.show()

    def display_pseudo_inputs(
        self,
        model: CVAE_TCN_Vamp | VAE_TCN_Vamp,
        path: str,
        k: int = 1,
        landings: bool = False,
    ) -> None:
        pseudo_in_rec = model.get_pseudo_inputs_recons()[::k, ...]
        traf = self.data_clean.output_converter(pseudo_in_rec, landing=landings)
        self.plot_traffic(traf, plot_path=path)

    def plot_list_traff(
        self, l_traf: list[Traffic], plot_path: str, background: bool = False
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
                traffic.plot(
                    ax, alpha=0.3, color=color_list[i % len(color_list)]
                )
                i += 1
            plt.savefig(plot_path)

    def plot_vamp_generated(
        self,
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
        pseudo_traff = self.data_clean.output_converter(
            chosen_pseudo.unsqueeze(0), landing=landings
        )
        gen_taff = self.data_clean.output_converter(gen_traj, landing=landings)

        self.plot_list_traff([pseudo_traff, gen_taff], path)

    def plot_latent_spaces_labels(
        self, model: CVAE_TCN_Vamp, plt_path: str
    ) -> int:
        labels_list = self.data_clean.get_typecodes_labels()
        list_all_points = []
        for label, _ in labels_list:
            traff_data = self.data_clean.basic_traffic_data.data
            traffic_curated = Traffic(
                traff_data[traff_data["typecode"] == label]
            )
            data = self.data_clean.clean_data_specific(traffic_curated)
            labels_true = np.array([label for _ in traffic_curated]).reshape(
                -1, 1
            )
            transformed_true_labels = self.data_clean.one_hot.transform(
                labels_true
            )
            train, _ = get_data_loader_labels(
                data, transformed_true_labels, 500, 0.99
            )

            list_tensor = [batch[0] for batch in train]
            all_tensor = torch.concat(list_tensor).to(
                next(model.parameters()).device
            )

            list_labels = [labels for _, labels in train]
            all_labels = torch.concat(list_labels).to(
                next(model.parameters()).device
            )
            all_tensor = all_tensor.permute(0, 2, 1)

            ## getting mu and sigma
            mu, log_var = model.encode(all_tensor, all_labels)
            scale = (log_var / 2).exp()
            # get the actual distribution
            distribution = Independent(Normal(mu, scale), 1)
            posterior_samples = distribution.rsample()

            num_samples = all_labels.shape[0]
            print(all_labels.shape)
            print(num_samples)
            prior_sample = (
                model.sample_from_prior(num_samples).squeeze(1)
                if not model.prior_weights_layers
                else model.sample_from_conditioned_prior(
                    num_samples, all_labels[0]
                )
            )
            # print(f"posterior shape {posterior_samples.shape}")
            # print(f"prior shape : {prior_sample.shape}")

            total = np.concat(
                (
                    posterior_samples.cpu().detach().numpy(),
                    prior_sample.cpu().detach().numpy(),
                ),
                axis=0,
            )
            list_all_points.append(total)
        print(list_all_points[0].shape)
        all_size = [elem.shape[0] for elem in list_all_points]
        total_fit = np.concat(
            [tot[: tot.shape[0] // 2] for tot in list_all_points], axis=0
        )
        all_total = np.concat(list_all_points, axis=0)
        print(all_total.shape)

        pca = PCA(n_components=2).fit(total_fit)
        embeded = pca.transform(all_total)

        with plt.style.context("traffic"):
            fig, ax = plt.subplots(1, figsize=(15, 10))
            total_size = 0
            cmap = plt.get_cmap("tab10")
            for i, current_size in enumerate(all_size):
                # some  specifc shape
                color = cmap(i % 10)
                ax.scatter(
                    embeded[total_size : total_size + current_size // 2, 0],
                    embeded[total_size : total_size + current_size // 2, 1],
                    s=4,
                    c=[color],
                    marker="o",
                    label="True" if i == 0 else None,
                    alpha=0.7,
                )
                # another shape
                ax.scatter(
                    embeded[
                        total_size + current_size // 2 : total_size
                        + current_size,
                        0,
                    ],
                    embeded[
                        total_size + current_size // 2 : total_size
                        + current_size,
                        1,
                    ],
                    s=4,
                    c=[color],
                    marker="s",
                    label="Generated" if i == 0 else None,
                    alpha=0.4,
                )
                total_size += current_size
            ax.title.set_text("Latent Space")
            ax.legend()

            plt.savefig(plt_path)

        return 0

    def plot_compare_traffic_hue(
        self,
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
        seq_len = self.data_clean.seq_len
        flights_og_traffic = [
            Flight(traffic.data.iloc[i * seq_len : (i + 1) * seq_len])
            for i in range(min(len(traffic), n_trajectories))
        ]
        flights_gen_traffic = [
            Flight(generated_traffic.data.iloc[i * seq_len : (i + 1) * seq_len])
            for i in range(min(len(generated_traffic), n_trajectories))
        ]

        with plt.style.context("traffic"):
            fig, ax = plt.subplots(
                1, 2, subplot_kw=dict(projection=Lambert93())
            )

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

    def plot_generated_traff_hue_datasets(
        self,
        labels: list[str],
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

        unique_labels = np.unique(labels)
        cmap = plt.get_cmap(f"tab{len(unique_labels)}")
        label_to_color = {
            lab: cmap(i % 10) for i, lab in enumerate(unique_labels)
        }

        with plt.style.context("traffic"):
            fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))

            for i, flight in enumerate(
                generated_traffic[: min(n_trajectories, len(generated_traffic))]
            ):
                lab = labels_hue[i]
                color = label_to_color[lab]
                flight.plot(ax, color=color, alpha=0.7)

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
            ax.legend(handles=handles, title="Labels")
            plt.savefig(plot_path)
        return 1

    def plot_generated_label(
        self,
        model: CVAE_TCN_Vamp,
        label: str,
        plot_path: str,
        num_point: int = 2000,
        batch_size: int = 500,
        take_off: bool = False,
    ) -> None:
        labels_array = np.full(batch_size, label).reshape(-1, 1)
        labels_transf = self.data_clean.one_hot.transform(labels_array)
        labels_tensor = torch.Tensor(labels_transf).to(
            next(model.parameters()).device
        )

        generated_point = model.sample(num_point, batch_size, labels_tensor)

        # print("|--Converting--|")
        traf = self.data_clean.output_converter(
            generated_point, landing=take_off
        )

        self.plot_traffic(traf, plot_path=plot_path)

    def plot_generated_VAE(
        self,
        model: VAE_TCN_Vamp,
        plot_path: str,
        num_point: int = 2000,
        batch_size: int = 500,
        take_off: bool = False,
    ) -> None:

        generated_point = model.sample(num_point, batch_size)

        # print("|--Converting--|")
        traf = self.data_clean.output_converter(
            generated_point, landing=take_off
        )

        self.plot_traffic(traf, plot_path=plot_path)

    def plot_generated_label_dataset(
        self,
        model: CVAE_TCN_Vamp,
        label: str,
        plot_path: str,
        num_point: int = 2000,
        batch_size: int = 500,
    ) -> None:
        labels_array = np.full(batch_size, label).reshape(-1, 1)
        labels_transf = self.data_clean.one_hot.transform(labels_array)
        labels_tensor = torch.Tensor(labels_transf).to(
            next(model.parameters()).device
        )

        generated_point = model.sample(num_point, batch_size, labels_tensor)

        # print("|--Converting--|")
        labels_list = np.full(num_point, label).tolist()
        traf = self.data_clean.output_converter_several_datasets(
            generated_point, labels_list
        )

        self.plot_traffic(traf, plot_path=plot_path)

    def give_bounds_vr_rate(self) -> tuple[float, float]:
        """ "
        Returns the min and max values found in the vertical rates means in the data.
        Be careful to compute the vertical rates befor using the function
        """
        if not self.type_code_vrate:
            raise ValueError("Please compute the vertical rates beforehand")

        vertical_rates = [
            rate for rates in self.type_code_vrate.values() for rate in rates
        ]
        return min(vertical_rates), max(vertical_rates)

    def plot_v_rate_true_top_generated(
        self,
        plt_path: str,
        n_point: int,
        model: CVAE_TCN_Vamp,
        regenerate: bool = False,
        batch_size: int = 500,
        hist: bool = False,
    ) -> None:
        """
        Plots the distribution of the vertical rates for each of the typecodes.
        Compares the generated vertical rates with the true ones.
        """

        labels_list = self.data_clean.get_typecodes_labels()

        if not self.type_code_vrate or regenerate:
            self.set_v_rates_per_typecode(labels_list)
        if not self.type_code_vrate_gen or regenerate:
            bounds = self.give_bounds_vr_rate()
            self.generated_data_v_rates_to_be_displayed(
                model,
                n_point,
                bounds,
                labels_to_gen=labels_list,
                batch_size=batch_size,
            )

        js_shan = self.return_label_shanon()
        map_typecode_vrate = self.type_code_vrate
        map_typecode_vrate_gen = self.type_code_vrate_gen

        # Define number of rows and columns
        rows, cols = 2, 5
        fig, axes = plt.subplots(
            rows, cols, figsize=(15, 8)
        )  # Adjust figure size

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Iterate over each typecode and corresponding subplot
        for ax, (typecode, vertical_rates) in zip(
            axes, map_typecode_vrate.items()
        ):
            if not hist:
                sns.kdeplot(
                    vertical_rates,
                    ax=ax,
                    linewidth=2,
                    color="blue",
                    alpha=0.5,
                    label="True",
                )
                sns.kdeplot(
                    map_typecode_vrate_gen[typecode],
                    ax=ax,
                    linewidth=2,
                    color="orange",
                    alpha=0.5,
                    label="Generated",
                )
                ax.set_ylabel("Density")
            else:
                sns.histplot(
                    vertical_rates,
                    ax=ax,
                    bins=30,
                    kde=True,
                    color="blue",
                    alpha=0.5,
                    label="True",
                )
                sns.histplot(
                    map_typecode_vrate_gen[typecode],
                    ax=ax,
                    bins=30,
                    kde=True,
                    color="orange",
                    alpha=0.5,
                    label="generated",
                )
                ax.set_ylabel("Count")
            ax.set_title(
                f"Typecode: {typecode}, Shanon : {js_shan[typecode]:.2f}"
            )
            ax.set_xlabel("Vertical Rate")
            ax.legend()

        # Remove any empty subplots (if less than 10 typecodes)
        for i in range(len(map_typecode_vrate), rows * cols):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(plt_path)
        plt.show()

    def display_traffic_per_typecode(
        self, plt_path: str, background: bool = False
    ) -> None:
        all_traffic = self.data_clean.traffic_per_label()
        labels = self.data_clean.get_typecodes_labels()
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
        self, plt_path: str, traffic_list: list[Traffic], labels: list[str]
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
