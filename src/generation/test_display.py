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
)
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_TCN_VampPrior import (
    get_data_loader as get_data_loader_labels,
)
from data_orly.src.generation.models.VAE_TCN_VampPrior import (
    VAE_TCN_Vamp,
    get_data_loader,
)
from traffic.core import Traffic

###TEST for seeing the reconstruction of the autoencoder


class Displayer:
    def __init__(self, data_clean: Data_cleaner) -> None:
        self.data_clean = data_clean

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
            traffic.plot(ax, alpha=0.7, color="#4c78a8")
            plt.savefig(plot_path)

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

    def plot_distribution_typecode(
        self, path1: str, path2: str, hist: bool = False
    ) -> None:
        top_10 = self.data_clean.get_top_10_planes()
        data = self.data_clean.basic_traffic_data
        if hist:
            path1 = path1.split(".")[0] + "_hist.png"
            path2 = path2.split(".")[0] + "_hist.png"
        map_typecode_vrate = {
            code: [
                flight.vertical_rate_mean
                for flight in data.query(f"typecode in ['{code}']")
            ]
            for code, _ in top_10
        }
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

    def plot_distribution_typecode_label_generation(
        self,
        path1: str,
        path2: str,
        model: CVAE_TCN_Vamp,
        number_of_point: int = 2000,
        hist: bool = False,
        bounds: tuple[float, float] = (-2000, 2000),
    ) -> None:
        top_10 = self.data_clean.get_top_10_planes()
        # data = self.data_clean.basic_traffic_data
        # map_typecode_vrate = {
        #     code: [
        #         flight.vertical_rate_mean
        #         for flight in data.query(f"typecode in ['{code}']")
        #     ]
        #     for code, _ in top_10
        # }
        if hist:
            path1 = path1.split(".")[0] + "_hist.png"
            path2 = path2.split(".")[0] + "_hist.png"
        map_typecode_vrate = dict()
        print("|--Data Generation--|")

        for label, _ in tqdm(top_10, desc="Generating"):
            with tqdm(total=4, desc=f"Processing {label}") as pbar:
                # print("|--Encoding--|")
                labels_array = np.full(500, label).reshape(-1, 1)
                transformed_label = self.data_clean.one_hot.transform(
                    labels_array
                )
                pbar.update(1)

                # print("|--Sampling--|")
                labels_final = torch.Tensor(transformed_label).to(
                    next(model.parameters()).device
                )
                sampled = model.sample(number_of_point, 500, labels_final)
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

        print("|--Data Generated--|")
        for typecode, vertical_rates in map_typecode_vrate.items():
            if not hist:
                sns.kdeplot(vertical_rates, label=typecode, linewidth=2)
                plt.ylabel("Density")
            else:
                sns.histplot(vertical_rates, label=typecode, bins=30, kde=True)
                plt.ylabel("Count")

        # plt.xlim(, 0)
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

        # plt.xlim(-2000, 0)
        plt.tight_layout()
        plt.savefig(path2)
        plt.show()

    # def test_generated(self, model: CVAE_TCN_Vamp) -> None:
    #     labels_array = np.full(500, label).reshape(-1, 1)
    #     transformed_label = self.data_clean.one_hot.transform(labels_array)

    #     # print("|--Sampling--|")
    #     labels_final = torch.Tensor(transformed_label).to(
    #         next(model.parameters()).device
    #     )
    #     sampled = model.sample(number_of_point, 500, labels_final)

    #     # print("|--Converting--|")
    #     traf = self.data_clean.output_converter(sampled)

    def display_pseudo_inputs(self, model: CVAE_TCN_Vamp, path: str) -> None:
        pseudo_in_rec = model.get_pseudo_inputs_recons()
        traf = self.data_clean.output_converter(pseudo_in_rec)
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
                    ax, alpha=0.7, color=color_list[i % len(color_list)]
                )
                i += 1
            plt.savefig(plot_path)

    def plot_vamp_generated(
        self,
        model: CVAE_TCN_Vamp,
        path: str,
        index: int,
        num_traj: int,
    ) -> None:
        pseudo_in_rec = model.get_pseudo_inputs_recons()
        chosen_pseudo = pseudo_in_rec[index]

        gen_traj = model.generate_from_specific_vamp_prior(index, num_traj)
        pseudo_traff = self.data_clean.output_converter(
            chosen_pseudo.unsqueeze(0)
        )
        gen_taff = self.data_clean.output_converter(gen_traj)

        self.plot_list_traff([pseudo_traff, gen_taff], path)

    def plot_latent_spaces_labels(
        self, model: CVAE_TCN_Vamp, plt_path: str
    ) -> int:
        top10 = self.data_clean.get_top_10_planes()
        list_all_points = []
        for label, _ in top10:
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
        unique_labels = np.unique(labels_hue)
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(unique_labels))]
        label_to_color = {
            lab: colors[i % len(unique_labels)] for i, lab in enumerate(unique_labels)
        }

        with plt.style.context("traffic"):
            fig, ax = plt.subplots(
                1, 2, subplot_kw=dict(projection=Lambert93())
            )

            for i, flight in enumerate(traffic[: min(n_trajectories, len(traffic))]):
                lab = labels_hue[i]  # assuming the labels_hue order corresponds to traffic order
                color = label_to_color[lab]
                flight.plot(ax[0], color=color, alpha=0.7)

            for i, flight in enumerate(generated_traffic[: min(n_trajectories, len(generated_traffic))]):
                lab = labels_hue[i]
                color = label_to_color[lab]
                flight.plot(ax[1], color=color, alpha=0.7)

            import matplotlib.lines as mlines
            handles = [
                mlines.Line2D([], [], color=label_to_color[lab], marker="o", linestyle="", label=str(lab))
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
            fig, ax = plt.subplots(
                subplot_kw=dict(projection=Lambert93())
            )



            for i, flight in enumerate(generated_traffic[: min(n_trajectories, len(generated_traffic))]):
                lab = labels_hue[i]
                color = label_to_color[lab]
                flight.plot(ax, color=color, alpha=0.7)

            import matplotlib.lines as mlines
            handles = [
                mlines.Line2D([], [], color=label_to_color[lab], marker="o", linestyle="", label=str(lab))
                for lab in unique_labels
            ]
            ax.legend(handles=handles, title="Labels")
            plt.savefig(plot_path)
        return 1
    
    
