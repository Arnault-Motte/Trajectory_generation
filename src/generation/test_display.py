import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from cartes.atlas import france
from cartes.crs import Lambert93, PlateCarree
from cartes.osm import Nominatim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.distributions import Independent, Normal

from data_orly.src.generation.data_process import Data_cleaner
from data_orly.src.generation.models.VAE_TCN_VampPrior import (
    VAE_TCN_Vamp,
    get_data_loader,
)
from traffic.core import Traffic
import numpy as np

###TEST for seeing the reconstruction of the autoencoder


class Displayer:
    def __init__(self, data_clean: Data_cleaner) -> None:
        self.data_clean = data_clean

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

    def plot_latent_space(self, num_point: int, model: VAE_TCN_Vamp,plt_path:str) -> int:
        data = self.data_clean.clean_data()
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
