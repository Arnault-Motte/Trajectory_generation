from cartes.atlas import france
from cartes.crs import Lambert93, PlateCarree
from cartes.osm import Nominatim
from matplotlib import pyplot as plt
from traffic.core import Traffic
import matplotlib as mpl
###TEST for seeing the reconstruction of the autoencoder


class Displayer():
    def __init__(self):
        pass

    def plot_compare_traffic(
        self,
        traffic: Traffic,
        generated_traffic: Traffic,
        n_trajectories=None,
        background=True,
        plot_path="data_orly/plot.png",
    )  :
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
            plt.savefig( plot_path)
            return fig
        
