# %%
from traffic.algorithms.generation import Generation
import matplotlib.pyplot as plt
from traffic.data.datasets import landing_zurich_2019
from cartes.crs import EuroPP

t = (
    landing_zurich_2019
    .query("runway == '14' and initial_flow == '162-216'")
    .assign_id()
    .unwrap()
    .resample(100)
    .eval(desc="")
)

# %%
with plt.style.context("traffic"):
    ax = plt.axes(projection=EuroPP())
    t.plot(ax, alpha=0.05)
    t.centroid(nb_samples=None, projection=EuroPP()).plot(
        ax, color="#f58518"
    )

# %%

t = t.compute_xy(projection=EuroPP())
from traffic.core import Traffic

def compute_timedelta(df: "pd.DataFrame"):
    return (df.timestamp - df.timestamp.min()).dt.total_seconds()

t = t.iterate_lazy().assign(timedelta=compute_timedelta).eval()

# %%
from traffic.core import Traffic

def compute_timedelta(df: "pd.DataFrame"):
    return (df.timestamp - df.timestamp.min()).dt.total_seconds()

t = t.iterate_lazy().assign(timedelta=compute_timedelta).eval()

# %%
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

g1 = Generation(
    generation=GaussianMixture(n_components=2),
    features=["x", "y", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1, 1))
).fit(t)
# %%
g2 = t.generation(
    generation=GaussianMixture(n_components=1),
    features=["x", "y", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1, 1))
)

# %%
t_gen1 = g1.sample(500, projection=EuroPP())
t_gen2 = g2.sample(500, projection=EuroPP())

with plt.style.context("traffic"):
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=EuroPP()))

    t_gen1.plot(ax[0], alpha=0.2)
    t_gen1.centroid(nb_samples=None, projection=EuroPP()).plot(
        ax[0], color="#f58518"
    )

    t_gen2.plot(ax[1], alpha=0.2)
    t_gen2.centroid(nb_samples=None, projection=EuroPP()).plot(
        ax[1], color="#f58518"
    )
# %%
plt.show()
# %%
