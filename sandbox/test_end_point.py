import sys  # noqa: I001
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))

from traffic.core import Traffic, Flight
from traffic.core.mixins import PointMixin
from cartes.crs import Lambert93, PlateCarree

t = Traffic.from_file(
    "data_orly/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl"
)

data = t.data

end_points = pd.DataFrame(t.data.iloc[::200])
end_points =  pd.DataFrame(end_points.iloc[:200])
print(len(end_points))


with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    i =0
    for i,row in tqdm(end_points.iterrows()):
        # print(row)
        point = PointMixin()
        point.latitude = row["latitude"]
        point.longitude = row["longitude"]
        point.name = ""
        point.plot(ax)
    t[:10].plot(ax,alpha=0.3)

    plt.savefig("end_points.png",dpi= 600)
    plt.show()
