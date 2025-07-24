import os
import sys  # noqa: I001

from traffic.core import Traffic

current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path))

print("Current working directory:", current_path)
print(os.path.dirname(__file__))


from src.data_process import split_trajectories_label

traff = Traffic.from_file(
    "/home/arnault/traffic/data/sampled_data/combined_data/B738_A320_all2.pkl"
)

# traff = Traffic(traff.data[["track","altitude","groundspeed",'timedelta']])

traff1,traff2 = split_trajectories_label(traff,0.5)

print(traff1[0].data["track"])
print(traff1[0].data["track"].value_counts())
print('_____________________________')
print(traff2[0].data["track"])
print(traff2[0].data["track"].value_counts())


