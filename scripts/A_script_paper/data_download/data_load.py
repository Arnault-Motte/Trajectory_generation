import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm 

from traffic.core import Traffic

from traffic.data import opensky
from traffic.data import eurofirs
from traffic.core.structure import Airport

import datetime
from dateutil.relativedelta import relativedelta
from collections import Counter

import os

from pitot.geodesy import destination
from math import sqrt



from traffic.data import airports


def shift_NM(lat:float,lon:float,nm:float,alpha:float):
    """
    Shifts the latitude and logitude by nm nautical miles in the direction of alpha
    """

    meter = nm *1852
    lat2,lon2,_ = destination(lat,lon,alpha,meter)
    return lat2,lon2

def add_NM_to_lat_long(nm: float, lon: float, lat: float, sub: bool = False):
    """
    Add nm nautical miles to the  latitude and longitude
    """

    new_lat = lat + nm / 60 if not sub else lat - nm / 60
    new_lon = (
        lon + nm / (60 * np.cos(np.deg2rad(lat)))
        if not sub
        else lon - nm / (60 * np.cos(np.deg2rad(lat)))
    )
    return float(new_lon), new_lat

def add_to_bounds_0(bounds, nm: float):
    return tuple(list(add_NM_to_lat_long(nm, bounds[0], bounds[1],sub =True)) + list(
        add_NM_to_lat_long(nm, bounds[2], bounds[3]))
    )

def add_to_bounds(bounds, nm: float):
    """
    Stretch the bounds by nm nautical miles
    """
    west,south = shift_NM(bounds[1],bounds[0],nm,225)
    east,north = shift_NM(bounds[3],bounds[2],nm,45)
    return  south,west,north,east 

def months_between(d1, d2):
    # Ensure d1 <= d2
    if d1 > d2:
        d1, d2 = d2, d1
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

from multiprocessing import Pool, cpu_count
from tqdm import tqdm




# Save all to LFPO -----------------------------------------------------------------------------------------------

airport_name = "LFPO"
# start = datetime.datetime(2021, 1, 1)
start = datetime.datetime(2022, 5, 1)
end = datetime.datetime(2024, 12, 31)
dist = 50
true_dist = sqrt(dist**2 + dist**2) #can contain a circle of 50nm
airport = airports[airport_name]
bounds = airport.bounds
og_bounds = bounds
print("Og bounds", og_bounds)
bounds = add_to_bounds(bounds, true_dist)
bounds_0 = add_to_bounds_0(og_bounds, dist)
print(bounds)
print(bounds_0)
months = months_between(start, end)
weeks = (end - start).days // 7
print(months,"  ", weeks)
# total_traffs = None
#for i in range(months+1):
for i in range(weeks):

    
    new_start = start + relativedelta(days=7*i)
    new_end = start + relativedelta(days=7*i+7)
    # new_start = start + relativedelta(months=i)
    # new_end = start + relativedelta(months=i + 1)
    
    save_path_name = f'results/t_TO_{airport_name}_{new_start.strftime("%Y%m%d")}_{new_end.strftime("%Y%m%d")}.pkl'
    print(save_path_name)
    if os.path.exists(save_path_name):
        print("File exists.")
        continue

    print(f"|-- Start = {new_start}, End = {new_end} --|")

   
    landings_orly = opensky.history(
            start=new_start,
            stop=new_end,
            departure_airport=airport_name,
            bounds=bounds,
    )
    print("Data downloaded successfully")
    # f_ids = landings_orly.assign_id().eval(max_workers=8).flight_ids
    # counter = Counter(f_ids).most_common(10)

    # print(len(landings_orly.data))
    # print(len(landings_orly.drop_duplicates(subset=["timestamp", "icao24", "callsign"], keep='first').data))
    # landings_orly = landings_orly.drop_duplicates(subset=["timestamp", "icao24", "callsign"], keep='last') #clean the duplicated data otherwise resample fails
    # landings_orly = landings_orly.resample(250).eval(desc="Resampling flights",max_workers=8)

    landings_orly.to_pickle(save_path_name)

    print("Data loaded successfully")
        
        # selected = filter_flights_multiprocess(landings_orly,max_workers = 8)
        # print(f"Selected {len(selected)} flights")
        # landings_orly = sum(selected)
        # if i == 0:
        #     total_traffs = landings_orly
        # else:
        #     total_traffs += landings_orly
    # except Exception as e:
    #     print(f"Error: {e}")
    #     input("Press to continue")



# Save all to LFPO -----------------------------------------------------------------------------------------------

airport_name = "LFPG"
start = datetime.datetime(2021, 1, 1)
end = datetime.datetime(2024, 12, 31)

dist = 50
true_dist = sqrt(dist**2 + dist**2)
airport = airports[airport_name]
bounds = airport.bounds
og_bounds = bounds
print("Og bounds", og_bounds)
bounds = add_to_bounds(bounds, true_dist)
bounds_0 = add_to_bounds_0(og_bounds, dist)
print(bounds)
print(bounds_0)
months = months_between(start, end)
weeks = (end - start).days // 7
print(months,"  ", weeks)

# total_traffs = None

#for i in range(months+1):
for i in range(weeks):

    new_start = start + relativedelta(days=7*i)
    new_end = start + relativedelta(days=7*i+7)
    # new_start = start + relativedelta(months=i)
    # new_end = start + relativedelta(months=i + 1)
    
    save_path_name = f'results/TO_{airport_name}_{new_start.strftime("%Y%m%d")}_{new_end.strftime("%Y%m%d")}.pkl'
    if os.path.exists(save_path_name):
        print("File exists.")
        continue

    print(f"|-- Start = {new_start}, End = {new_end} --|")

    try:
        landings_orly = opensky.history(
            start=new_start,
            stop=new_end,
            departure_airport=airport_name,
            bounds=bounds,
        )
        print("Data downloaded successfully")
        landings_orly = landings_orly.drop_duplicates(subset=["timestamp", "icao24", "callsign"], keep='first')
        landings_orly = landings_orly.resample(250).eval(desc="Resampling flights",max_workers=8)
        landings_orly.to_pickle(save_path_name)

        print("Data loaded successfully")
        
        # selected = filter_flights_multiprocess(landings_orly,max_workers = 8)
        # print(f"Selected {len(selected)} flights")
        # landings_orly = sum(selected)
        # if i == 0:
        #     total_traffs = landings_orly
        # else:
        #     total_traffs += landings_orly
    except Exception as e:
        print(f"Error: {e}")
