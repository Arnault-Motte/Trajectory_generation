from traffic.core import Traffic,Flight
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from cartes.crs import Lambert93
from traffic.data import airports

from shapely.geometry import Point
from shapely.ops import transform
import pyproj
from functools import partial

from shapely.geometry import Polygon
from pitot.geodesy import destination, greatcircle



path_to_results = 'results'
save_path = 'results/combined_traff'
airport = 'LFPO'

def centered_around_airport(airport,rad: float):
    """
    returns the a circle shape around the airports center. With a radius of rad in NM.
    """
    point = airports[airport].centroid

    radius_m = rad * 1852  # converting to meters

    # Step 3: Project to an appropriate projection (Azimuthal equidistant centered on the point)
    proj_wgs84 = pyproj.CRS('EPSG:4326')
    proj_aeqd = pyproj.Proj(proj='aeqd', lat_0=point.x, lon_0=point.y)

    project = partial(pyproj.transform, pyproj.Proj(proj_wgs84), proj_aeqd)
    project_back = partial(pyproj.transform, proj_aeqd, pyproj.Proj(proj_wgs84))

    # Project point to planar coordinates, buffer it, and transform back
    point_proj = transform(project, point)
    buffer_proj = point_proj.buffer(radius_m)
    buffer_wgs84 = transform(project_back, buffer_proj)
    return buffer_wgs84

def centered_around_airport2(airport,rad: float):
    """
    returns the a circle shape around the airports center. With a radius of rad in NM.
    """
    from shapely.ops import transform
    from pyproj import Geod
    from traffic.core import Traffic
    point = airports[airport].centroid

    # Define the center of the circle (latitude, longitude) and radius in NM
    center_lat = point.x  # Latitude of Paris
    center_lon = point.y   # Longitude of Paris
        # Radius in nautical miles

    print(center_lat)
    print(center_lon)

    # Convert radius from NM to meters (1 NM = 1852 meters)
    radius_m = rad * 1852

    # Create a geodesic buffer around the center point
    geod = Geod(ellps="WGS84")
    circle = Point(center_lon, center_lat).buffer(1)  # Create a dummy buffer
    circle = transform(
        lambda x, y: geod.fwd_intermediate(center_lon, center_lat, x, y, radius_m),
        circle
    )
    return circle


def centered_around_airport3(airport,rad: float):
    """
    returns the a circle shape around the airports center. With a radius of rad in NM.
    """
    from shapely.geometry import Point
    from shapely.ops import transform
    from functools import partial
    import pyproj
    point = airports[airport].centroid

    # Define the center of the circle (latitude, longitude) and radius in NM
    center_lat = point.x  # Latitude of Paris
    center_lon = point.y   # Longitude of Paris
        # Radius in nautical miles

    print(center_lat)
    print(center_lon)

    # Convert radius from NM to meters (1 NM = 1852 meters)
    radius_m = rad * 1852

    proj_wgs84 = pyproj.CRS("EPSG:4326")      # Geographic
    proj_l93 = pyproj.CRS("EPSG:2154")

    project_to_l93 = pyproj.Transformer.from_crs(proj_wgs84, proj_l93, always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(proj_l93, proj_wgs84, always_xy=True).transform    

    point_wgs84 = point
    point_l93 = transform(project_to_l93, point_wgs84)

    buffer_l93 = point_l93.buffer(radius_m)
    buffer_wgs84 = transform(project_to_wgs84, buffer_l93)  # back to lat/lon

    return buffer_wgs84

def geodesic_circle(airport, radius_nm, num_points=360):
    point = airports[airport].centroid
    lat = point.x
    lon = point.y
    #radius_km = radius_nm * 1.852  # convert nautical miles to km
    points = []
    meter = radius_nm * 1852
  

    for alpha in range(0, 360, int(360 / num_points)):
        desti_lon,desti_lat,_  = destination(lat,lon,alpha,meter)
        points.append((desti_lon, desti_lat))  # (lon, lat) for Shapely

    return Polygon(points)


def verif_typecodes(t:Traffic)->Counter:
    #returns a counter of all the typecodes in the traffic
    if 'typecode' not in t.data.columns:
        t= t.aircraft_data()
    print(t.data.columns)
    traj_len = len(t[0])
    # type_c = [f.typecode for f in tqdm(t,desc='getting typecodes')]
    type_c = t.data['typecode'].iloc[::traj_len].tolist() #every 200 points we get the label (so for each flight)
    print('|-- Typecodes optained --|')
    type_c = Counter(type_c)
    print('|-- Counter optained --|')
    return type_c


# t = Traffic.from_file(f'results/combined_traff/TO_LFPO_test_first_filter_all_runw_only_7.pkl')
# counter = verif_typecodes(t)
# print(counter.most_common(10))
# t = t.sample(2000)
# print([f.data["RWY"].iloc[0] for f in t])

# t = Traffic.from_file(f'results/combined_traff/TO_LFPO_filtered_all_runways_1_07_cutted_train_d1.pkl')
# print(t.data.columns)

# f = t['0a0046']

# # f = f.resample(100)#.eval(desc='')
# print(f.data)
# t=f
t = Traffic.from_file(f'/home/arnault/Traffic/traffic/results/combined_traff/TO_LFPO_filtered_all_runways_1_07_cutted_d2_stall_t_del.pkl')
# t_og = Traffic.from_file(f'takeoffs_LFPO_07.pkl')
print(len(t))

def compute_time_delta(t: Traffic) -> Traffic:
    """
    Adds a time delta columns to the dataset of every flight in the traffic
    """
    return t.pipe(add_time_deltas).eval(
        desc="computing time deltas", max_workers=4
    )

def add_time_deltas(f: Flight) -> Flight:
    df = f.data
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return f.assign(
        timedelta=(
            pd.to_datetime(df["timestamp"])
            - pd.to_datetime(df["timestamp"].iloc[0])
        ).dt.total_seconds()
    )


# t= compute_time_delta(t)
t.data = t.data.reset_index()

indices = t.data.iloc[199::200].index

    # Step 2: Create the boolean mask on the sliced data
mask = t.data.loc[indices, "timedelta"] > 1200

    # Step 3: Apply the mask to the DataFrame
selected_rows = t.data.loc[indices[mask]]["flight_id"]


print(selected_rows.head(5))
print((t.data["timedelta"].iloc[199::200] > 1200).sum())

print(t['TVF82BL_254113'].data["timedelta"].iloc[-1])
print(t['TVF82BL_254113'].data)

# t_f = t.sample(1)
# t_og_f = t_og.sample(1)

# print(t_f.data)
# print(t_og_f.data)
# print("-------------------------------------------------")
# # print(len(t)) 
# # print(t.data.columns)
# # if 'typecode' not in t.data.columns:
# #     t = t.aircraft_data()
# # # t = t.query('typecode == "A320"')
# print(t["DAH1007_18563"].data)
# # f= t["IBE34HM_35411"].resample(2000)
# # f= f.resample(200)

# # print("resamp \n", f.data)
# # print(t["IBE34HM_35411"].data)
# # print(t["AEA75YQ_34709"].data)
# # t = t.sample(10,seed= 0)
# # t= t[["DAH1007_18563","AEA75YQ_34709","AFR64JN_166601","66ZN_141552"]]



t=t.sample(int(0.05*len(t)))
# t= t.query('RWY == "07"')
# # t= t[["IBE34HM_35411","AEA75YQ_34709","DAH1007_18563"]]
# print(len(t))

# from traffic.core.mixins import PointMixin
# from traffic.data import airports

# def til_thres(flight):
#         g = flight.query("distance > 2*1.852")
#         if g is None:
#             return None
#         return flight.after(g.start)

# def cut_traj_7_LFPO_traff(t:Traffic)-> Traffic:
#     thres_25 = PointMixin()
#     thres_25.latitude, thres_25.longitude =  (airports["LFPO"].runways.data.query("name == '25'").latitude.values[0], airports["LFPO"].runways.data.query("name == '25'").longitude.values[0])
#     # print('threshold: ',thres_25.latitude," ", thres_25.longitude)

#     t_to = t.distance(thres_25).pipe(til_thres).eval(desc = "",max_workers = 20)
#     return t_to

# # t= cut_traj_7_LFPO_traff(t)

# print(len(t))
# # t=compute_time_delta(t)
# # print(len(t))
# # t.data = t.data.reset_index(drop=True)
# # indices = t.data.iloc[199::200].index
# # print(len(indices))
# # filtered_data = t.data.loc[indices]
# # print(len(filtered_data))
# # mask = (filtered_data["timedelta"] > 600) & (filtered_data["timedelta"] < 1200)
# # print("s ",mask.sum())
# # selected_rows = filtered_data.loc[mask, "flight_id"]
# # print(len(selected_rows))
# # cleaned_rows= []
# # print(selected_rows.head(5))
# # for elem in tqdm(selected_rows):
# #     # print(elem)
# #     cleaned_rows.append(elem)

# # print(cleaned_rows)
# # t = t[cleaned_rows]

# print(len(t))

# # def is_aligned_on_ils(f):
# #      return f.aligned_on_ils("LFPO")
     
# # ga = t.has(is_aligned_on_ils).eval(desc = "", max_workers = 10)
# # print(ga)


# # mean_lat = t.data['latitude'].iloc[::200].mean()
# # mean_lon = t.data['longitude'].iloc[::200].mean()

# # print(f'({mean_lat},{mean_lon})')

# # def simple(flight):
# #     return flight.assign(simple=lambda x: flight.shape.is_simple)

# # # last_track = t.data.groupby("flight_id")["track"].last()
# # # id_to_south = last_track[(last_track > 45) & (last_track < 210)].index
# # # t = t[id_to_south]

# # # t = t.iterate_lazy().pipe(simple).eval(desc ="")
# # # t = t.query("simple")

# # print(len(t))

# # # shape = geodesic_circle("LFPO",50)
# # # shape = centered_around_airport3("LFPO",50)
# # # print(shape)
# # # t = t.clip(shape).eval(desc="eval")
# # # print(len(t))
# # # t = t.aircraft_data()
# # # count = Counter([f.typecode for f in tqdm(t)])
# # # print(count)
from traffic.core.mixins import PointMixin

thres_25 = PointMixin()
thres_25.latitude, thres_25.longitude =  (airports["LFPO"].runways.data.query("name == '25'").latitude.values[0], airports["LFPO"].runways.data.query("name == '25'").longitude.values[0])
thres_25.name = "thres 25"
with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    t.plot(ax, alpha=0.1)
    thres_25.plot(ax)





plt.savefig('test_temp_stall.png',dpi = 300)

