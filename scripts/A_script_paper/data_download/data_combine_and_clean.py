from traffic.core import Traffic,Flight
from traffic.data import airports
from traffic.core.mixins import PointMixin
import os
from tqdm import tqdm
import pandas as pd
from collections import Counter

import sys
from datetime import datetime
from traffic.data import airports


os.environ["PYARROW_LARGE_STRING"] = "1"  #other wise to much data


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()


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

def clip_circle(traff,distance,airport):
    shape = centered_around_airport3(airport,distance)
    traff = traff.clip(shape).eval(desc="eval",max_workers =40)
    return traff

# Generate timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"log/output_{timestamp}.log"

# Redirect stdout to both terminal and file
sys.stdout = Tee(sys.stdout, open(log_filename, "w"))


def load_files(path_to_results:str,airport:str,max_workers:int = 8) -> Traffic:
    """
    Loads all the files inside of the folder path to results.
    Store all of these traffic in the returned Traffic object.
    
    path_to_results:
        The path to the folder of interest.
    airport:
        The ICAO of the airport of interest.
    """
    data = pd.DataFrame()
    #traffic= Traffic(data)
    traffic_list = []

    #combining all the traffic files
    for filename in tqdm(os.listdir(path_to_results),desc="combining files"):
        if filename.startswith(f't_TO_{airport}'): #all takeoffs on LFPO 
            fullpath = os.path.join(path_to_results,filename)
            if os.path.isfile(fullpath):
                try :
                    traffic = Traffic.from_file(fullpath)
                    traffic_list.append(traffic)
                except Exception as e:
                    print(f"Error loading {fullpath}: {e}")
                    continue
    traffic = sum(traffic_list)
    #deleting duplicates --
    traffic = traffic.drop_duplicates(subset=["timestamp", "icao24", "callsign"], keep='first')
    print("Duplicates dropped")
    ##adding a flight id


    traffic = traffic.assign_id().eval(desc="Assigning flight ids",max_workers=50)
    print("Flight ids assigned")
    return traffic

def is_aligned_on_ils(f):
     return f.aligned_on_ils("LFPO")
     
def filter_flight_relanding(traffic : Traffic,airport:str = "LFPO")-> Traffic:
    """
    Filters out the flight that relands on the same airport.
    """

    ga = traffic.has(is_aligned_on_ils).eval(desc = "", max_workers = 5).flight_ids
    t_to = traffic.query(
        f"flight_id not in {ga}"
    )
    return t_to

def filter_flight_close_landing(traffic: Traffic)-> Traffic:
    """
    Filter flights that lands within the take off bounding box.
    """
    t_to = traffic
    t_to_phases = t_to.iterate_lazy().phases().eval(desc = "", max_workers = 20)
    id_descent = t_to_phases.query("phase ==  'DESCENT'").flight_ids
    t_to = t_to.query(
    f"flight_id not in {id_descent}"
    )
    return t_to 

# def cut_1NM_runway(traffic:Traffic, airport:str = "LFPO",corresponding_runway:str ="25")-> Traffic:
#     """
#     Cuts the trajectory 1NM after the runway start.
#     """

#     from traffic.core.mixins import PointMixin

#     thres_25 = PointMixin()
#     thres_25.latitude, thres_25.longitude =  (airports[airport].runways.data.query(f"name == '{corresponding_runway}'").latitude.values[0], airports["LFPO"].runways.data.query("name == '25'").longitude.values[0])
#     t_to = traffic

#     def til_thres(flight):
#         g = flight.query("distance > 2*1.852")
#         if g is None:
#             return None
#         return flight.after(g.start)


#     t_to = t_to[idx].distance(thres_25).pipe(til_thres).eval(desc = "")
#     return t_to

def add_runway_to_flight(f):
    takeoff = f.takeoff("LFPO").next()
    if takeoff:
        f = f.assign(RWY = lambda _: takeoff.runway_max )
        f= f.assign(to_start = lambda _: takeoff.start )
    return f

def change_traffic(traffic_to_add:Traffic, traffic_to_mod:Traffic,seq_start:int,seq_end:int)->Traffic:
    data_add = traffic_to_add.data
    data_to_mod = traffic_to_mod.data
    seq_len = len(traffic_to_mod[0])
    # print(data_add[["RWY","to_start"]].head())
    # print(data_to_mod.dtypes)
    # print(data_add.dtypes)
    # print(len(data_to_mod))
    # print(len(data_to_mod.iloc[seq_start*seq_len:seq_end*seq_len,data_to_mod.columns.get_indexer(["RWY","to_start"])]))
    data_to_mod.iloc[seq_start*seq_len:seq_end*seq_len,data_to_mod.columns.get_indexer(["RWY","to_start"])] = data_add[["RWY","to_start"]]
    return Traffic(data_to_mod)

def sep_traff_sub(traff: Traffic, name:str,num_sep = 40)->list[str]:
    num_f = len(traff)//num_sep
    list_names = []
    i = 0
    name = name.split('.')[0]
    for i in tqdm(range(num_sep-1),desc = 'splitting'):
        t = traff[i*num_f:(i+1)*num_f]
        t = t.to_pickle(f'{name}_{i}.pkl')
        list_names.append(f'{name}_{i}.pkl')
    
    if len(traff)%num_sep != 0:
        t = traff[(i+1)*num_f:]
        t = t.to_pickle(f'{name}_{(i+1)}.pkl')
        list_names.append(f'{name}_{(i+1)}.pkl')

    return list_names
        
    

def add_runways(traff:Traffic)->Traffic:
    new_traff = traff.pipe(add_runway_to_flight).eval(desc="Adding runways to dataset",max_workers = 40)
    return new_traff

def add_runways_load_traffics(names:list[str])->list[str]:
    list_names =[]
    for name in tqdm(names,desc='adding runways'):
        new_name = name.split('.')[0] + f'_runways.pkl'
        t = Traffic.from_file(name)
        t = add_runways(t)
        t.to_pickle(new_name)
        list_names.append(new_name)
    
    return list_names
    



def add_runway_to_traffic(traff:Traffic,divide = 1):

    traff_size = len(traff)//divide
    traff = traff.assign(RWY = lambda _: None)
    traff = traff.assign(to_start = lambda _: None)
    # traffs = []  
    for i in tqdm(range(divide),desc="step"):
        new_traff = traff[i*traff_size:(i+1)*traff_size]
        print("data_splited")
        print(len(new_traff))
        new_traff = new_traff.pipe(add_runway_to_flight).eval(desc="Adding runways to dataset",max_workers = 40)
        # traffs.append(new_traff)
        traff=change_traffic(new_traff,traff,i*traff_size,(i+1)*traff_size)
        print("t",traff[i*traff_size:(i+1)*traff_size].data["to_start"].head(5))

    if (len(traff) % divide != 0):
        new_traff = traff[divide * traff_size:]
        new_traff = new_traff.pipe(add_runway_to_flight).eval(desc="Adding runways to dataset",max_workers = 40)
        # traffs.append(new_traff)
        # change_traffic(new_traff,traff)
        traff = change_traffic(new_traff,traff,divide * traff_size,len(traff))
    
    # new_traff = sum(traffs)
    return traff

# def filter_runway(traffic:Traffic, runway_num:str) -> Traffic,list:
    
#     #finding take offs
#     to_LFPO = list()
#     traffic = traffic.aircraft_data()
#     for f in tqdm(traffic):
#         if takeoff := f.takeoff("LFPO").next():
#             to_LFPO.append(
#                 {
#                     "callsign": f.callsign,
#                     "icao24": f.icao24,
#                     "flight_id": f.flight_id,
#                     "airport": "LFPO",
#                     "start": takeoff.start,
#                     "RWY": takeoff.runway_max, #shortcut
#                 })

#     takeoffs_LFPO = pd.DataFrame.from_records(to_LFPO)
#     print(takeoffs_LFPO.RWY.value_counts())

#     idx = takeoffs_LFPO.query(f"RWY == '{runway_num}'").flight_id.tolist()
#     return traffic[idx]



def til_thres(flight):
        g = flight.query("distance > 1.5* 1.852")
        if g is None:
            return None
        return flight.after(g.start)

def cut_traj_7_LFPO_traff(t:Traffic)-> Traffic:
    thres_25 = PointMixin()
    thres_25.latitude, thres_25.longitude =  (airports["LFPO"].runways.data.query("name == '25'").latitude.values[0], airports["LFPO"].runways.data.query("name == '25'").longitude.values[0])
    # print('threshold: ',thres_25.latitude," ", thres_25.longitude)

    t_to = t.distance(thres_25).pipe(til_thres).eval(desc = "",max_workers = 20)
    return t_to


def cut_traj_7_LFPO_load_traffics(names:list[str],runway:str="07")->list[str]:
    list_names =[]
    for name in tqdm(names,desc='cutting'):
        new_name = name.split('.')[0] + f'_filtered.pkl'
        t = Traffic.from_file(name)
        t= t.query(f'RWY == "{runway}"')
        # t = t.resample('').eval(desc="resampling",max_workers=40) already resampled at 1s
        t = cut_traj_7_LFPO_traff(t)
        # print(t.data.columns)
        t = t.resample(200).eval(desc="resampling",max_workers=40)
        t.to_pickle(new_name)
        list_names.append(new_name)
    
    return list_names



def cut_traj_7_LFPO(traff:Traffic,divide = 1)-> Traffic:
    thres_25 = PointMixin()
    thres_25.latitude, thres_25.longitude =  (airports["LFPO"].runways.data.query("name == '25'").latitude.values[0], airports["LFPO"].runways.data.query("name == '25'").longitude.values[0])
    print('threshold: ',thres_25)

    traff_size = len(traff)//divide
    traffs = []  
    for i in tqdm(range(divide),desc="step"):
        t_to = traff[i*traff_size:(i+1)*traff_size]
        t_to = t_to.distance(thres_25).pipe(til_thres).eval(desc = "",max_workers = 40)
        traffs.append(t_to)

    if (traffic % divide != 0):
        t_to = traff[divide * traff_size:]
        t_to = t_to.distance(thres_25).pipe(til_thres).eval(desc = "",max_workers = 40)
        traffs.append(t_to)

    t_to = sum(traffs)
    return t_to
    

def train_test_split(traff:Traffic,split:float =0.8,seed:int = None)->tuple[Traffic,Traffic]:
    """
    Splits the traffic into a test and train traffic
    """
    num_flight = int(len(traff)*(1-split))

    test_set = traff.sample(size = num_flight,seed = seed)
    test_ids = test_set.flight_ids
    train_set = traff.query(f"flight_id not in {test_ids}")

    return train_set,test_set  


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

def shape_circle_load_traffics(names:list[str],airport:str = "LFPO",rad:float = 50)->list[str]:
    list_names =[]
    shape = centered_around_airport3(airport,rad)
    for name in tqdm(names,'bounding'):
        new_name = name.split('.')[0] + f'_circle.pkl'
        t = Traffic.from_file(name)
        t = t.clip(shape).eval(desc="eval",max_workers = 40)
        t.to_pickle(new_name)
        list_names.append(new_name)
    
    return list_names

def recombine(names : list[str])->Traffic:
    traff = []
    for name in tqdm(names,desc='recombining'):
        t = Traffic.from_file(name)
        traff.append(t)
    return sum(traff)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df= df.copy()
    for col in df.columns:
        dtype_str = str(df[col].dtype)

        if "double" in dtype_str:
            df[col] = df[col].astype(float)

        elif "string" in dtype_str:
            df[col] = df[col].astype(str)

        elif "bool" in dtype_str:
            df[col] = df[col].astype(bool)

        elif "timestamp" in dtype_str:
            # Use utc=True to keep the timezone, or remove it with utc=False
            df[col] = pd.to_datetime(df[col], utc=True)

        # No conversion for 'object' or already-native types
    return df

def clear_na_flight(f:Flight,rate_delete=0.05):
    data = f.data
    n= len(data)
    data_new = data.dropna(subset=['latitude','longitude','groundspeed','altitude','track'])
    if n - len(data_new) > int(rate_delete * n):
        return None
    
    return Flight(data_new)

def clear_na_traff(t:Traffic,rate_delete=0.10):
    return t.pipe(clear_na_flight,rate_delete=rate_delete).eval(desc='deleting NAN',max_workers=20)


def true_bound(traff : Traffic, airport_name:str = 'LFPO', dist:float = 50,rsample:int = 1000) -> Traffic:
    airport_center = airports[airport_name]
    len_air = 200

    traff = Traffic(clean_dataset(traff.data))

    traff = clear_na_traff(traff)

    print("|-- Cleared NANs --|")
    traff = traff.resample('1s',how = 'interpolate').eval(desc = 'resampling', max_workers = 20)
    print("|-- finished resampling --|")

    traff = traff.distance(other = airport_center,column_name = 'dist_airp').eval(desc = 'computing distances', max_workers = 1)
    traff = traff.query(f"dist_airp <= {dist}")
    print("|-- finished bounding --|")
    # traff = traff.resample(len_air).eval(desc = 'resampling 2', max_workers = 20)
    # print("|-- finished resampling --|")
    return traff

def true_bound_load(traff_l : list[str], airport_name:str = 'LFPO', dist:float = 50,rsample:int = 1000)-> list[str]:
    list_names =[]
    for name in tqdm(traff_l,'bounding'):
        new_name = name.split('.')[0] + f'_bounded.pkl'
        t = Traffic.from_file(name)
        t = true_bound(t,airport_name,dist,rsample)
        t.to_pickle(new_name)
        list_names.append(new_name)
    
    return list_names


def add_time_deltas(f: Flight,time_limit:int=1200) -> Flight:
        df = f.data
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        f = f.assign(
            timedelta=(
                pd.to_datetime(df["timestamp"])
                - pd.to_datetime(df["timestamp"].iloc[0])
            ).dt.total_seconds()
        )
        return f if f.data["timedelta"].iloc[-1] < time_limit else None

def compute_time_delta(t: Traffic) -> Traffic:
        """
        Adds a time delta columns to the dataset of every flight in the traffic
        """
        return t.pipe(add_time_deltas).eval(
            desc="computing time deltas", max_workers=1
        )






def main():
    """
    Loading all the files and combining them.
    Saving the traffic in a new folder.
    """

    import warnings
    warnings.filterwarnings("ignore", message="nanoseconds in conversion")
    
    path_to_results = 'results'
    save_path = 'results/combined_traff'
    airport = 'LFPO'
    # traffic = load_files(path_to_results,airport)
    traffic = Traffic.from_file('results/TO_LFPG_20210910_20210917.pkl')

    # print(traffic[4].data.isna().any().any())
    # print(traffic[4].data.isna().any().any())
    # nan_rows = traffic[4].data[traffic[4].data.isna().any(axis=1)]
    # print(nan_rows)

    # counts = []
    # for f in traffic:
    #     df = f.data
    #     count = df.isna().any(axis=1).sum() / len(df)
    #     counts.append(count)
    #     # print(count)

    # import statistics

    # mean = statistics.mean(counts)
    # median = statistics.median(counts)

    # print(f"Mean: {mean}")
    # print(f"Median: {median}")

    # # traffic.to_pickle(f'{save_path}/TO_{airport}_test_all.pkl')
    # # print("Combining done")

    # traffic = Traffic.from_file(f'{save_path}/TO_{airport}_test_all.pkl')
    # print('loaded')
    # traffic = filter_flight_relanding(traffic,airport)
    # print(len(traffic))
    # print("filtered_relanding")
    # print(len(traffic))
    # traffic.to_pickle(f'{save_path}/TO_{airport}_test_first_filter_reland_all.pkl')
    # traffic = filter_flight_close_landing(traffic)
    # print("filtered close landings")
    # print(len(traffic))

    # # traffic = cut_1NM_runway(traffic)

    # # print("trajectories clamped")
    # traffic.to_pickle(f'{save_path}/TO_{airport}_test_first_filter_all.pkl')

    # traffic = Traffic.from_file(f'{save_path}/TO_{airport}_test_first_filter_all.pkl')
    # #print(traffic.data.dtypes)

    # print("traffic loaded")

    # name = f'{save_path}/temp/TO_{airport}.pkl'

    # small_traffs = sep_traff_sub(traffic,name,70)

    # del traffic
    # small_traffs = [f"{save_path}/temp/TO_{airport}_{i}.pkl" for i in range(70)]
    # file_names = add_runways_load_traffics(small_traffs)
    

    # print('Files separated')

    # file_names = cut_traj_7_LFPO_load_traffics(file_names)

    # print('Traj cutted')
#-------------------------------------------------------------------------------------------------------------------------------------
    # file_names = [f"{save_path}/temp/TO_{airport}_{i}_runways.pkl" for i in range(70)]
    # # # final_traff = recombine(file_names)

    # # final_traff = recombine(file_names)

    # # # print('Traj recombined')

    # # # final_traff = true_bound(final_traff)
    # traffs_bound = true_bound_load(file_names)
    # print('Traj bounded')
    # # # final_traff.to_pickle(f'{save_path}/TO_{airport}_filtered_all_runways_1.pkl')
    # runway_num = '07'
    # # # # final_traff= Traffic.from_file(f'{save_path}/TO_{airport}_filtered_all_runways_1.pkl')
    # final_traff = recombine(traffs_bound)
    # print(final_traff["BE34HM_35411"].data)
    # print(final_traff["TVF18PL_250574"].data) #stall at the end


    # t_f = final_traff.sample(5)
    # for f in t_f:
    #     print(f.data,"\n----------------")
    

    # print(final_traff[ "TVF67QY_205811"].data) #planes that stall
    # print(final_traff[ "TVF67DV_257474"].data)

    # # final_traff = final_traff.query(f"RWY == '{runway_num}'")

    # final_traff = Traffic.from_file(f'{save_path}/TO_{airport}_test_all.pkl')
    # # traffic = Traffic.from_file(f'{save_path}/TO_{airport}_test_all.pkl')
    
    # print(final_traff[:10].data.isna().any().any())
    # print(final_traff[:10].data.isna().any().any())
    # nan_rows = final_traff[:10].data[final_traff[:10].data.isna().any(axis=1)]
    # print(nan_rows)


    # name = f'{save_path}/temp/TO_{airport}_filtered_all_runways_1_07sep.pkl'

    # small_traffs = sep_traff_sub(final_traff,name,70)

    # small_traffs = [f'{save_path}/temp/TO_{airport}_filtered_all_runways_1_07sep_{i}.pkl' for i in range(70)]
    # small_traffs = [f'{save_path}/temp/TO_LFPO_{i}_runways_bounded.pkl' for i in range(70)]
    # file_names = cut_traj_7_LFPO_load_traffics(small_traffs) #error was here
    # final_traff = recombine(file_names)

    # final_traff.to_pickle(f'{save_path}/TO_{airport}_filtered_all_runways_1_07_cutted_d2_stall.pkl')

    final_traff = Traffic.from_file(f'{save_path}/TO_{airport}_filtered_all_runways_1_07_cutted_d2_stall.pkl')


    
    final_traff = compute_time_delta(final_traff)

    final_traff.to_pickle(f'{save_path}/TO_{airport}_filtered_all_runways_1_07_cutted_d2_stall_t_del.pkl')
    print('saved')

    train, test = train_test_split(final_traff,0.8,seed = 0)

    train.to_pickle(f'{save_path}/TO_{airport}_filtered_all_runways_1_07_cutted_train_d2_stall_t_del.pkl')
    test.to_pickle(f'{save_path}/TO_{airport}_filtered_all_runways_1_07_cutted_test_d2_stall_t_del.pkl')



#---------------------------------------------------------------------------------------------------





    # final_traff.to_pickle(f'{save_path}/TO_{airport}_test_first_filter_all_runw.pkl')
    # runway_num = '07'
    # traffic = Traffic.from_file("results/combined_traff/TO_LFPO_test_first_filter_all_runw.pkl")
    # traffic = traffic.query(f"RWY == '{runway_num}'")
    # traffic.to_pickle(f'results/combined_traff/TO_LFPO_test_first_filter_all_runw_only_7.pkl')

    # traffic = Traffic.from_file('results/combined_traff/TO_LFPO_test_first_filter_all_runw_only_7.pkl')
    # print("loaded")
    # traffic = true_bound(traffic,'LFPO',50)
    # traffic.to_pickle(f'results/combined_traff/TO_LFPO_test_first_filter_all_runw_only_7_t_bound.pkl')


    # traffic = Traffic.from_file(f'results/combined_traff/TO_LFPO_test_first_filter_all_runw_only_7_t_bound.pkl')

    # train,test = train_test_split(traffic,0.8,0)

    # traffic = add_runway_to_traffic(traffic,divide = 40)
    # print(traffic.data.dtypes)
    # traffic.to_pickle(f'{save_path}/TO_{airport}_test_first_filter_w_runway_all.pkl')

    # print("runway added")

    # runway_num = '07'
    # traffic = traffic.query(f"RWY == '{runway_num}'")
    # print(len(traffic))

    # traffic.to_pickle(f'{save_path}/TO_{airport}_test_first_filter_runway_07_all.pkl')


    # traffic = Traffic.from_file(f'{save_path}/TO_{airport}_test_first_filter_runway_07_all.pkl')
    # print("traffic loaded")
    
    # traffic = cut_traj_7_LFPO(traffic,divide = 40)
    # print("traffic cuted")
    # traffic.to_pickle(f'{save_path}/TO_{airport}_test_first_filter_runway_07_cuted_all.pkl')


    






if __name__ == "__main__":
    main()