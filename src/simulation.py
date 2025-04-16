import pickle
from datetime import datetime, timedelta, timezone, time
from typing import Iterator, Optional
import pandas as pd
from tqdm import tqdm

import minisky

from traffic.core import Flight, Traffic
from traffic.data import navaids
import csv

START = "0:00:00>"


# -----Code retaken from Thimothé Krauth : https://github.com/kruuZHAW/deep-traffic-generation-paper.git
def aligned_stats(traj: "Flight") -> Optional[pd.DataFrame]:
    navaids_extent = navaids.extent(traj, buffer=0.1)
    if navaids_extent is None:
        return None

    df = pd.DataFrame.from_records(
        list(
            {
                "start": segment.start,
                "stop": segment.stop,
                "duration": segment.duration,
                "navaid": segment.max("navaid"),
                "distance": segment.min("distance"),
                "bdiff_mean": segment.data[
                    "shift"
                ].mean(),  # b_diff_mean is equivalent to shift.mean() now ?
                "bdiff_meanp": segment.data["shift"].mean() + 0.02,
            }
            for segment in traj.aligned_on_navpoint(
                list(navaids_extent.drop_duplicates("name"))
            )
        )
    )

    if df.shape[0] == 0:
        return None
    return df.sort_values("start")


def groupby_intervals(table: pd.DataFrame) -> "Iterator[pd.DataFrame]":
    if table.shape[0] == 0:
        return
    table = table.sort_values("start")
    sweeping_line = table.query(
        "stop <= stop.iloc[0]"
    )  # take as much as you can
    # then try to push the stop line: which intervals overlap the stop line
    additional = table.query("start <= @sweeping_line.stop.max() < stop")

    while additional.shape[0] > 0:
        sweeping_line = table.query("stop <= @additional.stop.max()")
        additional = table.query("start <= @sweeping_line.stop.max() < stop")

    yield sweeping_line
    yield from groupby_intervals(
        table.query("start > @sweeping_line.stop.max()")
    )


def reconstruct_navpoints(traj: "Flight") -> "Iterator[pd.DataFrame]":
    table = aligned_stats(traj)
    if table is None:
        return
    for block in groupby_intervals(table):
        t_threshold = block.eval("duration.max()") - pd.Timedelta(  # noqa: F841
            "30s"
        )
        yield (
            block.sort_values("bdiff_mean")
            .query("duration >= @t_threshold")
            .head(1)
        )


def navpoints_table(flight: "Flight") -> Optional["Flight"]:
    from traffic.data import navaids

    navaids_extent = navaids.extent(flight, buffer=0.1)
    if navaids_extent is None:
        return None

    list_ = list(reconstruct_navpoints(flight))
    if len(list_) == 0:
        print(f"fail with {flight.flight_id}")
        return None

    navpoints_table = pd.concat(list(reconstruct_navpoints(flight))).merge(
        navaids_extent.drop_duplicates("name").data,
        left_on="navaid",
        right_on="name",
    )

    cd_np = navaids_extent.drop_duplicates("name")

    try_list = list(
        (i, cd_np[elt.navaid], elt.stop, elt.duration)
        for i, elt in navpoints_table.assign(
            delta=lambda df: df.start.shift(-1) - df.stop
        )
        .drop(
            columns=[
                "altitude",
                "frequency",
                "magnetic_variation",
                "description",
            ]
        )
        .query('delta > "30s"')
        .iterrows()
    )
    for i, fix, stop, duration in try_list:
        cd = (
            flight.after(stop)
            .first(duration)  # type: ignore
            .assign(track=lambda df: df.track + 180)
            .aligned_on_navpoint([fix])
            .next()
        )
        if cd is not None:
            navpoints_table.loc[i, ["stop", "distance"]] = (
                cd.stop,
                -cd.distance_max,
            )

    return Flight(
        navpoints_table.assign(
            flight_id=flight.flight_id,
            callsign=flight.callsign,
            icao24=flight.icao24,
            registration=flight.registration,
            typecode=flight.typecode,
            # runway=flight.runway_max,
            coverage=navpoints_table.duration.sum().total_seconds()
            / flight.duration.total_seconds(),
            latitude_0=flight.at_ratio(0).latitude,  # type: ignore
            longitude_0=flight.at_ratio(0).longitude,  # type: ignore
            altitude_0=flight.at_ratio(0).altitude,  # type: ignore
            track_0=flight.at_ratio(0).track,  # type: ignore
            groundspeed_0=flight.at_ratio(0).groundspeed,  # type: ignore
        ).drop(
            columns=[
                "bdiff_mean",
                "bdiff_meanp",
                "frequency",
                "magnetic_variation",
                "description",
                "altitude",
            ]
        )
    )


# --- End of retaken code


# --- Code to create bluesky SCN files


def lvnav(acid: str) -> str:
    """
    Return the LNAV VNAV instructions as a scenario command
    """
    line = START + f"LNAV {acid} ON \n"
    line += START + f"VNAV {acid} ON \n"
    return line


def cre(
    acid: str,
    type: str,
    lat: float,
    lon: float,
    hdg: float,
    alt: float,
    groundspeed: float,
) -> str:
    """
    Returns the CRE command for scenarios
    """
    return (
        START
        + f"CRE {acid}, {type}, {lat}, {lon}, {hdg}, {alt}, {groundspeed} \n"
    )


def init_log(log_name: str, file_name: str, log_time: int) -> str:
    """
    Returns the command used to initiate a logger.
    """
    line = START + f"CRELOG {log_name} {log_time} {file_name} \n"
    line += START + f"{log_name} ADD lat, lon, alt, cas \n"
    line += START + f"{log_name} ON \n"
    return line


def defwpt(wpt_name: str, lat: float, lon: float, type: str = "FIX") -> str:
    """
    Defining a waypoint
    """
    return START + f"DEFWPT {wpt_name} {lat} {lon} {type} \n"


def addwpt(acid: str, wpt_name: str, alt: float, groundspeed: float) -> str:
    """
    Command to add the wpt to the route of the aircraft
    """
    return START + f"ADDWPT {acid} {wpt_name}, {alt}, {groundspeed} \n"


def get_info(fl: Flight, navpoints: Flight) -> pd.DataFrame:
    """
    Take a flight and the navpoint table. Returns the points of the flight
    that are the closest to each navpoint.
    Used to retrieve the altitude and speed at each navpoint.
    """
    navpoints_data = navpoints.data
    f_data = fl.data
    infos = []

    nav_aid = list(navaids.extent(fl, buffer=0.1).drop_duplicates("name"))

    dic_navaids = {nav.name: nav for nav in nav_aid}

    for i, row in navpoints_data.iterrows():
        item = fl.closest_point(dic_navaids[row["name"]])
        item["start"] = row["start"]
        infos.append(item.to_frame().T)

    return pd.concat(infos)


def def_navaids(nav_points: Traffic) -> str:
    """
    Creates the string for the navaids definition
    """
    print(nav_points.data.columns)
    nav = nav_points.data.drop_duplicates("name")
    ret = ""
    for i, row in nav[["name", "latitude", "longitude"]].iterrows():
        ret += defwpt(row["name"], row["latitude"], row["longitude"])
    return ret


def convert_time_delta_to_start_str(time_delta: int) -> str:
    """
    convert tim delta into the format to be used in a scenario
    """

    time = str(timedelta(seconds=float(time_delta)))

    if len(time) == 7:
        time = "0" + time
    return time


def delwpt(acid: str, wpt_name: str) -> str:
    """
    delete the waypoint from the designated flight trajectory
    """
    return START + f"DELWPT {acid} {wpt_name} \n"


def schedule(time: int, command: str) -> str:
    """
    schedule a command at a specifc time
    """
    return (
        START
        + f"SCHEDULE {convert_time_delta_to_start_str(time)} {command[len(START) :]} \n"
    )


def gen_instruct_f(flight: Flight, nav_p: Flight, path_log: str,typecode: str = "") -> str:
    """
    return the instruct for a specific flight in a scenario
    """
    f_data = flight.data
    acid = f_data.iloc[0]["flight_id"]
    text = cre(
        acid,
        typecode,
        f_data.iloc[0]["latitude"],
        f_data.iloc[0]["longitude"],
        f_data.iloc[0]["track"],
        f_data.iloc[0]["altitude"],
        f_data.iloc[0]["groundspeed"],
    )
    # nav_p = navpoints_table(flight)
    if nav_p is None:
        print("none")
        with open(path_log.split(".")[0] + "_denied_flight.txt", "a") as file:
            file.write(flight.flight_id + "\n")
        return ""
    points = get_info(flight, nav_p)
    prev_wpt = ""
    previous_start = points.iloc[0]["start"]

    for i, (_, row) in enumerate(points.iterrows()):
        # print(row)

        if i != 0:
            start = row["start"]
            diff_time = (start - previous_start).total_seconds()
            text += addwpt(
                acid, row["point"], row["altitude"], row["groundspeed"]
            )  # directing the trajectory towards the waypoint
            text += schedule(
                diff_time, delwpt(acid, prev_wpt)
            )  # deleting the previous waypoint, useful if the waypoint is never reached
        else:
            text += addwpt(
                acid, row["point"], row["altitude"], row["groundspeed"]
            )
        prev_wpt = row["point"]

    text += lvnav(acid)
    return text


def get_final_wpt_name(traff: Traffic) -> tuple[float, float]:
    f = traff[0]
    nav_p = navpoints_table(f)
    print(nav_p.data)
    print(nav_p.data.iloc[0]["name"])
    return nav_p.data.iloc[0]["latitude"], nav_p.data.iloc[0]["longitude"]


def time_to_sec(t: time) -> int:
    """
    Takes a time object and returns the total number of seconds elapsed since 00:00:00.
    """
    return (
        t.hour * 3600
        + t.minute * 60
        + t.second
        + int(round(t.microsecond / 1e6))
    )


def scn_file_to_minisky(scn_file: str) -> tuple[str, int]:
    """
    Takes a well organized scn file and returns the instrcution as text for use by minisky.
    In order to work the three first line must define and start the logger.
    The last two lines of the file must schedule the turn off the logger and quit.
    Also returns the time to be elpased in seconds.
    """
    with open(scn_file, "r") as file:
        text = file.read()

    text = "\n".join([line for line in text.splitlines() if line.strip()])
    text = text.replace(START, "")
    time_test = text.splitlines()[-1].split(" ")[1]
    text = "\n".join([line for line in text.splitlines()][3:-2])
    # with open(scn_file.split('.')[0] +'no_start.scn',"w") as file:
    #      file.write(text)

    time_test = (
        "0" + time_test if len(time_test.split(".")[0]) < 8 else time_test
    )
    time_test_f = time_to_sec(time.fromisoformat(time_test))
    return text, time_test_f


def launch_simulation(
    scn_file: str, log_file: str, log_step: int, n_write_step: int = 100
) -> None:
    """
    Simulates the snc_file using minisky and logs the result in log_file.
    Problem : VERY SLOW
    """
    text, total_time = scn_file_to_minisky(scn_file)

    minisky.init()
    minisky.sim.reset()
    for line in tqdm(text.splitlines(), desc="Initialization"):
        minisky.stack.stack(line)
    print("|Finished Init|")

    minisky.sim.simdt = log_step
    buffer = []

    with open(log_file, mode="w", newline="") as file:
        file.write("#\n#\n")  # 2first lines
        writer = csv.writer(file)
        with tqdm(
            total=total_time, desc="Simulation Progress", unit="s"
        ) as pbar:
            while minisky.sim.simt < total_time:
                minisky.sim.step()
                data_step = [
                    [
                        minisky.sim.simt,
                        minisky.traf.lat[i],
                        minisky.traf.lon[i],
                        minisky.traf.alt[i],
                        minisky.traf.cas[i],
                    ]
                    for i in range(len(minisky.traf.lat))
                ]
                buffer.extend(data_step)

                if (
                    minisky.sim.simt != 0
                    and ((minisky.sim.simt / log_step) % n_write_step) == 0
                ):
                    writer.writerows(buffer)
                    buffer.clear()
                pbar.update(minisky.sim.simt - pbar.n)
        if len(buffer) != 0:
            writer.writerows(buffer)

    return


def compute_alt(traff: Traffic, init_alt: float) -> Traffic:
    """
    Compute the altitude at every time_stamp thanks to the vertical
    rate and the altitude at t =0.
    """
    data = traff.data
    data["altitude"] = init_alt + data["vertical_rate"] * (
        data["timedelta"] / 60
    )
    return Traffic(data)


class Simulator:
    """
    Object to be used for launching a simulation. Can create scn files, and interpret logged files.
    """

    def __init__(self, traff: Traffic, initial_alt: int = 500):
        self.traff = traff
        if "typecode" not in self.traff.data.columns:
            self.traff = self.traff.aircraft_data()
        if "altitude" not in self.traff.data.columns:
            self.traff = compute_alt(self.traff, initial_alt)
        self.set_nave = set()

    def max_dt(self) -> int:
        """
        max time delta of the traffic
        """
        return self.traff.data["timedelta"].max()

    def navpoints(self) -> Traffic:
        t = (
            self.traff.iterate_lazy()
            .pipe(navpoints_table)
            .eval(desc="navpoints table", max_workers=6)
        )
        t.to_csv("data_orly/temp_navp/tcvae_generation_navpoints.csv")
        return t

    def load_navpoints(self, path: str) -> Traffic:
        """
        Loads the navpoints from a csv file
        """
        t_data = pd.read_csv(path, header=0)
        t_data["start"] = pd.to_datetime(t_data["start"], utc=True)
        t_data["stop"] = pd.to_datetime(t_data["stop"], utc=True)
        t_data["duration"] = pd.to_timedelta(t_data["duration"])
        t = Traffic(t_data)
        return t

    def full_test(self, path: str, log_f_name: str, log_time: int) -> Traffic:
        """
        Creates the scn file, then simulate the traff, and returns the traffic simulated.
        Can take a long time to run
        """

        self.scenario_create(path, log_f_name, log_time)
        launch_simulation(path, log_f_name, log_time)
        traff = self.read_csv_log_file(
            log_f_name, path.split(".")[0] + "_denied_flight.pkl"
        )
        return traff

    def scenario_create(
        self,
        path: str,
        log_f_name: str,
        log_time: int = 10,
        load_file: str = "",
        typecode: str = "",
    ) -> None:
        """
        creates the scenario file related to the traffic
        """
        max_delta = self.max_dt()
        t = (
            self.navpoints()
            if load_file == ""
            else self.load_navpoints(load_file)
        )
        traff = self.traff
        text = init_log("MY_LOG", log_f_name, log_time) + "\n"
        text += def_navaids(t) + "\n"
        list_flights = []
        for e in tqdm(t, desc="writing text in memory"):
            f = traff[e.data.iloc[0]["flight_id"]]
            text += gen_instruct_f(f, e, log_f_name,typecode)
            list_flights.append(f.flight_id)

        with open(
            path.split(".")[0] + "_denied_flight.pkl", "wb"
        ) as file:  # saving the flight ids for future reference
            pickle.dump(list_flights, file)

        text += schedule(max_delta, START + "MY_LOG OFF \n")
        text += schedule(max_delta, START + "QUIT\n")
        with open(path, "w") as file:
            file.write(text)

    def read_csv_log_file(
        self, path: str, flight_ids_paths: str = ""
    ) -> Traffic:
        """
        Return the reconstructed traffic from the simulation log file.
        Takes in account the flight for which simulation was impossible
        """
        # reading the data
        data = pd.read_csv(
            path,
            sep=",",
            header=None,
            names=[
                "timedelta",
                "latitude",
                "longitude",
                "altitude",
                "groundspeed",
            ],
            skiprows=2,
        )

        with open(flight_ids_paths, "rb") as f:
            flight_ids = pickle.load(f)

        print(flight_ids)
        print(data)
        num_flight = len(data[data["timedelta"] == 0])
        # check if we have the exact sam enumber of point for each flight
        if len(data) % num_flight != 0:
            print(len(data) % num_flight)
            data = data.iloc[: len(data) - (len(data) % num_flight)]
        new_df = pd.concat(
            [data.iloc[i::num_flight] for i in range(num_flight)],
            ignore_index=True,
        )

        len_flight = len(data) // num_flight
        flight_list = flight_ids  # keeping the right order
        f_ids = [
            flight_list[i // len_flight] for i in range(len(new_df))
        ]  # getting the list of flight ids
        new_df["flight_id"] = f_ids
        new_df["timestamp"] = datetime(
            year=2025, month=1, day=1, tzinfo=timezone.utc
        ) + pd.to_timedelta(new_df["timedelta"], unit="s")
        new_df["altitude"] = new_df["altitude"] * 3.28084  # from meters to feet
        new_df["groundspeed"] = new_df["groundspeed"] * 1.94384  # M/S to KTS
        f_traff = Traffic(new_df)
        return f_traff
