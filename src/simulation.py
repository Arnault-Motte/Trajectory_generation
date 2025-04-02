from typing import Iterator, Optional

import pandas as pd

from traffic.core import Flight, Traffic

import minisky.minisky.stack
import minisky.minisky.traffic
import minisky
from datetime import datetime

def aligned_stats(traj: "Flight") -> Optional[pd.DataFrame]:
    from traffic.data import navaids

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
        yield block.sort_values("bdiff_mean").query(
            "duration >= @t_threshold"
        ).head(1)

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
            #runway=flight.runway_max,
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

class Simulator:
    def __init__(self):
        pass

    def generate_sim_instruct(f: Flight)-> Flight:
        nav_p = navpoints_table(f)
        data = nav_p.data
        lat_0 = data.loc[0,'latitude_0']
        lon_0 = data.loc[0,'longitude_0']
        alt_0 = data.loc[0,'altitude_0']
        gs_0 = data.loc[0,'ground_speed_0']
        t_0 = 0

        dic = {"latitude":[lat_0],'longitude':[lon_0],'altitude':[alt_0],'groundspeed':[gs_0],'time'=[datetime()]}




    def simulate_flight(f: Flight)->Flight:
        pass
