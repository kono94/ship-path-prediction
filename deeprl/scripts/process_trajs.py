# %%
import pandas as pd
import movingpandas as mpd
import glob
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from datetime import datetime
import folium
from folium import plugins
from folium.features import DivIcon
import geopandas
from datetime import datetime, timedelta
from tqdm import tqdm
from shapely.geometry import Polygon

def resample_and_interpolate(trips, resample_interval="5S", interpolate_method="linear"):
    out = pd.DataFrame()

    for t in tqdm(trips.trajectories):
        tmp = t.df.resample(resample_interval, origin="end").mean()
        tmp["mmsi"] = tmp["mmsi"].interpolate(method="bfill")
        tmp["type"] = tmp["type"].interpolate(method="bfill")
        tmp["sog"] = tmp["sog"].interpolate(method=interpolate_method)
        tmp["cog"] = tmp["cog"].interpolate(method=interpolate_method)
        tmp["lat"] = tmp["lat"].interpolate(method=interpolate_method)
        tmp["lon"] = tmp["lon"].interpolate(method=interpolate_method)
        tmp["heading"] = tmp["heading"].interpolate(method=interpolate_method)
        tmp["speed"] = tmp["speed"].interpolate(method=interpolate_method)
        tmp["direction"] = tmp["direction"].interpolate(method=interpolate_method)
        tmp["traj_id"] = str(tmp["mmsi"][0]) + str(tmp.iloc[0].name)
        tmp["timestamp"] = tmp.index
        tmp = tmp.reset_index(drop=True)
        out = out.append(tmp, ignore_index=True)
        
    return out

months = ["01", "04", "07", "10"]
for month in months:
    path_to_json = f'data/ais-hub/2020_{month}'

    json_pattern = os.path.join(path_to_json,'*.json')
    file_list = glob.glob(json_pattern)

    dfs = [] # an empty list to store the data frames
    for file in tqdm(file_list):
        data = pd.read_json(file, lines=True) # read data frame from json file
        data['absolute-time'] = pd.json_normalize(data['meta-data'])['absolute-time']
        dfs.append(data) # append the data frame to the list

    temp = pd.concat(dfs, ignore_index=True) # concatenate all the data frames in the list.

    df = temp[['longitude', 'latitude', 'absolute-time', 'source-mmsi', 'speed-over-ground',  'course-over-ground', \
            'true-heading', 'type' ]]
    df = df.dropna()
    df = df.rename(columns={'longitude': 'lon', 'latitude': 'lat', 'absolute-time': 'timestamp', 'source-mmsi': 'mmsi',\
        'speed-over-ground': 'sog', 'course-over-ground': 'cog',\
        'true-heading': 'heading'})

    df = df[df.sog>3]
    df = df[(df.lon<180) | (df.lat<90)]

    df = df[df.sog<=30]

    # convert to GeoDataFrame
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(x=df.lon, y=df.lat), crs="WGS84"
    )
    gdf['time'] = pd.to_datetime(gdf['timestamp'], unit='ms')
    gdf = gdf.set_index('time')

    # Specify minimum length for a trajectory (in meters)
    minimum_length = 1500 
    collection = mpd.TrajectoryCollection(gdf, 'mmsi', min_length=minimum_length)

    collection.add_speed(overwrite=True)
    collection.add_direction(overwrite=True)

    # Remove Trajectories that have too long gaps in consecutive AIS signals
    trips = mpd.ObservationGapSplitter(collection).split(gap=timedelta(minutes=5))
    # Remove all anchoring chips with a tolerance of 15 diameter (for example the Tugs laying in the "Schlepperhafen")
    trips = mpd.StopSplitter(trips).split(max_diameter=15, min_duration=timedelta(minutes=3), min_length=1500)
    # Outlier detecting and cleaning (Outlier (interquantile range - iqr) based cleaner.)
    # From moving pandas: "Note: Setting alpha=3 is widely used."
    trips = mpd.OutlierCleaner(trips).clean({'speed': 3})

    linear_out = resample_and_interpolate(trips, resample_interval='10S', interpolate_method='linear')
    linear_out

    linear_out.to_csv(f'data/processed/aishub_linear_artificial_{month}.csv', index=False)


