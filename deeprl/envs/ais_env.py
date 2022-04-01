from sqlite3 import Timestamp
from turtle import color
import numpy as np
from numpy import sin, cos, pi
from numpy import radians as rad
import pandas as pd
import plotly.express as px 
import geopy
from geopy.distance import geodesic, distance
import time
import plotly.graph_objects as go
from gym import core, spaces
from gym.envs.registration import register
import matplotlib.pyplot as plt
import random
import sys
from tqdm import tqdm

# State boundaries
MIN_LON, MAX_LON = 8.486, 8.5869
MIN_LAT, MAX_LAT = 53.497, 53.6172
# COG is the same as the artificial movingpandas's value for heading
MIN_COG, MAX_COG = 0., 359.9
MIN_SOG, MAX_SOG = 1e-7, 29.9
MIN_SPEED, MAX_SPEED = 1.5433, 15.433
# Define inverse scales
DLON = MAX_LON - MIN_LON
DLAT = MAX_LAT - MIN_LAT
DCOG = MAX_COG - MIN_COG
DSOG = MAX_SOG - MIN_SOG
DSPEED = MAX_SPEED - MIN_SPEED
KNOTS_TO_KMH = 1.852
def resample_and_interpolate(trips, resample_interval="5S", interpolate_method="linear"):
    out = pd.DataFrame()
    grp = trips.groupby(["traj_id"])
    for traj_id in list(trips.groupby(["traj_id"]).groups.keys()):
        tmp = grp.get_group(traj_id)
        tmp['time'] = pd.to_datetime(tmp['timestamp'])
        tmp = tmp.set_index('time')
        tmp.set_index('timestamp')
        tmp = tmp.resample(resample_interval, origin="end").mean()
        tmp["mmsi"] = tmp["mmsi"].interpolate(method="bfill")
        tmp["sog"] = tmp["sog"].interpolate(method=interpolate_method)
        tmp["cog"] = tmp["cog"].interpolate(method=interpolate_method)
        tmp["lat"] = tmp["lat"].interpolate(method=interpolate_method)
        tmp["lon"] = tmp["lon"].interpolate(method=interpolate_method)
        tmp["heading"] = tmp["heading"].interpolate(method=interpolate_method)
        tmp["speed"] = tmp["speed"].interpolate(method=interpolate_method)
        tmp["direction"] = tmp["direction"].interpolate(method=interpolate_method)
        tmp["traj_id"] = str(tmp["mmsi"][0]) + str(tmp.iloc[0].name)
        out = out.append(tmp, ignore_index=True)
        
    return out


class AISenv(core.Env):

    def __init__(self, dataset='data/processed/aishub_linear_artificial_tanker.csv', time_interval=10):
        # Trajectory ID column 'traj_id'
        print("loading in ais data...")
        self.df = pd.read_csv(dataset)
        self.df = resample_and_interpolate(self.df, "1S", "linear")
        self.num_trajectories = self.df.groupby('traj_id').count().shape[0]
        self.trajectory_list = list(self.df.groupby('traj_id'))
        random.shuffle(self.trajectory_list)
        self.time_interval_secs = time_interval
        global MIN_LON, MAX_LON
        global MIN_LAT, MAX_LAT
        global MIN_COG, MAX_COG
        global MIN_SOG, MAX_SOG
        MIN_LON, MAX_LON = self.df["lon"].min(), self.df["lon"].max()
        MIN_LAT, MAX_LAT = self.df["lat"].min(), self.df["lat"].max()
        MIN_COG, MAX_COG = self.df["direction"].min(), self.df["direction"].max()
        MIN_SOG, MAX_SOG = self.df["speed"].min(), self.df["speed"].max()
        global DLON, DLAT, DCOG, DSOG
        DLON = MAX_LON - MIN_LON
        DLAT = MAX_LAT - MIN_LAT
        DCOG = MAX_COG - MIN_COG
        DSOG = MAX_SOG - MIN_SOG

        # Observations: lat, lon, cog, sog
        low = np.array([MIN_LAT, MIN_LON, MIN_COG, MIN_SOG], dtype=np.float32)
        high = np.array([MAX_LAT, MAX_LON, MAX_COG, MAX_SOG], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Actions: cog, sog
        low = np.array([MIN_COG, MIN_SOG], dtype=np.float32)
        high = np.array([MAX_COG, MAX_SOG], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high)
        # Custom variables
        self.step_counter = 0
        self.scale = np.array([1/DLAT, 1/DLON, 1/DCOG, 1/DSOG])
        self.shift = np.array([MIN_LAT, MIN_LON, MIN_COG, MIN_SOG])

        self.training = True
        self.trajectory_index = 0
        self.figure = None
        # curve the agent has to follow
        self.true_traj = None
        # curve that the agent took
        self.agent_traj = None
        self.time_multipler = 1
      
    def get_trajectory_count(self):
        return len(self.trajectory_list)
    
    def set_trajectory_index(self, index):
        self.trajectory_index = index
        
    def __getitem__(self, i):
        return self.episode_df.iloc[i,:].values

    def _scale_action(self, action):
        return self.scale_action * (action - self.shift_action)


    def reset(self):
        self.step_counter = 0             
        self.trajectory_index = self.trajectory_index + 1
        if self.trajectory_index >= self.num_trajectories:
            random.shuffle(self.trajectory_list)
            self.trajectory_index = 0
        self.episode_df = self.trajectory_list[self.trajectory_index][1]
        self.episode_df = self.episode_df[['lat', 'lon', 'direction', 'speed']]
        self.length_episode = self.episode_df.shape[0] 
        self.state = self[0]
        #print(f'state {self.state}')
        self.true_traj = np.expand_dims(self.state[:2],0)
        self.agent_traj = np.expand_dims(self.state[:2],0)
        return self.scale * (self.state - self.shift)
    
    def step_expert(self):
        last_obs = self.scale * (self.state - self.shift)
        k = self.state
        self.step_counter = np.clip(self.step_counter + self.time_multipler, 0, self.length_episode - 1)
        self.state = self[self.step_counter]
        #print(k[0], k[1], k[2], k[3], self.state[2], self.state[3])
        next_obs = self.scale * (self.state - self.shift)

        done = bool(self.step_counter >= self.length_episode-1)
        self.true_traj = np.concatenate((self.true_traj, np.array([[self.state[0], self.state[1]]])), axis=0)
        self.next_obs = next_obs
        
        return last_obs, np.array([next_obs[2], next_obs[3]]), {}, done

    def step(self, action):
        # Read current agent state and agent's action
        lat_agent = self.agent_traj[int(self.step_counter / self.time_multipler)][0]
        lon_agent = self.agent_traj[int(self.step_counter / self.time_multipler)][1]
        #print(f'lat_a: {lat_agent} lon_a: {lon_agent}')
        cog_a, sog_a = map(lambda x: np.clip(x, 0, 1), action)
        # The agent's outputs need to be tranformed back to original scale
        cog_a = MIN_COG + DCOG * cog_a 
        sog_a = MIN_SOG + DSOG * sog_a
        # artificial speed is in meters per second
        d = distance(meters = sog_a * self.time_interval_secs * self.time_multipler)
        # Agent's suggestion of state update
        #d = distance(kilometers = sog_a * KNOTS_TO_KMH * self.time_interval_secs * self.time_multipler / 3600)
        #d = distance(meters=sog_a * self.time_interval_secs * self.time_multipler)
        p = d.destination(point=geopy.Point(latitude=lat_agent, longitude=lon_agent), bearing=cog_a)
        lat_pred = p[0]
        lon_pred = p[1]
        # Ensure that predictions are within bounds
        lon_pred = np.clip(lon_pred, MIN_LON, MAX_LON)
        lat_pred = np.clip(lat_pred, MIN_LAT, MAX_LAT)
        # Compare with observation at next step
        self.step_counter = np.clip(self.step_counter + self.time_multipler, 0, self.length_episode - 1)
        self.state = self[self.step_counter]
        #print(f'{self.state} {cog_a} {sog_a}')
        lat_true, lon_true = self.state[:2]
        #print(f'TRUE: {self.state[3]} PRED: {sog_a}')
        # Compute the error based on the path track error
        geo_dist_meters = geopy.distance.distance((lat_pred, lon_pred), (lat_true, lon_true)).meters
        # not used by behavioral cloning
        reward = 0
        # Record predictions and observations of vessel location
        self.agent_traj = np.concatenate((self.agent_traj, np.array([[lat_pred, lon_pred]])), axis=0)
        self.true_traj = np.concatenate((self.true_traj, np.array([[lat_true, lon_true]])), axis=0)
        # is the end of trajectory reached?
        done = bool(self.step_counter >= self.length_episode-1)
        # The agent's networks need normalized observations 
        #observation = self.scale * (self.state - self.shift)
        #observation = np.array([lat_pred, lon_pred, cog_a, sog_a])
        observation = self.scale * (np.array([lat_pred, lon_pred, cog_a, sog_a]) - self.shift)
        return observation, reward, done, {'distance_in_meters': geo_dist_meters}
    
    def render(self, mode):
        if self.figure == None:
            plt.ion()
            self.figure = 1
            
        plt.plot(self.true_traj[:,1], self.true_traj[:,0], zorder=2, color="blue")
        plt.plot(self.agent_traj[:,1], self.agent_traj[:,0], zorder=3, color="orange")
        plt.xlim([MIN_LON, MAX_LON])
        plt.ylim([MIN_LAT, MAX_LAT])
        plt.draw()  
        plt.pause(0.0001)
        plt.clf()
        time.sleep(0.08)

register(
    id="ais-v0",
    entry_point="deeprl.envs.ais_env:AISenv",
)           
