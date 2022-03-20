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
from scipy.stats import norm
import geopandas
#import contextily as ctx

# State boundaries
MIN_LON, MAX_LON = 8.372, 8.58776333
MIN_LAT, MAX_LAT = 53.4570, 53.71997
MIN_COG, MAX_COG = 0., 359.9
MIN_SOG, MAX_SOG = 1e-7, 29.9
# Define inverse scales
DLON = MAX_LON - MIN_LON
DLAT = MAX_LAT - MIN_LAT
DCOG = MAX_COG - MIN_COG
DSOG = MAX_SOG - MIN_SOG
KNOTS_TO_KMH = 1.852

def update_lat(cog, sog, dt):
    return (dt / 60) * cos(rad(cog)) * sog

def update_lon(lat, cog, sog, dt):
    return (dt / 60) * sin(rad(cog)) * sog / cos(rad(lat))

def geo_distance(p1, p2):
    """ Distance between points p1 and p2 in Km"""
    lat1, lon1 = p1
    lat2, lon2 = p2
    R = 6378.137 
    hx = sin(0.5*rad(lat2-lat1))**2
    hy = sin(0.5*rad(lon2-lon1))**2
    h = hx + cos(rad(lat1))*cos(rad(lat2))* hy
    return 2*R* np.arcsin(np.sqrt(h))


class AISenv(core.Env):

    def __init__(self, dataset='data/processed/aishub_linear_artificial.csv', time_interval=5):
        # Trajectory ID column 'traj_id'
        print("loading in ais data...")
        self.df = pd.read_csv(dataset)
        self.num_trajectories = self.df.groupby('traj_id').count().shape[0]
        self.trajectory_list = list(self.df.groupby('traj_id'))
        self.time_interval_secs = time_interval
        
        # Observations: lon, lat, cog, sog, dt
        low = np.array([MIN_LON, MIN_LAT, MIN_COG, MIN_SOG], dtype=np.float32)
        high = np.array([MAX_LON, MAX_LAT,  MAX_COG, MAX_SOG], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Actions: cog, sog
        low = np.array([MIN_COG, MIN_SOG], dtype=np.float32)
        high = np.array([MAX_COG, MAX_SOG], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high)
        # Custom variables
        self.step_counter = 0
        self.scale = np.array([1/DLON, 1/DLAT, 1/DCOG, 1/DSOG])
        self.shift = np.array([MIN_LON, MIN_LAT, MIN_COG, MIN_SOG])

        self.scale_action = np.array([1/DCOG, 1/DSOG])
        self.shift_action = np.array([MIN_COG, MIN_SOG])

        self.training = True
        self.trajectory_index = 0
        self.figure = None
        # curve the agent has to follow
        self.true_traj = None
        # curve that the agent took
        self.agent_traj = None
        self.time_multipler = 5
      
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
        self.episode_df = self.episode_df[['lon', 'lat', 'direction', 'speed']]
        self.length_episode = self.episode_df.shape[0] 
        self.state = self[0]
        #print(f'state {self.state}')
        self.true_traj = np.expand_dims(self.state[:2],0)
        self.agent_traj = np.expand_dims(self.state[:2],0)
        #return self.scale * (self.state - self.shift)
        return self.state
    
    def step_expert(self):
        last_obs = self.scale * (self.state - self.shift)
        last_obs = self.state
        self.step_counter = np.clip(self.step_counter + self.time_multipler, 0, self.length_episode - 1)
        self.state = self[self.step_counter]
        next_obs = self.scale * (self.state - self.shift)
        next_obs = self.state
        done = bool(self.step_counter >= self.length_episode-1)
        self.true_traj = np.concatenate((self.true_traj, np.array([[self.state[0], self.state[1]]])), axis=0)
        self.next_obs = next_obs
        #print(last_obs[0], last_obs[1], last_obs[2], next_obs[2], next_obs[3])
        return last_obs, np.array([next_obs[2], next_obs[3]]), {}, done

    def step(self, action):
        # Read current agent state and agent's action
        lon_agent = self.agent_traj[int(self.step_counter / self.time_multipler)][0]
        lat_agent = self.agent_traj[int(self.step_counter / self.time_multipler)][1]
        #print(f'lat_a: {lat_agent} lon_a: {lon_agent}')
        #cog_a, sog_a = map(lambda x: np.clip(x, 0, 1), action)
        # The agent's outputs need to be tranformed back to original scale
        #cog_a = MIN_COG + DCOG * cog_a 
        #sog_a = MIN_SOG + DSOG * sog_a
        cog_a = action[0]
        sog_a = action[1]
        # Agent's suggestion of state update
        #d = distance(kilometers = sog_a * KNOTS_TO_KMH * self.time_interval_secs * self.time_multipler / 3600)
        d = distance(meters=sog_a * self.time_interval_secs * self.time_multipler)
        p = d.destination(point=geopy.Point(latitude=lat_agent, longitude=lon_agent), bearing=cog_a)
        lat_pred = p[0]
        lon_pred = p[1]
        # Ensure that predictions are within bounds
        lon_pred = np.clip(lon_pred, MIN_LON, MAX_LON)
        lat_pred = np.clip(lat_pred, MIN_LAT, MAX_LAT)

        # Compare with observation at next step
        self.step_counter = np.clip(self.step_counter + self.time_multipler, 0, self.length_episode - 1)
        self.state = self[self.step_counter]
        lon_true, lat_true = self.state[:2]
        #print(f'TRUE: {self.state[3]} PRED: {sog_a}')
        # Compute the error based on the path track error
        geo_dist_meters = geopy.distance.distance((lat_pred, lon_pred), (lat_true, lon_true)).meters
        reward = norm.pdf(geo_dist_meters, 0, 50) * 125.3315
        if reward < 0.001:
            reward = 0
        #print(geo_dist)
        # Record predictions and observations of vessel location
        self.agent_traj = np.concatenate((self.agent_traj, np.array([[lon_pred, lat_pred]])), axis=0)
        self.true_traj = np.concatenate((self.true_traj, np.array([[lon_true, lat_true]])), axis=0)

        # is the end of trajectory reached?
        done = bool(self.step_counter >= self.length_episode-1)
        
        # The agent's networks need normalized observations 
        observation = self.scale * (self.state - self.shift)
        #observation = self.scale * (np.array([lon_pred, lat_pred, cog_a, sog_a]) - self.shift)
        observation = np.array([lon_pred, lat_pred, cog_a, sog_a])
        return observation, reward, done, {'distance_in_meters': geo_dist_meters}
    
    def render(self, mode):
        #print(self.pos_pred[:,0])
        if self.figure == None:
            plt.ion()
            self.figure = 1
            west, south, east, north = (
            8.45,
            53.45,
            8.6,
            53.8
            )
          #  self.ghent_img, self.ghent_ext = ctx.bounds2img(west,
           #                     south,
            #                    east,
             #                   north,
              #                  ll=True,
               #                 source=ctx.providers.OpenStreetMap.Mapnik
                #            )
            #plt.figure(figsize=(15,10))
           
            # pos_history = np.concatenate((self.pos_pred, self.pos_true), axis=0)
            # hist_len = pos_history.shape[0] // 2
            # df_pos = pd.DataFrame(pos_history, columns=['lon', 'lat'])
            # df_pos['entity'] = ['prediction'] * hist_len + ['observation'] * hist_len

            # fig = px.scatter_mapbox(df_pos, lon="lon", lat="lat", color="entity",
            #                 zoom=11, height=800, 
            #                 center={'lat': 53.53, 'lon':8.56})
            # fig.update_layout(mapbox_style="open-street-map")
            # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            # self.figure = go.FigureWidget(fig)
        #else:
            # self.figure.data[0].lat = self.pos_pred[:,0]
            # self.figure.data[0].lon = self.pos_pred[:,1]
            # self.figure.data[1].lat = self.pos_true[:,1]
            # self.figure.data[1].lon = self.pos_true[:,0]
       # self.figure.plotly_update()
       # plt.scatter(self.agent_traj[:,0], self.agent_traj[:,1])
        # plt.plot(self.agent_traj[:,0], self.agent_traj[:,1])
        plt.plot(self.true_traj[:,0], self.true_traj[:,1], zorder=3, color="blue")
        plt.plot(self.agent_traj[:,0], self.agent_traj[:,1], zorder=2, color="orange")
        plt.ylim([53.50, 53.63])
        plt.xlim([8.48, 8.6])
        #plt.ylim([MIN_LAT, MAX_LAT])
       # plt.xlim([MIN_LON,MAX_LON])
        # # plt.draw()
        # plt.pause(0.0001)
        # plt.clf()
        # time.sleep(0.01)
        
        #f, ax = plt.subplots(1, figsize=(9, 9))
      #  df = pd.DataFrame({'lat': self.agent_traj[:,0], 'lon': self.agent_traj[:,1]})
      #  gdf = geopandas.GeoDataFrame(
      #      df, geometry=geopandas.points_from_xy(df.lon ,df.lat))

        #ax.set_xlim(8.4,8.6)
       # ax.set_ylim(53.4, 53.9)
 
      #  ax = gdf.plot(alpha=0.80, color='#d66058', figsize=(10,10))
       # ctx.add_basemap(ax, crs="epsg:4326", source=ctx.providers.OpenStreetMap.Mapnik, interpolation="sinc")
        #plt.plot(self.agent_traj[:,0], self.agent_traj[:,1])
      #  plt.imshow(self.ghent_img, extent=self.ghent_ext, zorder=1)
        plt.draw()  
        plt.pause(0.0001)
        plt.clf()
        time.sleep(0.08)

register(
    id="ais-v0",
    entry_point="deeprl.envs.ais_env:AISenv",
)           
