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
import sys
# State boundaries
MIN_LON, MAX_LON = 8.4361, 8.5927
MIN_LAT, MAX_LAT = 53.4570, 53.6353
MIN_COG, MAX_COG = 0., 359.9
MIN_SOG, MAX_SOG = 3., 29.9
# Define inverse scales
DLON = MAX_LON - MIN_LON
DLAT = MAX_LAT - MIN_LAT
DCOG = MAX_COG - MIN_COG
DSOG = MAX_SOG - MIN_SOG

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

    def __init__(self, dataset='deeprl/envs/trajectories_linear_interpolate.csv', time_interval=5):
        # Trajectory ID column 'traj_id'
        self.df = pd.read_csv(dataset)
        self.num_trajectories = self.df.groupby('traj_id').count().shape[0]
        self.time_interval_secs = time_interval
        
        # Observations: lon, lat, cog, sog, dt
        low = np.array([MIN_LON, MIN_LAT, MIN_COG, MIN_SOG], dtype=np.float32)
        high = np.array([MAX_LON, MAX_LAT, MAX_COG, MAX_SOG], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Actions: cog, sog
        low = np.array([MIN_COG, MIN_SOG], dtype=np.float32)
        high = np.array([MAX_COG, MAX_SOG], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high)
        # Custom variables
        self.step_counter = 0
        self.scale = np.array([1/DLON, 1/DLAT, 1/DCOG, 1/DSOG])
        self.shift = np.array([MIN_LON, MIN_LAT, MIN_COG, MIN_SOG])
        self.training = True
        self.trajectory_index = 0
        self.figure = None
        # curve the agent has to follow
        self.true_traj = None
        # curve that the agent took
        self.agent_traj = None
        self.time_multipler = 10
        
    def __getitem__(self, i):
        return self.episode_df.iloc[i,:].values

    def reset(self):
        self.episode_df = list(self.df.groupby('traj_id'))[self.trajectory_index][1]
        self.episode_df = self.episode_df[['lat', 'lon', 'cog', 'sog']]
        self.length_episode = self.episode_df.shape[0] 
        self.state = self[0]
        print(f'state {self.state}')
        self.true_traj = np.expand_dims(self.state[:2],0)
        self.agent_traj = np.expand_dims(self.state[:2],0)
        return self.scale * (self.state - self.shift)
    
    def step(self, action):
        if not self.the_end:
            # Read current agent state and agent's action
            lat_agent = self.agent_traj[int(self.step_counter / self.time_multipler)][0]
            lon_agent = self.agent_traj[int(self.step_counter / self.time_multipler)][1]
            print(f'lat_a: {lat_agent} lon_a: {lon_agent}')
            cog_a, sog_a = map(lambda x: np.clip(x, 0, 1), action)
            # The agent's outputs need to be tranformed back to original scale
            sog_a = MIN_SOG + DSOG * sog_a
            cog_a = MIN_COG + DCOG * cog_a 
        
            # Agent's suggestion of state update
            d = distance(kilometers = sog_a * self.time_interval_secs * self.time_multipler / 3600)
            p = d.destination(point=geopy.Point(lat_agent, lon_agent), bearing=cog_a)
            lat_pred = p[0]
            lon_pred = p[1]
            # Ensure that predictions are within bounds
            lat_pred = np.clip(lat_pred, MIN_LAT, MAX_LAT)
            lon_pred = np.clip(lon_pred, MIN_LON, MAX_LON)
            cog_pred = np.clip(cog_a, MIN_COG, MAX_COG)

            # Compare with observation at next step
            self.step_counter = np.clip(self.step_counter + 1 * self.time_multipler, 0, self.length_episode - 1)
            self.state = self[self.step_counter]
            lat_true, lon_true = self.state[:2]
            print(f'KEK {lat_true} {lon_true}')
            # If the agent is self-looping, modify the next state accordingly
            self.state = self.state if self.training else np.array([lon_pred, lat_pred, cog_pred])
            # Compute the error committed by the agent's state suggestion
            geo_dist = geopy.distance.distance((lat_pred, lon_pred), (lat_true, lon_true)).meters
            reward = 0
            print(geo_dist)
            # Record predictions and observations of vessel location
            self.agent_traj = np.concatenate((self.agent_traj, np.array([[lat_pred, lon_pred]])), axis=0)
            self.true_traj = np.concatenate((self.true_traj, np.array([[lat_true, lon_true]])), axis=0)
        else: 
            reward = self.finish()
        # The agent's networks need normalized observations 
        observation = self.scale * (self.state - self.shift)
        return observation, reward, self.the_end, {}
    
    def render(self):
        #print(self.pos_pred[:,0])
        if self.figure == None:
            plt.ion()
            self.figure = 1
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
        #plt.scatter(self.agent_traj[:,0], self.agent_traj[:,1])
        plt.plot(self.agent_traj[:,0], self.agent_traj[:,1])
        plt.plot(self.true_traj[:,0], self.true_traj[:,1])
        plt.xlim([MIN_LAT, MAX_LAT])
        plt.ylim([MIN_LON, MAX_LON])
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        time.sleep(0.01)
    
    @property
    def the_end(self):
        return bool(self.step_counter >= self.length_episode-1)
    
    def finish(self, reset_trajectory_index=False):
        self.step_counter = 0
        self.episode = 0 if reset_trajectory_index else self.trajectory_index + 1
        self.episode = self.trajectory_index % self.num_trajectories 
        self.state = self.shift + self.reset() / self.scale
        return 0

def run_agent_env_loop(env, agent, random_process, 
                       num_episodes=None, render=True, self_loop=False, in_sample=False):

    num_episodes = num_episodes if num_episodes else env.num_episodes
    # Since agent.fit trains on nb_steps, the last training episode may not be finished yet
    try:
        if not env.the_end: _ = env.finish(reset_episodes=in_sample) # reset_episodes -> test in-sample
    except: # when running before calling agent.fit
        pass
    # Tag env as non-trainable: states not read from dataset now but from agent predictions
    env.training = False if self_loop else True
    # Reset random_process
    random_process.reset_states()
    for episode in range(num_episodes):
        print(f"Episode {episode}/{num_episodes}")
        observation = env.reset()
        for t, _ in enumerate(env):
            action = agent.forward(observation) + random_process.sample()
            observation, _, done, _ = env.step(action)
            if done:
                print(f"Episode finished after {t+1} steps")
                break
        if render: env.render()
        
ais = AISenv()
ais.reset()
for i in range(0,600):
    ais.step([0.4, 0.4])
    ais.render()
    
register(
    id="ais-v0",
    entry_point="deeprl.envs.ais_env:AISenv",
)           
