from matplotlib import projections
import numpy as np
import pandas as pd
import geopy
import time
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import pyproj
from geopy.distance import distance
from gym import core, spaces
from statistics import mean, median, stdev
from matplotlib.artist import Artist
import datetime

KNOTS_TO_KMH = 1.852
MS_TO_KNOTS = 1.94384


class AISenv(core.Env):
    def __init__(
        self,
        dataset="data/usable/aishub_linear_big_ships_2020_wind_tides_lengths.csv",
        time_interval=10,
    ):
        # Trajectory ID column 'traj_id'
        print("loading in ais data...")
        self.df = pd.read_csv(dataset, dtype={'speed': np.float32, 'cog': np.float32, 'lat': np.float32, 'lon': np.float32, 'direction': np.float32})
        self.num_trajectories = self.df.groupby("traj_id").count().shape[0]
        self.trajectory_list = list(self.df.groupby("traj_id"))
        random.shuffle(self.trajectory_list)
        self.time_interval_secs = time_interval
        print(self.num_trajectories)
        #################################################
        ##            DEFINE CONSTANTS                 ##
        #################################################
        
        # State boundaries
        self.MIN_LON, self.MAX_LON = self.df["lon"].min(), self.df["lon"].max()
        self.MIN_LAT, self.MAX_LAT = self.df["lat"].min(), self.df["lat"].max()
        self.MIN_COURSE, self.MAX_COURSE = -180, 180
        self.MIN_TEMPO, self.MAX_TEMPO = (
            self.df["speed"].min() * MS_TO_KNOTS,
            self.df["speed"].max() * MS_TO_KNOTS,
        )
        self.MIN_CURRENT_HEADING, self.MAX_CURRENT_HEADING = (
            self.df["direction"].min(),
            self.df["direction"].max(),
        )
        self.MIN_CURRENT_SPEED, self.MAX_CURRENT_SPEED = (
            self.df["speed"].min(),
            self.df["speed"].max(),
        )
        self.MIN_LENGTH, self.MAX_LENGTH = (
            self.df["length"].min(),
           self.df["length"].max(),
       )
        self.MIN_WIDTH, self.MAX_WIDTH = (
            self.df["width"].min(),
            self.df["width"].max(),
       )
        self.MIN_LEVEL, self.MAX_LEVEL = (
            self.df["tide_level"].min(),
            self.df["tide_level"].max(),
        )
        self.MIN_WINDFORCE, self.MAX_WINDFORCE = (
            self.df["wind_force"].min(),
            self.df["wind_force"].max(),
        )
        self.MIN_WINDDIRECTION, self.MAX_WINDDIRECTION = (
            self.df["wind_direction"].min(),
            self.df["wind_direction"].max(),
        )
        self.MIN_ANGLE_TO_DESTINATION, self.MAX_ANGLE_TO_DESTINATION = -180, 180
        _, max_dist = self._calculate_angle_distance(
            [self.MIN_LON, self.MIN_LAT], [self.MAX_LON, self.MAX_LAT]
        )
        self.MIN_DIST_TO_DESTINATION, self.MAX_DIST_TO_DESTINATION = 0, max_dist
        

        self.DLON = self.MAX_LON - self.MIN_LON
        self.DLAT = self.MAX_LAT - self.MIN_LAT
        self.DCOURSE = self.MAX_COURSE - self.MIN_COURSE
        self.DTEMPO = self.MAX_TEMPO - self.MIN_TEMPO
        self.DHEADING = self.MAX_CURRENT_HEADING - self.MIN_CURRENT_HEADING
        self.DSPEED = self.MAX_CURRENT_SPEED - self.MIN_CURRENT_SPEED
        self.DLENGTH = self.MAX_LENGTH - self.MIN_LENGTH
        self.DWIDTH = self.MAX_WIDTH - self.MIN_WIDTH
        self.DLEVEL = self.MAX_LEVEL - self.MIN_LEVEL
        self.DWINDFORCE = self.MAX_WINDFORCE - self.MIN_WINDFORCE
        self.DWINDDIRECTION = self.MAX_WINDDIRECTION - self.MIN_WINDDIRECTION
        self.DANGLE = self.MAX_ANGLE_TO_DESTINATION - self.MIN_ANGLE_TO_DESTINATION
        self.DDIST = self.MAX_DIST_TO_DESTINATION - self.MIN_DIST_TO_DESTINATION
        
        # Observations: lat, lon, heading, speed, length of vessel, width of vessel, tide_level, angle, distance to destination
        low = np.array(
            [
                self.MIN_LAT,
                self.MIN_LON,
                self.MIN_CURRENT_HEADING,
                self.MIN_CURRENT_SPEED,
               # self.MIN_LENGTH,
               # self.MIN_WIDTH,
                self.MIN_LEVEL,
                self.MIN_WINDFORCE,
                self.MIN_WINDDIRECTION,
                self.MIN_ANGLE_TO_DESTINATION,
                self.MIN_DIST_TO_DESTINATION,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.MAX_LAT,
                self.MAX_LON,
                self.MAX_CURRENT_HEADING,
                self.MAX_CURRENT_SPEED,
         #       self.MAX_LENGTH,
          #      self.MAX_WIDTH,
                self.MAX_LEVEL,
                self.MAX_WINDFORCE,
                self.MAX_WINDDIRECTION,
                self.MAX_ANGLE_TO_DESTINATION,
                self.MAX_DIST_TO_DESTINATION,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Actions: course, tempo, next position heading, next position speed
        low = np.array(
            [
                self.MIN_COURSE,
                self.MIN_TEMPO,
                self.MIN_CURRENT_HEADING,
                self.MIN_CURRENT_SPEED,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.MAX_COURSE,
                self.MAX_TEMPO,
                self.MAX_CURRENT_HEADING,
                self.MAX_CURRENT_SPEED,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=low, high=high)
        
        # Custom variables
        self.step_counter = 0
        self.scale = np.array(
            [
                1 / self.DLAT,
                1 / self.DLON,
                1 / self.DHEADING,
                1 / self.DSPEED,
           #     1 / self.DLENGTH,
            #    1 / self.DWIDTH,
                1 / self.DLEVEL,
                1 / self.DWINDFORCE,
                1 / self.DWINDDIRECTION,
                1 / self.DANGLE,
                1 / self.DDIST,
            ]
        )
        self.shift = np.array(
            [
                self.MIN_LAT,
                self.MIN_LON,
                self.MIN_CURRENT_HEADING,
                self.MIN_CURRENT_SPEED,
           #     self.MIN_LENGTH,
          #      self.MIN_WIDTH,
                self.MIN_LEVEL,
                self.MIN_WINDFORCE,
                self.MIN_WINDDIRECTION,
                self.MIN_ANGLE_TO_DESTINATION,
                self.MIN_DIST_TO_DESTINATION,
            ]
        )

        self.scale_action = np.array(
            [1 / self.DCOURSE, 1 / self.DTEMPO, 1 / self.DHEADING, 1 / self.DSPEED]
        )
        self.shift_action = np.array(
            [
                self.MIN_COURSE,
                self.MIN_TEMPO,
                self.MIN_CURRENT_HEADING,
                self.MIN_CURRENT_SPEED,
            ]
        )

        self.training = True
        self.trajectory_index = -1
        self.figure = None
        # curve the agent has to follow
        self.true_traj = None
        # curve that the agent took
        self.agent_traj = None
        self.time_multipler = 1
        print(self.MIN_LON, self.MAX_LON, self.MIN_LAT, self.MAX_LAT)

    def get_trajectory_count(self):
        return len(self.trajectory_list)

    def set_trajectory_index(self, index):
        self.trajectory_index = index

    def __getitem__(self, i):
        values = self.episode_df.iloc[i, :].values
        angle, dist = self._calculate_angle_distance(values[:2], self.final_pos)
        return np.append(values, [[angle], [dist]])

    def reset(self):
        self.step_counter = 0
        self.trajectory_index = self.trajectory_index + 1
        if self.trajectory_index >= self.num_trajectories:
            random.shuffle(self.trajectory_list)
            self.trajectory_index = 0
        self.episode_df = self.trajectory_list[self.trajectory_index][1]
        # , "length", "width" "tide_level"
        self.episode_df = self.episode_df[["lat", "lon", "direction", "speed", "tide_level", "wind_force", "wind_direction"]]
        self.length_episode = self.episode_df.shape[0]
        self.final_pos = self.episode_df.iloc[-1, :].values[:2]
        self.state = self[self.step_counter]
        self.true_traj = np.expand_dims(self.state[:2], 0)
        self.agent_traj = np.expand_dims(self.state[:2], 0)
        return self.scale * (self.state - self.shift)

    def _calculate_course_tempo(self, prev_state, next_state):
        geodesic = pyproj.Geod(ellps="WGS84")
        fwd_azimuth, back_azimuth, distance = geodesic.inv(
            prev_state[1], prev_state[0], next_state[1], next_state[0]
        )
        course = fwd_azimuth
        # distance is in meters; speed is in meters per second
        tempo = distance / (self.time_interval_secs * self.time_multipler)
        return course, tempo

    def _calculate_angle_distance(self, agent, destination):
        geodesic = pyproj.Geod(ellps="WGS84")
        fwd_azimuth, back_azimuth, distance = geodesic.inv(
            agent[1], agent[0], destination[1], destination[0]
        )
        return fwd_azimuth, distance

    def step_expert(self):
        last_obs = self.state
        self.step_counter = np.clip(
            self.step_counter + self.time_multipler, 0, self.length_episode - 1
        )
        self.state = self[self.step_counter]
        next_obs = self.state
        course, tempo = self._calculate_course_tempo(last_obs, next_obs)

        self.true_traj = np.concatenate(
            (self.true_traj, np.array([[self.state[0], self.state[1]]])), axis=0
        )

        action = self.scale_action * (
            np.array([course, tempo, next_obs[2], next_obs[3]]) - self.shift_action
        )
        last_obs = self.scale * (last_obs - self.shift)
        self.next_obs = self.scale * (self.state - self.shift)
        done = bool(self.step_counter >= self.length_episode - 1)

        return last_obs, action, {}, done

    def step(self, action):
        # Read current agent state and agent's action
        lat_agent, lon_agent = self.agent_traj[-1][0:2]
        # print(f'lat_a: {lat_agent} lon_a: {lon_agent}')
        course, tempo, heading, speed = map(lambda x: np.clip(x, 0, 1), action)
        # The agent's outputs need to be tranformed back to original scale
        course = self.MIN_COURSE + self.DCOURSE * course
        tempo = self.MIN_TEMPO + self.DTEMPO * tempo
        heading = self.MIN_CURRENT_HEADING + self.DHEADING * heading
        speed = self.MIN_CURRENT_SPEED + self.DSPEED * speed
        # artificial speed is in meters per second
        d = distance(meters=tempo * self.time_interval_secs * self.time_multipler)
        # Agent's suggestion of state update
        # d = distance(kilometers = sog_a * KNOTS_TO_KMH * self.time_interval_secs * self.time_multipler / 3600)
        lat_pred, lon_pred = d.destination(
            point=geopy.Point(latitude=lat_agent, longitude=lon_agent), bearing=course
        )[:2]
        # Ensure that predictions are within bounds
        lon_pred = np.clip(lon_pred, self.MIN_LON, self.MAX_LON)
        lat_pred = np.clip(lat_pred, self.MIN_LAT, self.MAX_LAT)
        # Compare with observation at next step
        self.step_counter = np.clip(
            self.step_counter + self.time_multipler, 0, self.length_episode - 1
        )
        self.state = self[self.step_counter]
        # print(f'{self.state} {cog_a} {sog_a}')
        lat_true, lon_true = self.state[:2]
        # print(f'TRUE: {self.state[3]} PRED: {sog_a}')
        # Compute the error based on the path track error
        geo_dist_meters = geopy.distance.distance(
            (lat_pred, lon_pred), (lat_true, lon_true)
        ).meters
        
        # rectified reward function based on distance between agent and GT position, alpha=8000
        reward = max(1 - (geo_dist_meters / 8000), 0)
        # Record predictions and observations of vessel location
        self.agent_traj = np.concatenate(
            (self.agent_traj, np.array([[lat_pred, lon_pred]])), axis=0
        )
        self.true_traj = np.concatenate(
            (self.true_traj, np.array([[lat_true, lon_true]])), axis=0
        )
        angle, dist = self._calculate_angle_distance(
            [lat_pred, lon_pred], self.final_pos
        )
        # is the end of trajectory reached?
        done = self.step_counter >= self.length_episode - 1 or dist < 300

        # The agent's networks need normalized observations

        # print([lat_pred, lon_pred, heading, speed, angle, dist])
        observation = self.scale * (
            # , self.state[4], self.state[5]
            np.array([lat_pred, lon_pred, heading, speed, self.state[4], self.state[5], self.state[6], angle, dist]) - self.shift
        )

        return observation, reward, done, {"distance_in_meters": geo_dist_meters}

    def render(self, mode="human", svg=None, agent_traj=None, true_traj=None):
        
        if agent_traj is not None and true_traj is not None:
            t = true_traj
            a = agent_traj
        else:
            t = self.true_traj
            a = self.agent_traj
            
        if self.figure is None:
            plt.ion()
            self.figure = 1
            print("read")
            self.img = plt.imread("deeprl/bg.png")
            self.figure, self.ax = plt.subplots()
            self.ax.imshow(self.img, extent=[self.MIN_LON, self.MAX_LON, self.MIN_LAT, self.MAX_LAT], aspect="equal")
            #self.figure = 1

        if hasattr(self, 'aLine'):
            self.ax.lines.pop(0)
            self.ax.lines.pop(0)
            self.ax.lines.pop(0)
            self.ax.lines.pop(0)
            Artist.remove(self.frame)
            Artist.remove(self.frame2)
            time.sleep(0.0005)
        
        self.tLine = self.ax.plot(
            t[:, 1],
            t[:, 0],
            zorder=2,
            linewidth=3,
            color="black",
        )
        self.aLine = self.ax.plot(
            a[:, 1],
            a[:, 0],
            zorder=3,
            alpha=0.6,
            linewidth=3,
            color="red",
        )
        self.ax.plot( 
            a[-1:, 1],
            a[-1:, 0],
            marker="o",
            zorder=3,
            alpha=0.6,
            markersize=6,
            markeredgecolor="red",
            markerfacecolor="red"
            )
        
        self.ax.plot( 
            t[-1:, 1],
            t[-1:, 0], 
            marker="o",
            zorder=2,
            markersize=6,
            markeredgecolor="black",
            markerfacecolor="black"
            )
        self.ax.legend(
            handles=[
                mpatches.Patch(color="black", label="Ground-Truth"),
                mpatches.Patch(color="red", label="Agent"),
            ]
        )
        self.frame = plt.text(1.10, 0.95, f'Time elasped: \n{str(datetime.timedelta(seconds= len(t) *10))}', ha='center', va='center', transform=self.ax.transAxes)
        self.frame2 = plt.text(1.11, 0.85, f'Distance: {int(geopy.distance.distance((t[-1:, 1], t[-1:, 0]), (a[-1:, 1], a[-1:, 0])).meters)} m', ha='center', va='center', transform=self.ax.transAxes)

        plt.xlim([self.MIN_LON, self.MAX_LON])
        plt.ylim([self.MIN_LAT, self.MAX_LAT])
        plt.draw()
        plt.pause(0.0005)
        time.sleep(0.0002)
        # Save output as .svg
        if svg is not None:
            self.ax.pause(0.3)
            self.ax.savefig(f"{svg}.svg", format="svg")


