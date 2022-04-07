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
from gym.envs.registration import register


KNOTS_TO_KMH = 1.852
MS_TO_KNOTS = 1.94384


# Optional, to do resampling on the fly without explicitly generating a new dataset. This is not recommended!
def resample_and_interpolate(
    trips, resample_interval="5S", interpolate_method="linear"
):
    out = pd.DataFrame()
    grp = trips.groupby(["traj_id"])
    for traj_id in list(trips.groupby(["traj_id"]).groups.keys()):
        tmp = grp.get_group(traj_id)
        tmp["time"] = pd.to_datetime(tmp["timestamp"])
        tmp = tmp.set_index("time")
        tmp.set_index("timestamp")
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
    def __init__(
        self,
        dataset="data/processed/aishub_linear_artificial_big_ships.csv",
        time_interval=10,
    ):
        # Trajectory ID column 'traj_id'
        print("loading in ais data...")
        self.df = pd.read_csv(dataset)
        # self.df = resample_and_interpolate(self.df, "1S", "linear")
        self.num_trajectories = self.df.groupby("traj_id").count().shape[0]
        # tmp = list(self.df.groupby('traj_id'))
        self.trajectory_list = list(self.df.groupby("traj_id"))
        random.shuffle(self.trajectory_list)
        self.time_interval_secs = time_interval
        # define constants
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
        self.DLON = self.MAX_LON - self.MIN_LON
        self.DLAT = self.MAX_LAT - self.MIN_LAT
        self.DCOURSE = self.MAX_COURSE - self.MIN_COURSE
        self.DTEMPO = self.MAX_TEMPO - self.MIN_TEMPO
        self.DHEADING = self.MAX_CURRENT_HEADING - self.MIN_CURRENT_HEADING
        self.DSPEED = self.MAX_CURRENT_SPEED - self.MIN_CURRENT_SPEED

        # Observations: lat, lon, cog, sog
        low = np.array(
            [
                self.MIN_LAT,
                self.MIN_LON,
                self.MIN_CURRENT_HEADING,
                self.MIN_CURRENT_SPEED,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.MAX_LAT,
                self.MAX_LON,
                self.MAX_CURRENT_HEADING,
                self.MAX_CURRENT_SPEED,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Actions: cog, sog
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
            [1 / self.DLAT, 1 / self.DLON, 1 / self.DHEADING, 1 / self.DSPEED]
        )
        self.shift = np.array(
            [
                self.MIN_LAT,
                self.MIN_LON,
                self.MIN_CURRENT_HEADING,
                self.MIN_CURRENT_SPEED,
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

    def get_trajectory_count(self):
        return len(self.trajectory_list)

    def set_trajectory_index(self, index):
        self.trajectory_index = index

    def __getitem__(self, i):
        return self.episode_df.iloc[i, :].values

    def reset(self):
        self.step_counter = 0
        self.trajectory_index = self.trajectory_index + 1
        if self.trajectory_index >= self.num_trajectories:
            random.shuffle(self.trajectory_list)
            self.trajectory_index = 0
        self.episode_df = self.trajectory_list[self.trajectory_index][1]
        self.episode_df = self.episode_df[["lat", "lon", "direction", "speed"]]
        self.length_episode = self.episode_df.shape[0]
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
        # is the end of trajectory reached?
        done = bool(self.step_counter >= self.length_episode - 1)
        # The agent's networks need normalized observations
        # observation = self.scale * (self.state - self.shift)
        observation = self.scale * (
            np.array([lat_pred, lon_pred, heading, speed]) - self.shift
        )
        # observation = np.array([lat_pred, lon_pred, heading, speed])
        return observation, reward, done, {"distance_in_meters": geo_dist_meters}

    def render(self, mode="human", svg=None):
        if self.figure is None:
            plt.ion()
            self.figure = 1
        plt.clf()
        plt.plot(
            self.true_traj[:, 1],
            self.true_traj[:, 0],
            zorder=2,
            linewidth=3,
            color="black",
        )
        plt.plot(
            self.agent_traj[:, 1],
            self.agent_traj[:, 0],
            zorder=3,
            alpha=0.6,
            linewidth=3,
            color="red",
        )
        plt.legend(
            handles=[
                mpatches.Patch(color="black", label="Ground-Truth"),
                mpatches.Patch(color="red", label="Agent"),
            ]
        )
        plt.xlim([self.MIN_LON, self.MAX_LON])
        plt.ylim([self.MIN_LAT, self.MAX_LAT])
        plt.draw()
        plt.pause(0.0001)
        time.sleep(0.02)
        # Save output as .svg
        if svg is not None:
            plt.pause(0.3)
            plt.savefig(f"{svg}.svg", format="svg")


register(
    id="ais-v0",
    entry_point="deeprl.envs.ais_env:AISenv",
)
