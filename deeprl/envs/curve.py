from typing import Tuple
import gym
from gym import spaces
import numpy as np
from tkinter import *
import tkinter as tk
import random
import threading
import math
import time
import deeprl.common.util as util
from scipy.stats import norm
from gym.envs.registration import register


class CurveEnv(gym.Env):
    """Custom Environment that follows gym interface

    actions: -1 to 1    (internally 0 to 360 degrees)

    state: [width, height] => [0-1, 0-1]
    """

    def __init__(self, animation_delay=0.1):
        super(CurveEnv, self).__init__()
        self._rng_seed = 12345
        self.animation_delay = animation_delay
        self.width = 700
        self.height = 500
        self.min_speed = 5
        self.max_speed = 40
        self.min_angle = -math.pi
        self.max_angle = math.pi

        # Action space is heading which is 0 to 360 degrees and speed from 0 to 40
        # action(angle, speed)
        # self.action_space = spaces.Box(low=-1, high=1,
        #                                 shape=(2,), dtype=np.float32)

        # obervation state (x, y, speed)
        #  self.observation_space = spaces.Box(low=-1, high=1,
        #                                  shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        self.trajectories = {
            "sinus-small": {
                "starting_pos": (20, 400),
                "func": lambda t: math.sin(t / 1.5) - math.pi / 6,
            },
            "sinus-big": {
                "starting_pos": (400, 20),
                "func": lambda t: math.sin(t/2) + math.pi * 0.6,
            },
            "sinus-ultra": {
                "starting_pos": (10, 10),
                "func": lambda t: math.sin(t / 4) + math.pi / 6,
            },
        }
        self.current_generator_curve = None
        self.current_generator_name = None

        self.true_traj = None
        self.agent_traj = None
        self.step_count = 0

        self.rng = np.random.default_rng(self._rng_seed)

        self.curr_alpha = 0
        self.curr_speed = 0
        self.curr_reward = 0

        self.master = None

    def _reset_starting_pos(self, starting_pos: Tuple[int, int]):
        x = starting_pos[0]
        y = starting_pos[1]
        self.true_traj = [(x, y), (x, y)]
        self.agent_traj = [(x, y), (x, y)]

    def _normalize_action(self, action):
        norm_angle = util.lmap(action[0], [self.min_angle, self.max_angle], [-1, 1])
        # norm_speed = util.lmap(action[1], [self.min_speed, self.max_speed], [-1, 1])

        # return np.array([norm_angle, norm_speed])
        return np.array([norm_angle])

    def _denormalize_action(self, action):
        original_angle = util.lmap(action[0], [-1, 1], [self.min_angle, self.max_angle])
        # original_speed = util.lmap(action[1], [-1, 1], [self.min_speed, self.max_speed])
        return np.array([original_angle], dtype=np.float32)

    def _normalize_state(self, state):
        """
        state idx: 0 => x coord
                   1 => y coord
                   2 => speed

        Scale to -1 to 1
        """
        norm_x = util.lmap(state[0], [0, self.width], [-1, 1])
        norm_y = util.lmap(state[1], [0, self.height], [-1, 1])
        # norm_speed = util.lmap(state[2], [self.min_speed, self.max_speed], [-1, 1])
        return np.array([norm_x, norm_y], dtype=np.float32)

    def _denormalize_state(self, state):
        original_x = util.lmap(state[0], [-1, 1], [0, self.width])
        original_y = util.lmap(state[1], [-1, 1], [0, self.height])
        # original_speed = util.lmap(state[2], [-1, 1], [self.min_speed, self.max_speed])
        return np.array([original_x, original_y], dtype=np.float32)

    def step_expert(self):
        self.step_count += 1
        last_pos = self.true_traj[-1]
        pre_curve = self.generate_curve(self.step_count)
        pre_speed = min(self.min_speed * ((self.step_count + 5) / 5), self.max_speed)
        next_x = last_pos[0] + int((pre_speed * math.cos(pre_curve)))
        next_y = last_pos[1] + int((pre_speed * math.sin(pre_curve)))
        next_point = (next_x, next_y)
        self.true_traj.append(next_point)
        done = (
            next_y > self.height
            or next_y < 0
            or next_x > self.width
            or next_x < 0
            or self.step_count > 1000
        )

        last_obs = self._normalize_state([last_pos[0], last_pos[1], self.last_speed])
        # action = self._normalize_action([pre_curve, pre_speed])
        action = self._normalize_action([pre_curve])
        # clip values to stay in observation space when leaving the world
        next_x_clipped = np.clip(next_x, 0, self.height)
        next_y_clipped = np.clip(next_y, 0, self.width)

        if done:
            self.final_obs = self._normalize_state(
                [next_x_clipped, next_y_clipped, pre_speed]
            )
        self.last_speed = pre_speed
        return last_obs, action, {}, done

    def generate_curve(self, step_count):
        return self.current_generator_curve(self.step_count)

    def step(self, action, expert=False):
        action = self._denormalize_action(action)

        self.step_count += 1
        last_agent_pos = self.agent_traj[-1]
        last_pos = self.true_traj[-1]
        alpha = action[0]
        # speed = action[1]
        pre_speed = min(self.min_speed * ((self.step_count + 5) / 5), self.max_speed)
        speed = pre_speed
        next_agent_x = last_agent_pos[0] + int((speed * math.cos(alpha)))
        next_agent_y = last_agent_pos[1] + int((speed * math.sin(alpha)))
        next_agent_point = (next_agent_x, next_agent_y)
        self.agent_traj.append(next_agent_point)

        pre_curve = self.generate_curve(self.step_count)
        # pre_speed = min(self.min_speed * ((self.step_count + 5) / 5), self.max_speed)
        # pre_speed = 10
        next_x = last_pos[0] + int((pre_speed * math.cos(pre_curve)))
        next_y = last_pos[1] + int((pre_speed * math.sin(pre_curve)))
        next_point = (next_x, next_y)
        self.true_traj.append(next_point)
        dist_from_path = math.sqrt(
            (next_agent_x - next_x) ** 2 + (next_agent_y - next_y) ** 2
        )

        # print(dist_from_path)
        # print(f'distance: {math.sqrt((next_agent_x - next_x)**2 + (next_agent_y - next_y)**2)}')
        # 75.199
        reward = norm.pdf(dist_from_path, 0, 5) * 12.5331  # scale amplitude to 1

        done = (
            next_agent_y > self.height
            or next_agent_y < 0
            or next_agent_x > self.width
            or next_agent_x < 0
            or self.step_count > 1000
            #or dist_from_path > 50
        )
        if done or reward < 0.001:
            reward = 0

        # clip values to stay in observation space when leaving the world
        next_agent_y = np.clip(next_agent_y, 0, self.height)
        next_agent_x = np.clip(next_agent_x, 0, self.width)

        self.curr_alpha = alpha
        self.curr_speed = speed
        self.curr_reward = reward
        return self._normalize_state([next_agent_x, next_agent_y]), reward, done, {}

    def reset(self):
        self.step_count = 0
        self.rng = np.random.default_rng(self._rng_seed)
        # only used for expert trajectory generation
        self.last_speed = self.min_speed
        self.current_generator_name, current_generator = random.choice(
            list(self.trajectories.items())
        )
        self.current_generator_curve = current_generator['func']
        self._reset_starting_pos(current_generator['starting_pos'])
        return self._normalize_state([10, 10])

    def reset_deterministically(self, idx):
        self.step_count = 0
        self.rng = np.random.default_rng(self._rng_seed)
        # only used for expert trajectory generation
        self.last_speed = self.min_speed
        self.current_generator_name, current_generator = list(self.trajectories.items())[idx]
        self.current_generator_curve = current_generator['func']
        self._reset_starting_pos(current_generator['starting_pos'])
        return self._normalize_state([10, 10])
    
    def render(self, mode="human"):
        if self.master == None:
            self.master = Tk()
            self.master.eval("tk::PlaceWindow . center")
            self.canvas = Canvas(
                self.master, width=self.width, height=self.height, bg="white"
            )
            self.canvas.pack()
        self.canvas.delete("all")
        self.canvas.create_line(self.true_traj, width=2, fill="black")
        self.canvas.create_line(self.agent_traj, width=2, fill="red")
        self.canvas.create_oval(
            self._generate_circle_coords(*self.true_traj[-1], r=5), fill="black"
        )
        self.canvas.create_oval(
            self._generate_circle_coords(*self.agent_traj[-1], r=5), fill="red"
        )
        self.canvas.create_text(
            500,
            20,
            fill="red",
            font="Times 10",
            text=f"[{self.agent_traj[-1][0]}, {self.agent_traj[-1][1]}] speed: {round(self.curr_speed)} heading: {round(self.curr_alpha, 3)}",
            anchor=tk.NW,
        )
        self.canvas.create_text(
            500,
            40,
            fill="black",
            font="Times 10",
            text=f"reward: {round(self.curr_reward, 3)}",
            anchor=tk.NW,
        )
        self.canvas.create_text(
            400,
            100,
            fill="black",
            font="Times 10",
            text=f"Trajectory: {self.current_generator_name}",
        )
        self.master.update()
        time.sleep(self.animation_delay)

    def close(self):
        pass

    def _generate_circle_coords(self, x, y, r):
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return x0, y0, x1, y1


print("REGISTER")
register(
    id="curve-v0",
    entry_point="deeprl.envs.curve:CurveEnv",
)
