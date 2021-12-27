from tkinter.constants import N
from typing import Tuple
import gym
import numpy as np

import random
import math
import time

import deeprl.common.util as util
from scipy.stats import norm


class Position:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def to_tuples(positions):
        tuples = list()
        for i, elem in enumerate(positions):
            tuples.append((elem.x, elem.y))

        return tuples


class CurveBase(gym.Env):
    """Custom Environment that follows gym interface
    actions: -1 to 1    (internally -pi to pi and potential speed)
    state: -1 to 1      (internally to width, height)

    The agent has to "follow" one or multiple predefined (or rather generated) paths by selection
    the next heading to be as close to the >true< trajectory as possible.

    The reward function is modeled as Gaussian Track Error as described in:
    "Curved path following with deep reinforcement learning: Results from three vessel models" by
    Martinsen, Andreas B and Lekkas, Anastasios M

    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8604829
    """

    def __init__(self, animation_delay=0.1):
        super(CurveBase, self).__init__()
        self._rng_seed = 12345
        self.animation_delay = animation_delay
        self.width = 700
        self.height = 500
        self.min_speed = 5
        self.max_speed = 40

        # Sub-classes do not have to utilize all of them and just define a
        # subset of them
        self.STATE_DEFINITION_SET = {
            0: {"name": "x position", "min": 0, "max": self.width},
            1: {"name": "y position", "min": 0, "max": self.height},
            2: {"name": "heading / angle", "min": -math.pi, "max": math.pi},
            3: {"name": "current speed", "min": self.min_speed, "max": self.max_speed},
        }

        self.ACTION_DEFINITION_SET = {
            0: self.STATE_DEFINITION_SET[2],
            1: self.STATE_DEFINITION_SET[3],
        }

        # Action space is heading which is -pi to pi and speed from 5 to 40
        # action(angle, speed)
        # obervation state (x, y, speed)
        # DEFINED BY DERIVED CLASSES
        self.action_space = None
        self.observation_space = None

        # Curves to be generated
        self.trajectories = {
            "sinus-small": {
                "starting_pos": (20, 400),
                "func": lambda t: math.sin(t / 1.5) - math.pi / 6,
            },
            "sinus-big": {
                "starting_pos": (400, 20),
                "func": lambda t: math.sin(t / 2) + math.pi * 0.6,
            },
            "sinus-ultra": {
                "starting_pos": (10, 10),
                "func": lambda t: math.sin(t / 4) + math.pi / 6,
            },
        }
        self.current_generator_curve = None
        self.current_generator_name = None

        # curve the agent has to follow
        self.true_traj = None
        # curve that the agent
        self.agent_traj = None
        self.step_count = 0

        self.rng = np.random.default_rng(self._rng_seed)

        self.position = None
        self.heading = None
        self.speed = None

        self.agent_position = None
        self.agent_heading = None
        self.agent_speed = None
        self.agent_reward = 0

        self.master = None

    def _reset_starting_pos(self, starting_pos: Tuple[int, int]):
        x = starting_pos[0]
        y = starting_pos[1]
        pos = Position(x, y)
        self.true_traj = [pos, pos]
        self.agent_traj = [pos, pos]
        self.agent_position = pos
        self.position = pos

    def _normalization_helper(self, object, range_definition, denormalize=False):
        tmp = np.empty_like(object, dtype=np.float32)
        for i, elem in enumerate(object):
            tmp[i] = (
                util.lmap(
                    elem,
                    [range_definition[i]["min"], range_definition[i]["max"]],
                    [-1, 1],
                )
                if not denormalize
                else util.lmap(
                    elem,
                    [-1, 1],
                    [range_definition[i]["min"], range_definition[i]["max"]],
                )
            )
        return tmp

    def _normalize_action(self, action):
        return self._normalization_helper(action, self.ACTION_DEFINITION_SET, False)

    def _denormalize_action(self, action):
        return self._normalization_helper(action, self.ACTION_DEFINITION_SET, True)

    def _normalize_state(self, state):
        return self._normalization_helper(state, self.STATE_DEFINITION_SET, False)

    def _denormalize_state(self, state):
        return self._normalization_helper(state, self.STATE_DEFINITION_SET, True)

    def _next_true_speed(self):
        return min(self.min_speed * ((self.step_count + 5) / 5), self.max_speed)

    def _calc_next_position(self, last_pos, angle, speed):
        next_x = last_pos.x + int((speed * math.cos(angle)))
        next_y = last_pos.y + int((speed * math.sin(angle)))
        next_pos = Position(next_x, next_y)
        return next_pos

    def _expert_output(self, last_pos, heading, speed):
        return self._normalize_state([last_pos.x, last_pos.y]), \
               self._normalize_action([heading])
               
    def step_expert(self):
        self.step_count += 1
        last_pos = self.true_traj[-1]
        heading = self.generate_heading(self.step_count)
        speed = self._next_true_speed()
        next_pos = self._calc_next_position(last_pos, heading, speed)
        self.true_traj.append(next_pos)

        done = (
            next_pos.y > self.height
            or next_pos.y < 0
            or next_pos.x > self.width
            or next_pos.x < 0
            or self.step_count > 1000
        )

        norm_last_obs, norm_action = self._expert_output(last_pos, self.last_heading, self.last_speed, heading, speed)

        if done:
            # clip values to stay in observation space when leaving the world
            next_x_clipped = np.clip(next_pos.x, 0, self.height)
            next_y_clipped = np.clip(next_pos.y, 0, self.width)
            self.final_obs,_ = self._expert_output(Position(next_x_clipped, next_y_clipped), self.last_heading, self.last_speed, heading, speed)
            
        self.last_speed = speed
        self.last_heading = heading
        
        return norm_last_obs, norm_action, {}, done

    def generate_heading(self, step_count):
        return self.current_generator_curve(step_count)

    def _next_agent_heading(self):
        return self.current_action[0]

    def _next_agent_speed(self):
        return self.speed

    def _calculate_reward(self, dist_to_path):
        return norm.pdf(dist_to_path, 0, 5) * 12.5331  # scale amplitude to 1

    def _step_observation(self):
        raise NotImplementedError

    def step(self, action, expert=False):
        self.step_count += 1
        self.current_action = self._denormalize_action(action)

        # True trajectory
        self.heading = self.generate_heading(self.step_count)
        self.speed = self._next_true_speed()
        self.position = self._calc_next_position(
            self.position, self.heading, self.speed
        )
        self.true_traj.append(self.position)

        # Agent calculation
        self.agent_heading = self._next_agent_heading()
        self.agent_speed = self._next_agent_speed()
        self.agent_position = self._calc_next_position(
            self.agent_position, self.agent_heading, self.agent_speed
        )
        self.agent_traj.append(self.agent_position)

        dist_to_path = math.sqrt(
            (self.agent_position.x - self.position.x) ** 2
            + (self.agent_position.y - self.position.y) ** 2
        )
        reward = self._calculate_reward(dist_to_path)

        done = (
            self.agent_position.y > self.height
            or self.agent_position.y < 0
            or self.agent_position.x > self.width
            or self.agent_position.x < 0
            or self.step_count > 1000
            or dist_to_path > 50
        )

        if done or reward < 0.001:
            reward = 0

        self.agent_reward = reward

        # clip values to stay in observation space when leaving the world
        self.agent_position = Position(
            np.clip(self.agent_position.x, 0, self.width),
            np.clip(self.agent_position.y, 0, self.height),
        )

        return self._step_observation(), reward, done, {}

    def _reset_env(self, deterministic_idx=None):
        self.step_count = 0
        self.rng = np.random.default_rng(self._rng_seed)
        self.current_generator_name, current_generator = (
            list(self.trajectories.items())[deterministic_idx]
            if deterministic_idx is not None
            else random.choice(list(self.trajectories.items()))
        )
        self.current_generator_curve = current_generator["func"]
        # only used for expert trajectory generation
        self.speed= self.min_speed
        self.heading = self.generate_heading(0)
        self.last_heading = self.heading
        self.last_speed = self.min_speed
        
        self.agent_heading = self.heading
        self.agent_speed = self.speed
        
        self._reset_starting_pos(current_generator["starting_pos"])

    def _reset(self):
        self._reset_env()

    def _reset_deterministically(self, idx):
        self._reset_env(idx)

    def render(self, sampling=False):
        if self.master == None:
            global tk
            import tkinter as tk

            self.master = tk.Tk()
            self.master.eval("tk::PlaceWindow . center")
            self.canvas = tk.Canvas(
                self.master, width=self.width, height=self.height, bg="white"
            )
            self.canvas.pack()
        true_traj_tuples = Position.to_tuples(self.true_traj)
        agent_traj_tuples = Position.to_tuples(self.agent_traj)
        self.canvas.delete("all")

        if not sampling:
            self.canvas.create_line(agent_traj_tuples, width=2, fill="red")
            self.canvas.create_oval(
                self._generate_circle_coords(*agent_traj_tuples[-1], r=5), fill="red"
            )
            self.canvas.create_text(
                400,
                20,
                fill="red",
                font="Times 10",
                text=f"[{agent_traj_tuples[-1][0]}, {agent_traj_tuples[-1][1]}] speed: {round(self.agent_speed)} heading: {round(self.agent_heading, 3)}",
                anchor=tk.NW,
            )
            self.canvas.create_text(
                500,
                40,
                fill="black",
                font="Times 10",
                text=f"reward: {round(self.agent_reward, 3)}",
                anchor=tk.NW,
            )
            self.canvas.create_text(
                400,
                100,
                fill="black",
                font="Times 10",
                text=f"Trajectory: {self.current_generator_name}",
            )
        else:
            self.canvas.create_text(
                400,
                100,
                fill="black",
                font="Times 10",
                text=f"CURRENTLY SAMPING EXPERT TRAJECTORY FOR: \n {self.current_generator_name}",
            )

        self.canvas.create_line(true_traj_tuples, width=2, fill="black")
        self.canvas.create_oval(
            self._generate_circle_coords(*true_traj_tuples[-1], r=5), fill="black"
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
