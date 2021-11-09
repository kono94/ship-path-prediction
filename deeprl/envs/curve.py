import gym
from gym import spaces
import numpy as np
from tkinter import *
import tkinter as tk
import random
import threading
import math
import time
from scipy.stats import norm

DEG2RAD = 0.0174533


class CurveEnv(gym.Env):
    """Custom Environment that follows gym interface
    
        actions: -1 to 1    (internally 0 to 360 degrees)

        state: [width, height] => [0-1, 0-1]        
    """

    def __init__(self, animation_delay=0):
        super(CurveEnv, self).__init__()
        self._rng_seed = 12345
        self.animation_delay = animation_delay
        self.width = 700
        self.height = 500
        self.min_speed = 5
        self.max_speed = 40
        self.min_angle = 0
        self.max_angle = 360

        # Action space is heading which is 0 to 360 degrees and speed from 0 to 40
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), 
                                       shape=(2,), dtype=np.float32)

        # obervation state (x, y, speed)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), 
                                        shape=(3,), dtype=np.int)

        self.master = Tk()
        self.master.eval('tk::PlaceWindow . center')
        self.canvas = Canvas(self.master,
                                width=self.width,
                                height=self.height,
                                bg='white')
        self.canvas.pack()
        self.true_traj = [(10, 10), (10, 10)]
        self.agent_traj = [(10, 10), (10, 10)]
        self.step_count = 0

        self.rng = np.random.default_rng(self._rng_seed)

        self.curr_alpha = 0
        self.curr_speed = 0
        self.curr_reward = 0

    def _denormalize_action(self, action):
        tmp = np.ones_like(action)
        spaces = {'low': [self.min_angle, self.min_speed],
                  'high': [self.max_angle, self.max_speed]}
        for i in range(0, len(action)):
            act_k = (spaces['high'][i] - spaces['low'][i])/ 2. 
            act_b = (spaces['high'][i] + spaces['low'][i])/ 2.  
            tmp[i] = act_k * action[i] + act_b

        return tmp
    
    def _denormalize_state(self, state):
        tmp = np.ones_like(state)
        spaces = {'low': [0, 0, self.min_speed],
                  'high': [self.width, self.height, self.max_speed]}

        for i in range(0, len(state)):
            act_k = (spaces['high'][i] - spaces['low'][i])/ 2. 
            act_b = (spaces['high'][i] + spaces['low'][i])/ 2.  
            tmp[i] = act_k * state[i] + act_b

        return tmp

    def _normalize_state(self, state):
        '''
        state idx: 0 => x coord
                   1 => y coord
                   2 => speed
                   
        Scale to 0 to 1
        '''
        new_x = 1/self.width * (state[0] - self.width) + 1
        new_y = 1/self.height * (state[1] - self.height) + 1
        new_speed = 1/self.min_speed * (state[2] - self.max_speed) + 1
        
        return (new_x, new_y, new_speed)

    def step(self, action):
        action = self._denormalize_action(action)

        self.step_count += 1
        last_agent_pos = self.agent_traj[-1]
        last_pos = self.true_traj[-1]
        alpha = action[0] * DEG2RAD
        speed = action[1]

        next_agent_x = last_agent_pos[0] + int((speed * math.cos(alpha)))
        next_agent_y = last_agent_pos[1] + int((speed * math.sin(alpha)))
        next_agent_point = (next_agent_x, next_agent_y)
        self.agent_traj.append(next_agent_point)

        pre_curve = self.rng.integers(low=20, high=60, size=1) * DEG2RAD
        pre_speed = min(self.min_speed * (self.step_count / 5), self.max_speed)
        next_x = last_pos[0] + int((pre_speed * math.cos(pre_curve)))
        next_y = last_pos[1] + int((pre_speed * math.sin(pre_curve)))
        next_point = (next_x, next_y)
        self.true_traj.append(next_point)

        last_agent_pos
        done = next_agent_y > self.height or next_agent_y < 0 or next_agent_x > self.width or next_agent_x < 0
      #  print(f'distance: {math.sqrt((next_agent_x - next_x)**2 + (next_agent_y - next_y)**2)}')
        reward = norm.pdf(math.sqrt((next_agent_x - next_x)**2 + (next_agent_y - next_y)**2),0,30) * 75.199 # scale amplitude to 1
        if reward < 0.001:
            reward = 0

        self.curr_alpha = alpha
        self.curr_speed = speed
        self.curr_reward = reward
        return self._normalize_state(next_agent_point + (speed,)), reward, done, {}

    def reset(self):
        self.true_traj = [(10, 10), (10, 10)]
        self.agent_traj = [(10, 10), (10, 10)]
        self.step_count = 0
        self.rng = np.random.default_rng(self._rng_seed)
        return self._normalize_state((10, 10, self.min_speed))

    def render(self, mode='human'):
        self.canvas.delete("all")
        self.canvas.create_line(self.true_traj, width=2, fill="black")
        self.canvas.create_line(self.agent_traj, width=2, fill="red")
        self.canvas.create_oval(self._generate_circle_coords(
            *self.true_traj[-1], r=5), fill="black")
        self.canvas.create_oval(self._generate_circle_coords(
            *self.agent_traj[-1], r=5), fill="red")
        self.canvas.create_text(500, 20,fill="red",font="Times 10",
                        text=f'[{self.agent_traj[-1][0]}, {self.agent_traj[-1][1]}] speed: {round(self.curr_speed)} heading: {round(self.curr_alpha, 3)}', anchor=tk.NW)
        self.canvas.create_text(500, 40,fill="black",font="Times 10",
                    text=f'reward: {round(self.curr_reward, 3)}', anchor=tk.NW)
                    
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
