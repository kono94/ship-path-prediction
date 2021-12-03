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

    def __init__(self, animation_delay=0):
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
        self.action_space = spaces.Box(low=-1, high=1, 
                                       shape=(2,), dtype=np.float32)

        # obervation state (x, y, speed)
        self.observation_space = spaces.Box(low=-1, high=1, 
                                        shape=(3,), dtype=np.float32)

        
        self.true_traj = [(10, 10), (10, 10)]
        self.agent_traj = [(10, 10), (10, 10)]
        self.step_count = 0

        self.rng = np.random.default_rng(self._rng_seed)

        self.curr_alpha = 0
        self.curr_speed = 0
        self.curr_reward = 0
        
        self.master = None

    def _denormalize_action(self, action):
       original_angle = util.lmap(action[0], [-1, 1], [self.min_angle, self.max_angle])
       original_speed = util.lmap(action[1], [-1, 1], [self.min_speed, self.max_speed])
       return np.array([original_angle, original_speed], dtype=np.float32)
    
    def _denormalize_state(self, state):
        original_x = util.lmap(state[0], [-1, 1], [0, self.width])
        original_y = util.lmap(state[1], [-1, 1], [0, self.height])
        original_speed = util.lmap(state[2], [-1, 1], [self.min_speed, self.max_speed])
        return np.array([original_x, original_y, original_speed], dtype=np.float32)


    def _normalize_state(self, state):
        '''
        state idx: 0 => x coord
                   1 => y coord
                   2 => speed
                   
        Scale to -1 to 1
        '''
        norm_x = util.lmap(state[0], [0, self.width], [-1, 1])
        norm_y = util.lmap(state[1], [0, self.height], [-1, 1])
        norm_speed = util.lmap(state[2], [self.min_speed, self.max_speed], [-1, 1])
        return np.array([norm_x, norm_y, norm_speed], dtype=np.float32)

    def step(self, action):
        action = self._denormalize_action(action)
      
        self.step_count += 1
        last_agent_pos = self.agent_traj[-1]
        last_pos = self.true_traj[-1]
        alpha = action[0]
        speed = action[1]

        next_agent_x = last_agent_pos[0] + int((speed * math.cos(alpha)))
        next_agent_y = last_agent_pos[1] + int((speed * math.sin(alpha)))
        next_agent_point = (next_agent_x, next_agent_y)
        self.agent_traj.append(next_agent_point)

        pre_curve = self.rng.integers(low=20, high=60, size=1) 
        pre_curve = 45
        pre_speed = min(self.min_speed * (self.step_count / 5), self.max_speed)
        next_x = last_pos[0] + int((pre_speed * math.cos(pre_curve)))
        next_y = last_pos[1] + int((pre_speed * math.sin(pre_curve)))
        next_point = (next_x, next_y)
        self.true_traj.append(next_point)

        # clip values to stay in observation space when leaving the world
        next_agent_y = np.clip(next_agent_y, 0, self.height)
        next_agent_x = np.clip(next_agent_x, 0, self.width)
        
        dist_from_path = math.sqrt((next_agent_x - next_x)**2 + (next_agent_y - next_y)**2)
        #print(f'distance: {math.sqrt((next_agent_x - next_x)**2 + (next_agent_y - next_y)**2)}')
        reward = norm.pdf(dist_from_path,0,30) * 75.199 # scale amplitude to 1
       
        done = next_agent_y > self.height or next_agent_y < 0 or \
               next_agent_x > self.width or next_agent_x < 0 or \
               self.step_count > 1000 or dist_from_path > 10
        if done or reward < 0.001:
            reward = 0
            

        self.curr_alpha = alpha
        self.curr_speed = speed
        self.curr_reward = reward
        return self._normalize_state((next_agent_x,) + (next_agent_y,) + (speed,)), reward, done, {}

    def reset(self):
        self.true_traj = [(10, 10), (10, 10)]
        self.agent_traj = [(10, 10), (10, 10)]
        self.step_count = 0
        self.rng = np.random.default_rng(self._rng_seed)
        return self._normalize_state((10, 10, self.min_speed))

    def render(self, mode='human'):
        if self.master == None:
            self.master = Tk()
            self.master.eval('tk::PlaceWindow . center')
            self.canvas = Canvas(self.master,
                                    width=self.width,
                                    height=self.height,
                                    bg='white')
            self.canvas.pack()
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


register(
    id='curve-v0',
    entry_point='deeprl.envs.curve:CurveEnv',
)