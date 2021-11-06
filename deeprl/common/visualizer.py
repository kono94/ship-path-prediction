import numpy as np
from scipy.io import savemat
import deeprl.common.util as util
import matplotlib.pyplot as plt
from tkinter import *
import threading
import time 

class Visualizer(object):

    def __call__(self, env, agent, episode_reward_history):
        agent.eval()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        step_count = 100
        step0 = 2 / step_count
        step1 = 2 / step_count

        low0 = -1
        low1 = -1
        tmp = agent.is_training
        agent.is_training = False
        pred_values = np.zeros((step_count +1, step_count + 1))
        for i in range(step_count + 1):
            low1 = -1
            for j in range(step_count + 1):
                low1 += step1
                pred_values[j][i] = agent.select_action([low0, low1], decay_epsilon = False)
            low0 += step0
            
        agent.is_training = True
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        #im1 = ax1.imshow(pred_values, cmap='hot', interpolation='none')
        mesh = ax1.pcolormesh(pred_values, cmap = 'hot')
        mesh.set_clim(-1,1)
        ax1.set_title('Axis [0, 0]')
        fig.colorbar(mesh,ax=ax1)
        #fig.colorbar(im1, orientation='horizontal')

        ax2 = fig.add_subplot(122)
        im2 = ax2.plot(range(0, len(episode_reward_history)), episode_reward_history)
        ax2.set_title('Axis [0, 1]')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.03)
        agent.train()