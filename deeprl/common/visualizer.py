import numpy as np
from scipy.io import savemat
import deeprl.common.util as util
import matplotlib.pyplot as plt
from tkinter import *
import threading
import time 

class Visualizer(object):

    def __init__(self, save_path=''):
        self.save_path = save_path

    def __call__(self, env, agent, episode_reward_history):
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        fig, axs = plt.subplots(2, 2)
        a = np.random.random((16, 16))
        axs[0, 0].imshow(a, cmap='hot', interpolation='nearest')
        axs[0, 0].set_title('Axis [0, 0]')
        axs[0, 1].plot(range(0, len(episode_reward_history)), episode_reward_history)
        axs[0, 1].set_title('Axis [0, 1]')
        axs[1, 0].plot(range(0, len(episode_reward_history)), episode_reward_history)
        axs[1, 0].set_title('Axis [1, 0]')
        axs[1, 1].plot(range(0, len(episode_reward_history)), episode_reward_history)
        axs[1, 1].set_title('Axis [1, 1]')

        for ax in axs.flat:
            ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        plt.draw()
        plt.pause(0.001)