import numpy as np
import matplotlib.pyplot as plt
import pickle
from dapi_env import *

n_seeds = 10
dapi_paths = {
    'fixed-start': KinkAndWell(timesteps=200, n_paths=500, seed=0),
    'free-start': FreeArmLogistic(timesteps=200, n_paths=500, T=20, seed=0),
    'ribbons': RotateRibbon(timesteps=200, n_paths=500, seed=0),
    'circles': Circles(timesteps=200, n_paths=500, seed=0),
}

def plot(all_states, all_xy_actions, method):

    for seed in range(n_seeds):
        plt.plot(all_states[seed,:,0], all_states[seed,:,1], label='observation')
        plt.plot(all_xy_actions[seed,:,0], all_xy_actions[seed,:,1], label ='policy')
        plt.legend()
        plt.title(f'{method}, seed {seed}')
        plt.show()

def plot_best(best_bc, best_gail, best_airl, best_dapi, num_epsisodes=100):
    
    omega = 0.9
    gamma = 0.9
    t = np.linspace(0, 10, 200)
    fontsize = 13

    plt.plot(best_bc[:,0], best_bc[:,1], label='BC')
    plt.plot(best_gail[:,0], best_gail[:,1], label='GAIL')
    plt.plot(best_airl[:,0], best_airl[:,1], label='AIRL')
    plt.plot(best_dapi[:,0], best_dapi[:,1]-0.5, label='DAPI')

    for alpha in np.linspace(5, 10, num_epsisodes):
        x = np.sqrt(alpha * t)
        y = np.cos(omega * x**2/alpha) * np.exp(-gamma*x**2/alpha)
        plt.plot(x,y, color='gray', alpha=0.05)

    plt.legend(prop={'size':fontsize})
    plt.xlabel(r'$x$', fontsize=fontsize)
    plt.ylabel(r'$y$', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    # plt.savefig('/home/sola_ed/Documents/paper_dapi/results.pdf')
    plt.show()

for name, _ in dapi_paths.items():
    with open(f'examples/output/BC/rollouts_{name}.pkl', 'rb') as f:
        all_states, all_xy_actions = pickle.load(f)
    plot(all_states, all_xy_actions, 'BC')
    best_bc = all_xy_actions[5]

    with open(f'examples/output/GAIL/rollouts_{name}.pkl', 'rb') as f:
        all_states, all_xy_actions = pickle.load(f)
    plot(all_states, all_xy_actions, 'GAIL')
    best_gail = all_xy_actions[5]

    with open(f'examples/output/AIRL/rollouts_{name}.pkl', 'rb') as f:
        all_states, all_xy_actions = pickle.load(f)
    plot(all_states, all_xy_actions, 'AIRL')
    best_airl = all_xy_actions[8]

    # with open(f'/home/sola_ed/Projects/rlearning/aisRL/dapi/log/dapi/rollouts_0.pkl', 'rb') as f:
    #     all_states, all_xy_actions = pickle.load(f)
    # # for k in range(10):
    # #     plt.plot(all_states[k][:,0], all_states[k][:,1])
    # #     plt.plot(all_xy_actions[k][:,0], all_xy_actions[k][:,1])
    # #     plt.show()
    # best_dapi = all_xy_actions[0]

    # plot_best(best_bc, best_gail, best_airl, best_dapi)