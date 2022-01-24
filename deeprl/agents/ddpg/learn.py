
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import os
import sys
import random
import time 
import logging
import networks
import matplotlib.pyplot as plt
import deeprl.common.util as  util
from deeprl.common.normalize_actions import NormalizedActions
from deeprl.agents.ddpg.agent import DDPG
from deeprl.common.evaluator import Evaluator
from deeprl.common.visualizer import Visualizer
from scipy.io import savemat
from deeprl.envs.golf import GolfHiddenHoles
import deeprl.envs.curve
import json

logging.config.fileConfig('logger.conf')
logger = logging.getLogger('learn.py')

def train(num_iterations, agent, env,  evaluate, reward_barrier, step_barrier, visualize, output, max_episode_length=None, debug=True, render=False):

    step = episode = episode_steps = 0
    episode_reward = 0.
    current_state = None
    episode_reward_history = []
    validate_reward_history = []

    while step < num_iterations:
        # reset on new episode
        if current_state is None:
            current_state = util.preprocess_state(deepcopy(env.reset()), env)
            agent.reset()

        in_warmup = step <= args.warmup
        if in_warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(current_state)
            if render:
                env.render()

        # env response with next_observation, reward, terminate_info
        next_state, reward, done, info = env.step(action)
        next_state = util.preprocess_state(deepcopy(next_state), env)
      
        #print(next_state)
        # agent stores transition and update policy
        agent.remember(current_state, action, reward, next_state, done)
        if step > args.warmup:
            agent.update_policy()


        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        current_state = deepcopy(next_state)
        
        if done or (max_episode_length and episode_steps >= max_episode_length -1): # end of episode
            util.prGreen(f'#{episode}: episode_reward:{episode_reward} steps:{step}')
            episode_reward_history.append(episode_reward)
              # [optional] evaluate
            if evaluate is not None and not in_warmup and step > step_barrier and episode_reward > reward_barrier:
                agent.eval()
                policy = lambda x: agent.select_action(util.preprocess_state(x, env), pure=True)
                validate_reward = evaluate(env, agent, visualize=False)
                validate_reward_history.append(validate_reward)
                util.prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
                agent.train()
                if validate_reward > reward_barrier:
                    'Task marked as solved, early stopping'
                    break
            # reset
            current_state = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
    agent.save_model(output)
    # reduce amount of data points by applying moving average (1k points in total)
    

    def reduce_datapoints(data, points_to_plot):
        '''
        split dataset into chunks and then calucate the mean of all chunks.
        This is done to reduce the amount of datapoints to plot...
        '''
        if len(data) < points_to_plot:
            return data
        splits = np.array_split(episode_reward_history, points_to_plot)
        return [np.mean(x) for x in splits]

    pps_to_plot = 1000
    y = reduce_datapoints(episode_reward_history, pps_to_plot)
    x = range(0,len(y))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    plt.xlabel(f'Episode (+1 times {"1" if (len(episode_reward_history) / pps_to_plot) < 1 else len(episode_reward_history) / pps_to_plot}')
    plt.ylabel('Average Reward')
    ax.plot(x, y)
    plt.savefig(f'{output}_episode_reward_{0 if len(episode_reward_history) == 0 else episode_reward_history[-1]}.png')
    
    
    y = reduce_datapoints(validate_reward_history, pps_to_plot)
    x = range(0,len(y))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.xlabel(f'Episode bundle by 3 (+1 times {"1" if (len(validate_reward_history) / pps_to_plot) < 1 else len(validate_reward_history) / pps_to_plot}')
    plt.ylabel('Average Reward')
    ax.plot(x, y)
    plt.savefig(f'{output}_validate_reward_{0 if len(validate_reward_history) == 0 else validate_reward_history[-1]}.png')

def test(num_episodes, agent, env, evaluate, model_path, visualize=True):

    agent.load_weights(model_path)
    print(sum(agent.actor.fc3.weight[0]))
    policy = lambda x: agent.select_action(x, pure=True)
    episode_reward_history = []
    agent.eval()
    for i in range(num_episodes):
        validate_reward = evaluate(env, agent, visualize=visualize, save=True)
        util.prYellow(f'[Evaluate] #{i}: mean_reward:{validate_reward}')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default=1, type=int, help='Environment; 1=cartpole, 2=line following')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--actor_lr_rate', default=0.0001, type=float, help='actor net learning rate')
    parser.add_argument('--critic_lr_rate', default=0.0005, type=float, help='critic net learning rate')
    parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--replay_max_size', default=50000, type=int, help='replay buffer size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network; TAU')
    parser.add_argument('--theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=3, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500000, type=int, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=50000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon_max_decay', default=40000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=42, type=int, help='')
    parser.add_argument('--reward_barrier', default=30, type=int, help='')
    parser.add_argument('--step_barrier', default=40000, type=int, help='')
    parser.add_argument('--file_prefix', default='', type=str, help='file prefix')
    
    
    args = parser.parse_args()
    file_prefix = args.file_prefix
    if file_prefix == '':
        file_prefix = os.path.join(sys.path[1], 'runs', f'{args.env}_{args.hidden1}_{args.hidden2}_{args.reward_barrier}_{args.seed}')
        with open(f'{file_prefix}_args', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        with open(f'{file_prefix}_args', 'r') as f:
            args.__dict__ = json.load(f)
            args.mode = 'test'
            args.render = True
           
    if args.env == 1:
        env = gym.make('MountainCarContinuous-v0')
    elif args.env == 2:
        env = gym.make('Pendulum-v1')
    elif args.env == 3:
        env = GolfHiddenHoles()
    elif args.env == 4:
        env = gym.make('curve-simple-v0', animation_delay=0.05)

    if args.seed > 0:
        util.seeding(args.seed, env)

    nr_of_states = env.observation_space.shape[0]
    nr_of_actions = env.action_space.shape[0]
    logger.info(env.observation_space)

    agent = DDPG(nr_of_states, nr_of_actions, args)

    evaluate = None
    if args.validate_episodes > 2:
        evaluate = Evaluator(args.validate_episodes, file_prefix, max_episode_length=args.max_episode_length)

    visualize = Visualizer()

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, args.reward_barrier, args.step_barrier, visualize, file_prefix, max_episode_length=args.max_episode_length, debug=True, render=args.render)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, file_prefix,
            visualize=args.render)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))