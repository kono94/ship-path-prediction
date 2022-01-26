import argparse
import gym
import pickle
import random
import os
import sys
from matplotlib.pyplot import step
import torch
from argparse import Namespace
import numpy as np
import stable_baselines3 as sb3
import torch as th
from tqdm import tqdm
from stable_baselines3.common import utils
import pandas as pd
import imitation.util.util as ut
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.adversarial import gail
from imitation.policies import serialize

# needs to be imported to register the custom environments
import deeprl.envs.curve
import deeprl.envs.ais_env
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

print(torch.cuda.is_available())


def set_seed(seed):
    torch.manual_seed(seed)
    # most crucial (and hidden) to shuffle deterministically
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)


class CustomFeedForwardPolicy(sb3.common.policies.ActorCriticPolicy):
    def __init__(self, net_arch, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=net_arch)


def policy_in_action(venv, policy, nr_samples, evalution_path):
    df = pd.DataFrame(columns=['id', 'ep_length', 'cum_reward', 'performance'])
    obs = venv.reset()
    cum_reward = 0
    t = 0
    for i in tqdm(range(0, nr_samples)):
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _ = venv.step(action)
            cum_reward += reward[0]
            t += 1
            venv.render(mode="human")
        if done:
            obs = venv.reset()
            df = df.append({'id': i+1, 'ep_length': t, 'cum_reward': cum_reward, 'performance': cum_reward/t}, ignore_index=True)
           # print(f'cum:{cum_reward} t:{t}')
            cum_reward = 0
            t = 0
            
    df.to_csv(evalution_path)


def sample_expert_demonstrations(sample_env, expert_samples_path, n_samples):
    trajectory_list = []
    for i in tqdm(range(0, n_samples)):
        sample_env.reset()
        done = False
        obs = []
        actions = []
        infos = []
        while not done:
            transition = sample_env.step_expert()
            obs.append(transition[0])
            actions.append(transition[1])
            infos.append(transition[2])
            done = transition[3]
            #sample_env.render()

        sample_env.finish()
        obs.append(sample_env.next_obs)
        trajectory_list.append(
            Trajectory(np.array(obs), np.array(actions), np.array(infos), terminal=True)
        )
        
    with open(expert_samples_path, "wb") as handle:
       pickle.dump(rollout.flatten_trajectories(trajectory_list), handle)

    return rollout.flatten_trajectories(trajectory_list)


def train_BC(venv, expert_transitions, steps, net_arch, policy_save_path):
    """
    Train BC on expert data.
    """
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=expert_transitions,
        batch_size=64,
        policy=CustomFeedForwardPolicy(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            net_arch=net_arch,
            lr_schedule=bc.ConstantLRSchedule(th.finfo(th.float32).max),
        )
    )
    bc_trainer.train(n_epochs=steps)
    bc_trainer.save_policy(policy_save_path)


def train_GAIL(venv, expert_transitions, steps, net_arch, policy_save_path):
    """  
    Train GAIL on expert data.
    GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
    iterates over dictionaries containing observations, actions, and next_observations.
    """
    gail_trainer = gail.GAIL(
        venv=venv,
        demonstrations=expert_transitions,
        demo_batch_size=32,
        gen_algo=sb3.PPO(
            sb3.common.policies.ActorCriticPolicy,
            venv,
            verbose=1,
            batch_size=32,
            n_epochs=3,
            policy_kwargs={"net_arch": net_arch},
    ),
    # gen_algo=sb3.DDPG("MlpPolicy", venv, verbose=1),
    allow_variable_horizon=True,
    )

    gail_trainer.train(total_timesteps=steps)
    th.save(gail_trainer.policy, policy_save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Imitation learning for the ais environment"
    )

    parser.add_argument(
        "--mode", default="train", type=str, help="support option: train/test/sample"
    )
    parser.add_argument(
        "--algo", default="bc", type=str, help="algorithm to use; 'bc', 'gail'"
    )
    parser.add_argument("--env", default="ais-v0", type=str, help="Environment;")
    parser.add_argument(
        "--hidden1",
        default=128,
        type=int,
        help="hidden num of first fully connect layer in policy network",
    )
    parser.add_argument(
        "--hidden2",
        default=128,
        type=int,
        help="hidden num of second fully connect layer in policy network",
    )
    parser.add_argument("--training_steps", default=50000, type=int, help=""),
    parser.add_argument("--seed", default=3, type=int, help=""),
    parser.add_argument("--animation_delay", default=0.1, type=float, help=""),
    parser.add_argument(
        "--policy_path", default="policy.pth", type=str, help="Load policy and visual in env"
    )
    parser.add_argument(
        "--expert_samples_path", default="curve_expert_trajectory.pickle", type=str, help="expert trajectories files"
    )
    parser.add_argument("--n_samples", default=30, type=int, help="Number of trajectories to learn and eval on"),
    parser.add_argument(
        "--evaluation_path", default="evaluation.csv", type=str, help="Path to store the evaluation dataframe"
    )
    args = parser.parse_args()
    #args = Namespace(algo='bc', animation_delay=1.0, env='curve-simple-v0', hidden1=128, hidden2=128, mode='test', policy_path='bc_policy.pth', training_steps=50000)

    set_seed(args.seed)
    if (args.mode == "sample" or args.mode == "train") and args.expert_samples_path == "":
        print("Provide a path to a saved the expert samples --expert_samples_path")
        sys.exit(2)
        
    if args.mode == "sample":
        sample_expert_demonstrations(gym.make(args.env), args.expert_samples_path, args.n_samples)
    if args.mode == "train":
        venv = ut.make_vec_env(args.env, n_envs=1)
        with open(args.expert_samples_path, "rb") as f:
            # This is a list of `imitation.data.types.Trajectory`, where
            # every instance contains observations and actions for a single expert
            # demonstration.
            transitions = pickle.load(f)
        if args.algo == "bc":
            train_BC(venv, transitions, args.training_steps, [args.hidden1, args.hidden2], args.policy_path)
        elif args.algo == "gail":
            train_GAIL(venv, transitions, args.training_steps, [args.hidden1, args.hidden2], args.policy_path)
        else:
            print("Unknown algorithm provided by --algo")
            sys.exit(2)
        pass
    elif args.mode == "test":
        if args.policy_path == "":
            print("Provide a path to a saved policy in parameter --policy_path")
            sys.exit(2)
        venv = ut.make_vec_env(args.env, n_envs=1)
        if args.algo == "bc":
            policy_in_action(venv, bc.reconstruct_policy(args.policy_path), args.n_samples, args.evaluation_path)
        elif args.algo == "gail":
            policy = th.load(args.policy_path, map_location=utils.get_device("auto"))
            policy_in_action(venv, policy, args.n_samples, args.evaluation_path)