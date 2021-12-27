import argparse
import gym
import pickle
import random
import sys
import torch
from argparse import Namespace
import numpy as np
import stable_baselines3 as sb3
import torch as th

import imitation.util.util as ut
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.adversarial import gail

# needs to be imported to register the custom environments
import deeprl.envs.curve

def set_seed(seed):
    torch.manual_seed(seed)
    # most crucial (and hidden) to shuffle deterministically
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.default_rng(seed)


class CustomFeedForwardPolicy(sb3.common.policies.ActorCriticPolicy):
    def __init__(self, net_arch, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=net_arch)


def policy_in_action(venv, policy):
    obs = venv.reset()
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(action)
        venv.render()
        if dones:
            obs = venv.reset()


def sample_expert_demonstrations(sample_env):
    trajectory_list = []
    for i in range(0, len(sample_env.trajectories)):
        done = False
        obs = []
        actions = []
        infos = []
        sample_env.reset_deterministically(i)
        while not done:
            transition = sample_env.step_expert()
            obs.append(transition[0])
            actions.append(transition[1])
            infos.append(transition[2])
            done = transition[3]    
        obs.append(sample_env.final_obs)
        trajectory_list.append(
            Trajectory(np.array(obs), np.array(actions), np.array(infos), terminal=True)
        )
    #with open("curve_expert_trajectory.pickle", "wb") as handle:
   #     pickle.dump(trajectory_list, handle)

  #  with open("curve_expert_trajectory.pickle", "rb") as f:
        # This is a list of `imitation.data.types.Trajectory`, where
        # every instance contains observations and actions for a single expert
        # demonstration.
    #    trajectories = pickle.load(f)
    return rollout.flatten_trajectories(trajectory_list)


def train_BC(venv, expert_transitions, steps, net_arch, policy_save_path):
    """
    Train BC on expert data.
    """
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=expert_transitions,
        policy=CustomFeedForwardPolicy(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            net_arch=net_arch,
            lr_schedule=bc.ConstantLRSchedule(th.finfo(th.float32).max),
        ),
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
        demo_batch_size=16,
        gen_algo=sb3.PPO(
            sb3.policies.ActorCriticPolicy,
            venv,
            verbose=1,
            batch_size=32,
            n_epochs=3,
            policy_kwargs={"net_arch": net_arch},
    ),
    # gen_algo=sb3.DDPG("MlpPolicy", venv, verbose=1),
    allow_variable_horizon=True,
    )

    gail_trainer.train(21000)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Imitation learning for curve environment"
    )

    parser.add_argument(
        "--mode", default="train", type=str, help="support option: train/test"
    )
    parser.add_argument(
        "--algo", default="bc", type=str, help="algorithm to use; 'bc', 'gail'"
    )
    parser.add_argument("--env", default="curve-simple-v0", type=str, help="Environment;")
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

    args = parser.parse_args()
    #args = Namespace(algo='bc', animation_delay=1.0, env='curve-simple-v0', hidden1=128, hidden2=128, mode='test', policy_path='bc_policy.pth', training_steps=50000)

    set_seed(args.seed)
    
    if args.mode == "train":
        venv = ut.make_vec_env(args.env, n_envs=1)
        transitions = sample_expert_demonstrations(gym.make(args.env))
        if args.algo == "bc":
            train_BC(venv, transitions, args.training_steps, [args.hidden1, args.hidden2], args.policy_path)
        elif args.algo == "gail":
            train_GAIL(venv, transitions, args.training_steps, [args.hidden1, args.hidden2], args.policy_path)
        else:
            print("Unknown algorithm")
            sys.exit(2)
        pass
    elif args.mode == "test":
        if args.policy_path == "":
            print("Provide a path to a saved policy in parameter --policy_path")
            sys.exit(2)
        venv = ut.make_vec_env(args.env, n_envs=1, env_make_kwargs={'animation_delay': args.animation_delay})
        policy_in_action(venv, bc.reconstruct_policy(args.policy_path))
