import argparse
import gym
import random
import sys
import torch
import numpy as np
import stable_baselines3 as sb3
import pandas as pd
import imitation.util.util as ut
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.adversarial import gail
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# needs to be imported to register the custom environments
import deeprl.envs.curve


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


def policy_in_action(venv, policy, evaluation_path):
    df = pd.DataFrame(columns=["id", "timestep", "distance"])
    for i in range(0, len(venv.trajectories)):
        obs = venv.reset_deterministically(i)
        distances = []
        done = False
        t = 0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(action, evaluate=True)
            dist = info["distance"]
            name = info["name"]
            dtw = info["dtw"]
            distances.append(dist)
            venv.render(False)
            df = df.append(
                {"id": i, "name": name, "timestep": t, "distance": dist, "dtw": dtw},
                ignore_index=True,
            )
            t += 1
            if done:
                # time.sleep(20)
                obs = venv.reset()
        df.to_csv(evaluation_path + ".csv", index=False)


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
            # sample_env.render(True)
        obs.append(sample_env.final_obs)
        trajectory_list.append(
            Trajectory(np.array(obs), np.array(actions), np.array(infos), terminal=True)
        )
    return rollout.flatten_trajectories(trajectory_list)


def train_DDPG(venv, seed, steps, net_arch, policy_save_path):
    n_actions = venv.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.ones(n_actions), sigma=0.05 * np.ones(n_actions)
    )

    policy_kwargs = dict(net_arch=dict(pi=net_arch, qf=net_arch))

    model = DDPG(
        "MlpPolicy",
        venv,
        action_noise=action_noise,
        buffer_size=50000,
        verbose=1,
        seed=seed,
        gamma=0.999,
        tau=1e-3,
        learning_rate=1e-4,
        batch_size=128,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=steps, log_interval=1)
    model.save(policy_save_path)


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
            lr_schedule=bc.ConstantLRSchedule(lr=1e-4),
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
    parser.add_argument(
        "--env", default="curve-simple-v0", type=str, help="Environment;"
    )
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
        "--policy_path",
        default="policy.pth",
        type=str,
        help="Load policy and visual in env",
    )
    parser.add_argument(
        "--evaluation_path",
        default="evaluation.csv",
        type=str,
        help="Path to store the evaluation dataframe",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        venv = ut.make_vec_env(args.env, n_envs=1)

        if args.algo == "ddpg":
            train_DDPG(
                venv,
                args.seed,
                args.training_steps,
                [args.hidden1, args.hidden2],
                args.policy_path,
            )
        elif args.algo == "bc":
            transitions = sample_expert_demonstrations(gym.make(args.env))
            train_BC(
                venv,
                transitions,
                args.training_steps,
                [args.hidden1, args.hidden2],
                args.policy_path,
            )
        elif args.algo == "gail":
            transitions = sample_expert_demonstrations(gym.make(args.env))
            train_GAIL(
                venv,
                transitions,
                args.training_steps,
                [args.hidden1, args.hidden2],
                args.policy_path,
            )
        else:
            print("Unknown algorithm")
            sys.exit(2)

    elif args.mode == "test":
        if args.policy_path == "":
            print("Provide a path to a saved policy in parameter --policy_path")
            sys.exit(2)

        venv = gym.make(args.env, animation_delay=args.animation_delay)
        if args.algo == "ddpg":
            policy_in_action(
                venv, DDPG.load(args.policy_path, env=venv), args.evaluation_path
            )
        elif args.algo == "bc":
            policy_in_action(
                venv, bc.reconstruct_policy(args.policy_path), args.evaluation_path
            )
