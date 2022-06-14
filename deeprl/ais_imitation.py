import argparse
import gym
import pickle
import random
import os
import sys
import torch
import numpy as np
import stable_baselines3 as sb3
from tqdm import tqdm
import pandas as pd
import time
import imitation.util.util as ut
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.rewards import reward_nets
from imitation.algorithms.adversarial import gail
from statistics import mean
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import ast

# needs to be imported to register the custom environments
import ais_env

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print(torch.cuda.is_available())
TRAIN_SPLIT = 0.85


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


OUTPUT = ""


def policy_in_action(env, policy, evalution_path, render):
    df = pd.DataFrame(columns=["id", "ep_length", "cum_reward", "performance"])
    n_trajs = env.get_trajectory_count()
    start_index = int(TRAIN_SPLIT * n_trajs)
    env.set_trajectory_index(start_index)  # +1 with the first reset()
    obs = env.reset()
    cum_reward = 0
    t = 0
    saved_tracks = []
    for i in tqdm(range(0, n_trajs - start_index - 1)):
        # for i in tqdm(range(0, 2)):
        done = False
        distances = []
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            distances.append(info["distance_in_meters"])
            cum_reward += reward
            t += 1 
           # env.render(mode="human", svg=None)      
        #if mean(distances) > 1500 and t > 100:
        #    print(mean(distances))
            #env.render(mode="human", svg=f'{mean(distances)}_{i}')
            #time.sleep(3)
            #np.save(f'deeprl/scripts/improved_{i}_agent.npy', env.agent_traj)
            #np.save(f'deeprl/scripts/improved_{i}_true.npy', env.true_traj)
      #     saved_tracks.append((env.agent_traj, env.true_traj))
        #      sys.exit(1)
        global OUTPUT
            
       # if render:
            # save as svg = f'DOUBLE_{OUTPUT}_mean_distance={int(mean(distances))}'
     #       env.render(mode="human", svg=None)
        obs = env.reset()
        df = df.append(
            {
                "id": i + 1,
                "ep_length": t,
                "cum_reward": cum_reward,
                "performance": cum_reward / t,
                "distances": distances,
            },
            ignore_index=True,
        )
        # print(f"'id': {i+1}, 'ep_length': {t}, 'cum_reward': {cum_reward}, 'performance': {cum_reward/t}'distances': {mean(distances)}")
        # print(f'cum:{cum_reward} t:{t}')

        cum_reward = 0
        t = 0
    #print(saved_tracks)
   # for s in saved_tracks:
   #     for i in range(0, len(s[0])):
   #         env.render("human", None, s[0][:i], s[1][:i])
    df.to_csv(evalution_path)


def sample_expert_demonstrations(sample_env, expert_samples_path):
    trajectory_list = []
    n_trajectories = sample_env.get_trajectory_count()
    print("recording expert trajectories...")
    for i in tqdm(range(0, int(n_trajectories * TRAIN_SPLIT))):
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
    global OUTPUT
    batch = 32
    lr = 5e-5
    OUTPUT = f"{OUTPUT}_batch={batch}_net={net_arch}_steps={steps}_lr={lr}"
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=expert_transitions,
        batch_size=batch,
        policy=CustomFeedForwardPolicy(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            net_arch=net_arch,
            lr_schedule=bc.ConstantLRSchedule(lr=lr),
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
    # the noise objects for DDPG
    n_actions = venv.action_space.shape[-1]
    action_noise = sb3.common.noise.NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )
    policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300]))
    gail_reward_net = reward_nets.BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
    )
    gail_trainer = gail.GAIL(
        venv=venv,
        demonstrations=expert_transitions,
        demo_batch_size=64,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
        reward_net=gail_reward_net,
        # gen_algo=sb3.DDPG("MlpPolicy", venv, verbose=1),
        allow_variable_horizon=True,
    )

    gail_trainer.train(total_timesteps=steps)
    gail_trainer.gen_algo.save(f"{policy_save_path}.zip")
    print(f"{policy_save_path}.zip")
    print(
        gail_trainer.gen_algo.predict([0.26154423, 0.9519157, 0.32064462, 0.00668896])
    )
    # torch.save(gail_trainer.policy, policy_save_path)


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
        "--network",
        default="[256,128,64]",
        type=str,
        help="Network structure of hidden layers described as string array, e.g. '[64,32]'",
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
        "--expert_samples_path",
        default="curve_expert_trajectory.pickle",
        type=str,
        help="expert trajectories files",
    )
    parser.add_argument(
        "--evaluation_path",
        default="evaluation.csv",
        type=str,
        help="Path to store the evaluation dataframe",
    )
    parser.add_argument("--render", dest="render", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if (
        args.mode == "sample" or args.mode == "train"
    ) and args.expert_samples_path == "":
        print("Provide a path to a saved the expert samples --expert_samples_path")
        sys.exit(2)

    if args.mode == "sample":
        sample_expert_demonstrations(gym.make(args.env), args.expert_samples_path)
        sys.exit(0)

    if args.mode == "train":
        with open(args.expert_samples_path, "rb") as f:
            # This is a list of `imitation.data.types.Trajectory`, where
            # every instance contains observations and actions for a single expert
            # demonstration.
            transitions = pickle.load(f)
        venv = ut.make_vec_env(args.env, n_envs=1)
        print(len(transitions))
        # transform string representation of network architecture to python array instance
        network_structure = ast.literal_eval(args.network)
        if args.algo == "bc":
            train_BC(
                venv,
                transitions,
                args.training_steps,
                network_structure,
                args.policy_path,
            )
        elif args.algo == "gail":
            train_GAIL(
                venv,
                transitions,
                args.training_steps,
                network_structure,
                args.policy_path,
            )
        else:
            print("Unknown algorithm provided by --algo")
            sys.exit(2)
        pass
    elif args.mode == "test":
        if args.policy_path == "":
            print("Provide a path to a saved policy in parameter --policy_path")
            sys.exit(2)
        env = gym.make(args.env)
        if args.algo == "bc":
            policy_in_action(
                env,
                bc.reconstruct_policy(args.policy_path),
                args.evaluation_path,
                args.render,
            )
        elif args.algo == "gail":
            policy = sb3.PPO.load(f"{args.policy_path}.zip")
            # torch.load(f'{args.policy_path}.zip', device='auto')
            policy_in_action(env, policy, args.evaluation_path, args.render)
