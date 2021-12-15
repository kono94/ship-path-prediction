import numpy as np
import pandas as pd
import torch as th
import time
import pickle
import pathlib
import functools
import os
import tempfile
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from imitation.data.types import Trajectory
from dapi_env import *

import stable_baselines3 as sb3
from gym.wrappers import TimeLimit
from stable_baselines3.common import monitor, policies
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util

num_episodes = 100
timesteps = 200
n_epochs = 1
n_seeds = 10
out_dir = 'output'
data_prefix = 'examples'
# data_prefix = '.'
var_horizon = True
_var_horizon = '_var_horizon' if var_horizon else ''

dapi_paths = {
    'fixed-start': KinkAndWell(timesteps=200, n_paths=500, seed=0),
    'free-start': FreeArmLogistic(timesteps=200, n_paths=500, T=20, seed=0),
    'ribbons': RotateRibbon(timesteps=200, n_paths=500, seed=0),
    'circles': Circles(timesteps=200, n_paths=500, seed=0),
}

def gen_expert_data(name, paths):
    env = MouseHiddenCheese(
        paths=paths,
        num_episodes=num_episodes, 
        timesteps=timesteps, 
        seed=0
    )
    Trajectories = []

    for _ in tqdm(range(num_episodes)):
        states = np.empty((timesteps+1, 2))
        actions = np.empty((timesteps, 2))

        state = env.reset()
        while True:
            step = env.step_counter-1
            # Sample perfect expert's action
            action = env.paths.velocity_at_t(step* env.paths.dt)
            states[step,:] = state
            actions[step,:] = action
            state, _, done, _ = env.step(action)
            if done: break
        
        #     env.render()
        #     time.sleep(0.1)
        # env.close()
        states[-1,:] = state
        Trajectories.append(Trajectory(obs=states, acts=actions, infos=None, terminal=True))

    pickle.dump(Trajectories, open(f'{data_prefix}/expert_demo_{name}.pkl', 'wb'))

def make_vec_env(paths, n_envs=8, seed=0, parallel=False, log_dir=None, var_horizon=False) -> VecEnv:

    def make_env(i, this_seed):
        env = MouseHiddenCheese(
            paths=paths,
            num_episodes=num_episodes, 
            timesteps=timesteps, 
            seed=int(this_seed),
            var_horizon=var_horizon
        )
        env = TimeLimit(env, max_episode_steps=timesteps)

        # Use Monitor to record statistics needed for Baselines algorithms logging
        # Optionally, save to disk
        log_path = None
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, "monitor")
            os.makedirs(log_subdir, exist_ok=True)
            log_path = os.path.join(log_subdir, f"mon{i:03d}")

        env = monitor.Monitor(env, log_path)
        return env
    
    rng = np.random.RandomState(seed)
    env_seeds = rng.randint(0, (1 << 31) - 1, (n_envs,))
    env_fns = [functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)]
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        return SubprocVecEnv(env_fns, start_method="forkserver"), env_seeds
    else:
        return DummyVecEnv(env_fns), env_seeds

class TestPolicy(object):

    def __init__(self, env_paths, name, policy, var_horizon) -> None:
        self.env_paths = env_paths
        self.name = name
        self.policy = policy 
        self.var_horizon = var_horizon
        self.test_envs = []
        all_states = []
        # Test environments have different seeds than during training
        for seed in range(n_seeds):
            env = MouseHiddenCheese(
                num_episodes=num_episodes, 
                timesteps=timesteps, 
                seed=int(seed),
                paths=self.env_paths,
                var_horizon=self.var_horizon
            )
            all_states.append(env.paths.get_positions())
            self.test_envs.append(env)

        self.all_states = np.stack(all_states, axis=0)
    
    def rollout(self, save_path):
        all_xy_actions = np.empty((n_seeds, timesteps, 2))
        all_xy_actions[:,0,:] = self.all_states[:,0,:]
        xy_action = self.all_states[:,0,:]
        
        for step in tqdm(range(1, timesteps)):
            velocity, _ = self.policy.predict(xy_action)
            next_xy_action = self.env_paths.update_position(xy_action, velocity, vectorized=True)
            done = self.termination(step, next_xy_action)
            xy_action = np.where(done[:,None], xy_action, next_xy_action.T)
            all_xy_actions[:,step,:] = xy_action
        
        norm = self.env_paths.dtw_diameter
        self.save_dtw_distances(save_path, all_xy_actions, self.all_states, norm)
    
        pickle.dump([self.all_states, all_xy_actions], open(save_path, 'wb'))
    
    def termination(self, step, xy_action, tol = 1e-6):
        t = step * self.env_paths.dt
        x_a, y_a = xy_action
        out_of_x_lim = (x_a > self.env_paths.x_max + tol) | (x_a < self.env_paths.x_min - tol)
        out_of_y_lim = (y_a > self.env_paths.y_max + tol) | (y_a < self.env_paths.y_min - tol)
        all_time_steps = True if np.isclose(t, self.env_paths.T) else False
        return out_of_x_lim | out_of_y_lim | all_time_steps

    def envs_next_obs(self, velocity):
        next_obs = np.empty((n_seeds, 2))
        for k, env in enumerate(self.test_envs):
            next_obs[k], _, _, _ = env.step(velocity[k])
        return next_obs
    
    def save_dtw_distances(self, path, all_xy_actions, all_states, norm):
        df = pd.DataFrame([], columns=['seed', 'dtw_distance'])
        path = '/'.join(path.split('/')[:-1]) + f'/dtw_distances_{self.name}{_var_horizon}.csv'
        for seed in range(n_seeds):
            df.loc[seed] = [
                seed, 
                fastdtw(all_xy_actions[seed], all_states[seed], dist=euclidean)[0] / norm
            ]
        df.to_csv(path, index=False)

class FeedForward128Policy(policies.ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[128]*4)

# for name, paths in dapi_paths.items():
#     print(f'Generating paths for {name}')
#     gen_expert_data(name, paths)

for name, paths in dapi_paths.items():
    print(f'----------- Processing paths for {name}-----------')

    # Load pickled test demonstrations.
    with open(f'{data_prefix}/expert_demo_{name}.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
    # This is a more general dataclass containing unordered
    # (observation, actions, next_observation) transitions.
    transitions = rollout.flatten_trajectories(trajectories)

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv, env_seeds = make_vec_env(paths, n_envs=n_seeds, var_horizon=var_horizon)
    np.save(f'{data_prefix}/output/env_seeds.npy', env_seeds)


    # Train BC on expert data.
    bc_logger = logger.configure(tempdir_path / "BC/")
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        custom_logger=bc_logger,
        policy = FeedForward128Policy(
                    observation_space=venv.observation_space,
                    action_space=venv.action_space,
                    lr_schedule=bc.ConstantLRSchedule(th.finfo(th.float32).max),
                )
    )
    bc_trainer.train(n_epochs=n_epochs)
    bc_policy = bc_trainer.policy

    test_policy = TestPolicy(paths, name, bc_policy, var_horizon)
    test_policy.rollout(f'{data_prefix}/output/BC/rollouts_{name}{_var_horizon}.pkl')


    # Train GAIL on expert data.
    max_n_transitions = 2 * timesteps * num_episodes * n_epochs

    gail_logger = logger.configure(tempdir_path / "GAIL/")
    gail_trainer = gail.GAIL(
        venv=venv,
        demonstrations=transitions,
        demo_batch_size=64,
        gen_algo=sb3.PPO(policies.ActorCriticPolicy, 
                        venv, verbose=1, 
                        batch_size=64, 
                        n_epochs=2*n_epochs,
                        policy_kwargs={'net_arch':[128]*4}),
        custom_logger=gail_logger,
        allow_variable_horizon=var_horizon
    )
    gail_trainer.train(total_timesteps=max_n_transitions)

    gail_policy = gail_trainer.policy
    test_policy = TestPolicy(paths, name, gail_policy, var_horizon)
    test_policy.rollout(f'{data_prefix}/output/GAIL/rollouts_{name}{_var_horizon}.pkl')


    # Train AIRL on expert data.
    airl_logger = logger.configure(tempdir_path / "AIRL/")
    airl_trainer = airl.AIRL(
        venv=venv,
        demonstrations=transitions,
        demo_batch_size=64,
        gen_algo=sb3.PPO(policies.ActorCriticPolicy, 
                        venv, verbose=1,  
                        batch_size=64, 
                        n_epochs=2*n_epochs,
                        policy_kwargs={'net_arch':[128]*4}),
        custom_logger=airl_logger,
        allow_variable_horizon=var_horizon
    )
    airl_trainer.train(total_timesteps=max_n_transitions)

    airl_policy = airl_trainer.policy
    test_policy = TestPolicy(paths, name, airl_policy, var_horizon)
    test_policy.rollout(f'{data_prefix}/output/AIRL/rollouts_{name}{_var_horizon}.pkl')