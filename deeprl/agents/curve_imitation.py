import math
import numpy as np
import deeprl.common.util as util
from imitation.data.types import Trajectory
import pickle
import torch as th
from imitation.data import rollout
import tempfile
import pathlib
from deeprl.envs.curve import CurveEnv

import stable_baselines3 as sb3
from imitation.algorithms import bc
from imitation.algorithms.adversarial import gail
from imitation.data import rollout
from imitation.util import logger
import imitation.util.util as ut
from stable_baselines3.common import monitor, policies

import random
import torch
import sys
import time
torch.manual_seed(3)
# most crucial to shuffle deterministically
torch.use_deterministic_algorithms(True)
random.seed(3)
np.random.default_rng(3)

class FeedForward128Policy(policies.ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[1028]*2)

sample_env = CurveEnv()

# bc_policy = None
# with open("bc_policy.pickle", "rb") as f:
#     # This is a list of `imitation.data.types.Trajectory`, where
#     # every instance contains observations and actions for a single expert
#     # demonstration.
#     bc_policy = pickle.load(f)
    
# venv = ut.make_vec_env("curve-v0", n_envs=1, env_make_kwargs={'animation_delay': 0.3})

# obs = venv.reset()
# while True:
#     action, _states = bc_policy.predict(obs, deterministic=True)
#     obs, rewards, dones, info = venv.step(action)
#     venv.render()



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
    trajectory_list.append(Trajectory(np.array(obs), np.array(actions), np.array(infos), terminal=True))

with open('curve_expert_trajectory.pickle', 'wb') as handle:
    pickle.dump(trajectory_list, handle)

with open("curve_expert_trajectory.pickle", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)
venv = ut.make_vec_env("curve-v0", n_envs=1)


print(transitions)

# Train BC on expert data.
bc_trainer = bc.BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=transitions,
    policy = FeedForward128Policy(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                lr_schedule=bc.ConstantLRSchedule(th.finfo(th.float32).max),
            )
)
bc_trainer.train(n_epochs=50000)
bc_policy = bc_trainer.policy
with open('bc_policy.pickle', 'wb') as handle:
    pickle.dump(bc_policy, handle)

obs = venv.reset()
while True:
    action, _states = bc_policy.predict(obs, deterministic=True)
    obs, rewards, dones, info = venv.step(action)
    venv.render()


sys.exit(1)
print(bc_policy.predict(venv.reset()))

#test_policy.rollout(f'{data_prefix}/output/BC/rollouts_{name}{_var_horizon}.pkl')


# Train GAIL on expert data.
# GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
print(transitions)
gail_trainer = gail.GAIL(
    venv=venv,
    demonstrations=transitions,
    demo_batch_size=16,
    gen_algo=sb3.PPO(policies.ActorCriticPolicy, 
                        venv, verbose=1, 
                        batch_size=32, 
                        n_epochs=3,
                        policy_kwargs={'net_arch':[128]*2}),
   #gen_algo=sb3.DDPG("MlpPolicy", venv, verbose=1),
    allow_variable_horizon=True
)

gail_trainer.train(total_timesteps=2100,)
print('Training done')
gail_policy = gail_trainer.policy

obs = venv.reset()
while True:
    action, _states = gail_policy.predict(obs)
    obs, rewards, dones, info = venv.step(action)
    venv.render()
    if dones:
        obs = venv.reset()
        