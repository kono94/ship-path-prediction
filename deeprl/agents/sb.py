import gym
import numpy as np
from deeprl.envs.curve import CurveEnv
from deeprl.envs.golf import GolfHiddenHoles
from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import random
import torch
import time

SEED = 4
torch.manual_seed(SEED)
# most crucial to shuffle deterministically
torch.use_deterministic_algorithms(True)
random.seed(SEED)
np.random.default_rng(SEED)
        
#env = DummyVecEnv([lambda: gym.make("curve-v0")])
#env = VecNormalize(env, norm_obs=True, norm_reward=False)
env = CurveEnv()

model = TD3.load("curve_DDPG_300k400300", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward)
print(std_reward)
time.sleep(3)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
        
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

policy_kwargs = dict(net_arch=dict(pi=[400,300], qf=[400,300]))

model = DDPG("MlpPolicy", env, action_noise=action_noise, buffer_size= 10000, verbose=1, seed=SEED, gamma=0.99, tau=1e-3, learning_rate=1e-4, batch_size=128, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=300000, log_interval=1)
model.save("curve_DDPG_sb700500")
env = model.get_env()

#del model # remove to demonstrate saving and loading

#model = TD3.load("curve_sac_sb", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward)
print(std_reward)
time.sleep(3)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()