import gym
import numpy as np
from deeprl.envs.curve import CurveEnv
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env

env = CurveEnv()
check_env(env)


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.8 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=50000, log_interval=1)
model.save("curve_sb")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("curve_sb")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()