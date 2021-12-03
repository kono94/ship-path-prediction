import gym
import numpy as np
from deeprl.envs.curve import CurveEnv
from deeprl.envs.golf import GolfHiddenHoles
from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

env = CurveEnv()
check_env(env)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.4 * np.ones(n_actions))

policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300]))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, seed=2, tau=0.001, learning_rate=0.001, batch_size=128, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=50000, log_interval=1)
model.save("curve_sac_sb")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = TD3.load("curve_sac_sb", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)


obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()