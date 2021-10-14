import os

os.environ["TF_CPP_MIN_golf/LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from golf_env import GolfHiddenHoles
from golf_model import get_model, policy, update_target
import json
import time
import random

def reset_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class OrnsteinUhlenbeckNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, latent_dim=64, x_initial=None):
        self.latent_dim = latent_dim
        self.theta = theta
        self.mean = mean * np.ones(latent_dim)
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt +     
             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
        return self.x_prev

class GaussianNoise():
    def __init__(self, latent_dim=64, x_initial=None):
        self.latent_dim = latent_dim
        self.x_initial = x_initial
        self.reset()
    
    def __call__(self):
        x = np.random.normal(size=self.latent_dim)
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros(self.latent_dim)
        return self.x_prev


class MemoryBuffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, num_states=2, num_actions=2):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.time_buffer = np.zeros((self.buffer_capacity, 1))
        self.prev_xy_action_buffer = np.zeros((self.buffer_capacity, num_actions)) 
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype='int32')

    # Takes (t,s,a,r,n) obervation tuple as input as well as realized noise
    def record(self, obs_tuple, episode=None):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.time_buffer[index] = obs_tuple[0]
        self.prev_xy_action_buffer[index] = obs_tuple[1]
        self.action_buffer[index] = obs_tuple[2]
        self.reward_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        indices = np.random.choice(record_range, self.batch_size)

        time = self.time_buffer[indices]
        prev_xy_action = self.prev_xy_action_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        
        return time, prev_xy_action, action, reward

class EarlyStopping(object):

    def __init__(self, threshold=5, patience=1) -> None:
        self.loss_a_prev = np.nan
        self.loss_c_prev = np.nan
        self.threshold = threshold
        self.patience = patience
        self.all_below_thrs = []
        self.now = False
    
    def episodic_losses(self, loss_a, loss_c):
        """ Compute relevant statistics for the episode """
        loss_a = np.mean(loss_a)
        loss_c = np.mean(loss_c)
        # Percentage change of critic gan gap
        d_a = 100 * abs(loss_a - self.loss_a_prev) / abs(self.loss_a_prev)
        d_c = 100 * abs(loss_c- self.loss_c_prev) / abs(self.loss_c_prev)

        msg = lambda z, z_prev: "decreased" if z < z_prev else "increased"
        msg_a = msg(loss_a, self.loss_a_prev)
        msg_c = msg(loss_c, self.loss_c_prev)
        print('=================================================')
        print(f'actor: {loss_a:.6f}, {msg_a} {d_a:.0f}%')
        print(f'critic: {loss_c:.6f}, {msg_c} {d_c:.0f}%')
        print('=================================================')
        self.loss_a_prev = loss_a
        self.loss_c_prev = loss_c
        # Check if need to set the stop signal
        all_chgs = np.asarray([d_a, d_c])
        stop_cond = np.all(all_chgs < self.threshold)
        self.all_below_thrs.append(1 if stop_cond else 0)
        self.now = True if sum(self.all_below_thrs) == self.patience else False

def train(batch_size, num_episodes, num_steps, time_emb_dim, actor_input_units,
          actor_hidden_units, actor_num_hidden, critic_input_units, critic_hidden_units, critic_num_hidden,
          hidden_activation, gamma=0.99, tau=0.9, critic_lr=0.002, actor_lr=0.001, seed=0):

    args = {
        'batch_size': batch_size,
        'num_episodes': num_episodes,
        'num_steps': num_steps,
        'time_emb_dim': time_emb_dim,
        'actor_input_units': actor_input_units,
        'actor_hidden_units': actor_hidden_units,
        'actor_num_hidden': actor_num_hidden,
        'hidden_activation': hidden_activation,
        'gamma': gamma,
        'tau': tau,
        'critic_lr': critic_lr,
        'actor_lr': actor_lr,
        'last_episode': 0
    }
    with open(f'golf/log/ddpg/args_training_seed{seed}.json', 'w') as foo:
        json.dump(args, foo)
        
    env = GolfHiddenHoles(num_episodes=num_episodes, num_steps=num_steps)
    # Generate latent_dim trajectories of OU noise
    ou_noise = OrnsteinUhlenbeckNoise(mean=0.0, std_dev=0.2, dt=env.dt, latent_dim=2)
    # ou_noise = GaussianNoise(latent_dim=latent_dim)
    memory_buffer = MemoryBuffer(buffer_capacity=50000, batch_size=batch_size)
    # Warm up
    memory_buffer = env._warmup(batch_size, buffer=memory_buffer, latent_dim=1)
    # Early stopping
    early_stop = EarlyStopping(patience=3)
    # Get the model
    actor, critic, target_actor, target_critic, c_actor, c_critic = get_model(env, 
                                                                              time_emb_dim,
                                                                              actor_input_units, 
                                                                              actor_hidden_units,
                                                                              actor_num_hidden,
                                                                              critic_input_units, 
                                                                              critic_hidden_units,
                                                                              critic_num_hidden,
                                                                              hidden_activation,
                                                                              gamma,
                                                                              critic_lr, 
                                                                              actor_lr)
    # Initialize target networks
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())

    for ep in range(env.num_episodes):
        # Initial state
        prev_xy_action = env._normalize_state(*env.reset())
        _ = ou_noise.reset()

        reward_hist = []
        loss_a, loss_c = [], []
        
        while True:
            # Sample batch of experiences
            time_batch, prev_xy_action_batch, action_batch, reward_batch  = memory_buffer.sample()

            # Train actor switching fake label to real
            loss_critic = c_critic.train_on_batch([prev_xy_action_batch, action_batch, reward_batch, time_batch], 
                                                    [np.zeros((batch_size,1))])
            
            loss_actor = c_actor.train_on_batch([prev_xy_action_batch, time_batch], 
                                                 [np.zeros((batch_size,1))])
            loss_a.append(loss_actor)
            loss_c.append(loss_critic)

            # Execute the policy based on current state
            step = env.step_counter-1
            prev_xy_action = env._normalize_state(*env.prev_action)
            action = policy(env, actor, prev_xy_action, ou_noise, step)
            # Get reward and next state from the action taken
            _, reward, done, _ = env.step(action)
            # dtw_distance = env._dtw_distance()
            print(f'Step {step+1}/{env.num_steps} in episode {ep+1}/{env.num_episodes}, Reward: {reward:.2f}, maDTW={env.maDTW:.2f}')
            reward_hist.append(reward)
            # Record experience
            memory_buffer.record((step, prev_xy_action, action, reward))
            if done: break

            update_target(target_actor.variables, actor.variables, tau)
            update_target(target_critic.variables, critic.variables, tau)

            # env.render()
            # time.sleep(0.1)

        # env.close()
        # plt.plot(loss_c, label='critic')
        # plt.plot(loss_a, label='actor')
        # plt.legend()
        # plt.show()

        early_stop.episodic_losses(loss_a, loss_c)

        args['last_episode'] = ep+1
        with open(f'golf/log/ddpg/args_training_seed{seed}.json', 'w') as foo:
            json.dump(args, foo)
        actor.save_weights(f'golf/log/ddpg/actor_weights_seed{seed}.h5')
        if early_stop.now: break

    
    return env, ou_noise, actor 

def test_agent(env=None, ou_noise=None, actor=None, num_runs=10):

    # Test in multiple runs, choose best performance
    steps_madtw = np.empty((num_runs,2))
    for k in range(num_runs):
        state = env._normalize_state(*env.reset())
        _ = ou_noise.reset()
        while True:
            step = env.step_counter-1
            action = policy(env, actor, state, ou_noise, step)
            _, reward, done, _ = env.step(action)
            state = env._normalize_state(*env.prev_action)
            print(f'Step: {step},  Reward: {reward:.2f}, maDTW: {env.maDTW:.2f}')
            # env.render()
            # time.sleep(0.1)
            if done: break
        # env.close()
        steps_madtw[k] = [env.step_counter, env.maDTW]
    
    return np.mean(steps_madtw,0)


if __name__ == '__main__':

    model = 'ddpg'
    project_path = f'golf/log/{model}'
    r = pd.DataFrame([], columns=['seed', 'avg_num_steps', 'avgMADTW'])
    r.to_csv(f'{project_path}/test_{model}.csv', index=False)

    for seed in range(10):
        reset_seeds(seed)
        # Train the agent
        env, ou_noise, actor = train(batch_size=64, 
                                    num_episodes=100, 
                                    num_steps=200,
                                    time_emb_dim=64, 
                                    actor_input_units=16,
                                    actor_hidden_units=128, 
                                    actor_num_hidden=4,
                                    critic_input_units=16,
                                    critic_hidden_units=128, 
                                    critic_num_hidden=4,
                                    hidden_activation='tanh',
                                    gamma=0.99, 
                                    tau=0.005,
                                    critic_lr=1e-3,
                                    actor_lr=1e-5,
                                    seed=seed)
        # Test it in multiple runs
        avg_steps, avg_madtw = test_agent(env, ou_noise, actor, num_runs=10)
        r = pd.DataFrame([], columns=['seed', 'avg_num_steps', 'avgMADTW'])
        r.loc[0] = [seed, avg_steps, avg_madtw]
        r.to_csv(f'{project_path}/test_{model}.csv', index=False, header=False, mode='a')
        

   