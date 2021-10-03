import logging
import logging.config
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

import networks
import deeprl.common.noise as noise
import deeprl.common.util as util
from deeprl.common.replay_buffer import ReplayBuffer

logging.config.fileConfig('logger.conf')
logger = logging.getLogger('ddpg')


class DDPG(object):

    def __init__(self, nr_of_states: int, nr_of_actions: int, args) -> None:
        super().__init__()

        self.nr_of_states = nr_of_states
        self.nr_of_actions = nr_of_actions
        self.batch_size = args.batch_size
        self.tau = args.tau # target update rate
        self.gamma = args.gamma # discount factor
        self.memory = ReplayBuffer(args.replay_max_size, nr_of_states, nr_of_actions)
        self.random_process = noise.OrnsteinUhlenbeckActionNoise(nr_of_actions, args.mu, args.sigma, args.theta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = networks.ActorNetwork(self.nr_of_states, self.nr_of_actions, **net_cfg)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr_rate)
        self.actor_target = networks.ActorNetwork(self.nr_of_states, self.nr_of_actions, **net_cfg)
        # Freeze target networks with respect to optimizers
        for p in self.actor_target.parameters():
            p.requires_grad = False
    
        self.critic = networks.CriticNetwork(self.nr_of_states, self.nr_of_actions, **net_cfg)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.critic_lr_rate)
        self.critic_target = networks.CriticNetwork(self.nr_of_states, self.nr_of_actions, **net_cfg)
        # Freeze target networks with respect to optimizers
        for p in self.critic_target.parameters():
            p.requires_grad = False
            
        self.nets_to_cuda()

        # Copy weights
        util.hard_update(self.actor, self.actor_target)
        util.hard_update(self.critic, self.critic_target)
    
            
    
    def update_policy(self):
        states, actions, rewards, next_states, dones = self.sample_train_batches()
        
        target_Q = self.calculate_target_Q(next_states, rewards, dones)

        self.update_critic(states, actions, target_Q)
        self.update_actor(states)

        self.update_targets_by_soft_copy()

         
    def sample_train_batches(self):
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample(self.batch_size)
        
        state_batch      = torch.from_numpy(state_batch).float().to(self.device)
        action_batch     = torch.from_numpy(action_batch).float().to(self.device)
        reward_batch     = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        terminal_batch   = torch.from_numpy(terminal_batch).float().to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def calculate_target_Q(self, next_state_batch, reward_batch, terminal_batch):
        # Bellman backup for Q function
        # Get Q values by passing s,a into the target networks
        with torch.no_grad():
            target_actions = self.actor_target(next_state_batch)
            next_q_values = self.critic_target([next_state_batch, target_actions])
        
        # r + gamma * Q, if not terminal
        return reward_batch + self.gamma * next_q_values * (1 - terminal_batch)

    def update_critic(self, states, actions, target_Q):
        # critic update
        self.critic.zero_grad()
        Q = self.critic([states, actions])
        value_loss = self.criterion(Q, target_Q)
        value_loss.backward()
        self.critic_optimizer.step() 

    def update_actor(self, states):
         # [optional] freeze Q-network parameters
        # actor update
        self.actor.zero_grad()
        policy_loss = -self.critic([states, self.actor(states)])
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

    def update_targets_by_soft_copy(self):
        # target update (copy weights with tau degree); FROM ACTOR to ACTOR_TARGET
        with torch.no_grad():
            util.soft_update(self.actor, self.actor_target, self.tau)
            util.soft_update(self.critic, self.critic_target, self.tau)
    
    def random_action(self):
        return np.random.uniform(-1.,1.,self.nr_of_actions)
    
    def select_action(self, s_t, pure=False):
        action = self.actor(torch.as_tensor(s_t, dtype=torch.float32)).detach().numpy()
        if not pure:
            action += self.random_process.noise()
            action = np.clip(action, -1., 1.)

        return action
    
    
    def remember(self, current_state, action, reward, next_state, done):
        self.memory.store_transition(current_state, action, reward, next_state, done)
   

    def reset(self):
        self.random_process.reset()
    
    def nets_to_cuda(self):
        if util.USE_CUDA: 
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()  

    def save_model(self,output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(output))
        
        
    def load_weights(self, output):
        if output is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))