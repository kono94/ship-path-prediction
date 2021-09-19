import gc
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

#actor = networks.ActorNetwork()
#critic = networks.CriticNetwork()


class DDPG(object):
        def __init__(self, nr_of_states: int, nr_of_actions: int, args) -> None:
            super().__init__()
                        
            # TODO test results of dropping this
            if args.seed > 0:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                if util.USE_CUDA:
                    torch.cuda.manual_seed(args.seed)
                
            self.nr_of_states = nr_of_states
            self.nr_of_actions = nr_of_actions
            
            self.criterion = nn.MSELoss()
            
            net_cfg = {
                'hidden1': args.hidden1,
                'hidden2': args.hidden2,
                'init_w': args.init_w
            }
            
            self.actor = networks.ActorNetwork(self.nr_of_states, self.nr_of_actions, **net_cfg)
            self.actor_target = networks.ActorNetwork(self.nr_of_states, self.nr_of_actions, **net_cfg)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr_rate)
            # Freeze target networks with respect to optimizers
            for p in self.actor_target.parameters():
                p.requires_grad = False
        
            self.critic = networks.CriticNetwork(self.nr_of_states, self.nr_of_actions, **net_cfg)
            self.critic_target = networks.CriticNetwork(self.nr_of_states, self.nr_of_actions, **net_cfg)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=args.critic_lr_rate)
            # Freeze target networks with respect to optimizers
            for p in self.critic_target.parameters():
                p.requires_grad = False
                
                
            # Copy weights
            util.hard_update(self.actor_target, self.actor)
            util.hard_update(self.critic_target, self.critic)
            
            self.memory = ReplayBuffer(args.replay_max_size, nr_of_states, nr_of_actions)
            self.random_process = noise.OrnsteinUhlenbeckActionNoise(args.mu * np.ones(nr_of_actions), args.sigma * np.ones(nr_of_actions), args.theta)

            self.batch_size = args.batch_size
            self.target_update_rate = args.target_update_rate
            self.discount = args.discount
            self.depsilon = 1.0 / args.epsilon
            
            self.epsilon = 1.0
            self.is_training = True
            
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            if util.USE_CUDA: 
                self.actor.cuda()
                self.actor_target.cuda()
                self.critic.cuda()
                self.critic_target.cuda()
                
                
        def update_policy(self):
            state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample(self.batch_size)
            
            state_batch      = torch.from_numpy(state_batch).float().to(self.device)
            action_batch     = torch.from_numpy(action_batch).float().to(self.device)
            reward_batch     = torch.from_numpy(reward_batch).float().to(self.device)
            next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
            terminal_batch   = torch.from_numpy(terminal_batch).float().to(self.device)
            
            # Bellman backup for Q function
            # Get Q values by passing s,a into the target networks
            with torch.no_grad():
                target_actions = self.actor_target(next_state_batch)
                next_q_values = self.critic_target([next_state_batch, target_actions])
            
            # r + gamma * Q, if not terminal
            target_q_batch = reward_batch + self.discount * next_q_values * terminal_batch
            
            
            # critic update
            self.critic.zero_grad()
            q_batch = self.critic([state_batch, action_batch])
            value_loss = self.criterion(q_batch, target_q_batch)
            value_loss.backward()
            self.critic_optimizer.step()
            
            # [optional] freeze Q-network parameters
            # actor update
            self.actor.zero_grad()
            policy_loss = -self.critic([state_batch, self.actor(state_batch)])
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            # target update (copy weights with tau degree); FROM ACTOR to ACTOR_TARGET
            with torch.no_grad():
                util.soft_update(self.actor_target, self.actor, self.target_update_rate)
                util.soft_update(self.critic_target, self.critic, self.target_update_rate)

        
        def eval(self):
            self.actor.eval()
            self.actor_target.eval()
            self.critic.eval()
            self.critic_target.eval()
        
        
        def random_action(self):
            return np.random.uniform(-1.,1.,self.nr_of_actions)
        
        
        def select_action(self, s_t, decay_epsilon=True):
            action = self.actor(torch.as_tensor(s_t, dtype=torch.float32)).detach().numpy()
            action += self.is_training * max(self.epsilon, 0) * self.random_process.noise()
            action = np.clip(action, -1., 1.)
            if decay_epsilon:
                self.epsilon -= self.depsilon
            return action
        
        
        def remember(self, current_state, action, reward, next_state, done):
            self.memory.store_transition(current_state, action, reward, next_state, done)
            
        def reset(self):
            self.random_process.reset()
            
            
        def save_model(self,output):
            torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
            torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(output))
            
            
        def load_weights(self, output):
            if output is None: return
            self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
            self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))