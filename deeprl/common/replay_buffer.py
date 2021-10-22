from typing import Tuple
import numpy as np

class ReplayBuffer(object):
    '''
    Simple ReplayBuffer implementation that just stores transitions
    and samples them randomly.
    '''
    def __init__(self, max_size: int, state_shape: Tuple, action_shape: Tuple) -> None:
        super().__init__()
        self.max_size = max_size
        self.mem_pos = 0
        self.state_memory        = np.zeros((self.max_size,) + (state_shape,))
        self.action_memory       = np.zeros((self.max_size,) + (action_shape,))
        self.reward_memory       = np.zeros((self.max_size,) + (1,))
        self.next_state_memory   = np.zeros((self.max_size,) + (state_shape,))
        self.terminal_memory     = np.zeros((self.max_size,) + (1,))
        
        
    
    def store_transition(self, state, action, reward, next_state, terminal) -> None:
        idx = self.mem_pos % self.max_size
        
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.terminal_memory[idx] = terminal
        
        self.mem_pos += 1
        
        
    def sample(self, batch_size: int):
        lower_bound = min(self.mem_pos, self.max_size)
        
        batch_idx = np.random.choice(lower_bound, batch_size, replace=False)
        
        states      = self.state_memory[batch_idx]
        actions     = self.action_memory[batch_idx]
        rewards     = self.reward_memory[batch_idx]
        next_states = self.next_state_memory[batch_idx]
        dones       = self.terminal_memory[batch_idx]
        
        #print(f'state: {states[0]} action: {actions[0]} reward: {rewards[0]} next_state: {next_states[0]}' )
        return states, actions, rewards, next_states, dones
