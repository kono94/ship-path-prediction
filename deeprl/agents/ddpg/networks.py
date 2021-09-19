import torch
import numpy as np
from torch.cuda import init
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fanin_init(size: int, fanin: int = None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorNetwork(nn.Module):
    def __init__(self, nr_of_states: int, nr_of_actions: int, hidden1:int = 400, hidden2:int = 300, init_w: float = 3e-3) -> None:
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(nr_of_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nr_of_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
        
    def init_weights(self, init_w: float) -> None:
        print(init_w)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out
    
    
class CriticNetwork(nn.Module):
    def __init__(self, nr_of_states: int, nr_of_actions: int, hidden1: int = 400, hidden2: int = 300, init_w: float = 3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(nr_of_states, hidden1)
        # Concat action
        self.fc2 = nn.Linear(hidden1 + nr_of_actions, hidden2)
        # Outputs state action value at the end, hence the "1"
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
        
    def init_weights(self, init_w: float) -> None:
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(init_w, init_w)
        
    def forward(self, state_action_pairs):
        states, actions = state_action_pairs
        out = self.fc1(states)
        out = self.relu(out)
        # concating the actions
        out = self.fc2(torch.cat([out, actions], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out