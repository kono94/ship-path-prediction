import torch
from torch.autograd import Variable
import numpy as np


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(param.data * tau)

def hard_update(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def preprocess_state(state):
    # mapping the state values to [-1,1]
    s = np.array(state)
    s[0] = ((state[0] + 1.2) / 1.8) * 2 - 1
    s[1] = ((state[1] + 0.07) / 0.14) * 2 - 1
    return s