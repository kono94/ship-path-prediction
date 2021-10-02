import numpy as np
from math import sqrt

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, nr_of_actions, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu * np.ones(nr_of_actions)
        self.sigma = sigma * np.ones(nr_of_actions)
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)