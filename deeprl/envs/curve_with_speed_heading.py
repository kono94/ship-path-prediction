import numpy as np
from gym.envs.registration import register
from gym import spaces

from deeprl.envs.curve_base import CurveBase
import deeprl.common.util as util


class CurveWithSpeedAndHeading(CurveBase):
    """Advanced curve environment with 4-dimensional state space
    state := (x, y, heading, speed)
    
    and 2-dim action
    action := (heading, speed)
    
    Speed in constantly increasing over time during an episode
    """
    def __init__(self, animation_delay=0.1):
        super().__init__(animation_delay=animation_delay)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

    def _normalize_action(self, action):
        norm_angle = util.lmap(action[0], [self.min_angle, self.max_angle], [-1, 1])
        norm_speed = util.lmap(action[1], [self.min_speed, self.max_speed], [-1, 1])
        return np.array([norm_angle, norm_speed])

    def _denormalize_action(self, action):
        original_angle = util.lmap(action[0], [-1, 1], [self.min_angle, self.max_angle])
        original_speed = util.lmap(action[1], [-1, 1], [self.min_speed, self.max_speed])
        return np.array([original_angle, original_speed], dtype=np.float32)

    def _normalize_state(self, state):
        norm_x = util.lmap(state[0], [0, self.width], [-1, 1])
        norm_y = util.lmap(state[1], [0, self.height], [-1, 1])
        norm_speed = util.lmap(state[2], [self.min_speed, self.max_speed], [-1, 1])
        return np.array([norm_x, norm_y, norm_speed], dtype=np.float32)

    def _denormalize_state(self, state):
        original_x = util.lmap(state[0], [-1, 1], [0, self.width])
        original_y = util.lmap(state[1], [-1, 1], [0, self.height])
        original_speed = util.lmap(state[2], [-1, 1], [self.min_speed, self.max_speed])
        return np.array([original_x, original_y, original_speed], dtype=np.float32)

    def reset(self):
        super()._reset()
        return self._normalize_state([self.last_position.x, self.last_position.y, self.last_heading, self.last_speed])
    
    def reset_deterministically(self, idx):
        super()._reset_deterministically(idx)
        return self._normalize_state([self.last_position.x, self.last_position.y, self.last_heading, self.last_speed])


register(
    id="curve-v1",
    entry_point="deeprl.envs.curve_with_speed_heading:CurveWithSpeedAndHeading",
)
