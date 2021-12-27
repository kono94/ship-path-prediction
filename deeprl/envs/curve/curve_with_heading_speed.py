import numpy as np
from gym.envs.registration import register
from gym import spaces

from deeprl.envs.curve.curve_base import CurveBase


class CurveWithHeadingAndSpeed(CurveBase):
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
        
    def _step_observation(self):
        return self._normalize_state([self.agent_position.x, self.agent_position.y, self.agent_heading, self.agent_speed])
    
    def _expert_output(self, last_pos, heading, speed):
        return self._normalize_state([last_pos.x, last_pos.y, heading, speed]), \
               self._normalize_action([heading, speed])
      
    def reset(self):
        super()._reset()
        return self._normalize_state([self.last_position.x, self.last_position.y, self.last_heading, self.last_speed])
    
    def reset_deterministically(self, idx):
        super()._reset_deterministically(idx)
        return self._normalize_state([self.last_position.x, self.last_position.y, self.last_heading, self.last_speed])


register(
    id="curve-heading-speed-v0",
    entry_point="deeprl.envs.curve.curve_with_heading_speed:CurveWithHeadingAndSpeed",
)
