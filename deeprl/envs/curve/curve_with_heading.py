import numpy as np
from gym.envs.registration import register
from gym import spaces

from deeprl.envs.curve.curve_base import CurveBase
import deeprl.common.util as util

class CurveWithHeading(CurveBase):
    """Curve environment with a 3-dim state space
    state := (x, y, heading)
    
    and a just one dimensional action space;
    action := (heading)
    
    Speed is constant
    """
    
    def __init__(self, animation_delay=0.1):
        super().__init__(animation_delay=animation_delay)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

    def _step_observation(self):
        return self._normalize_state([self.agent_position.x, self.agent_position.y, self.heading])
    
    def _expert_output(self, last_pos, last_heading, last_speed, expert_heading, expert_speed):
        return self._normalize_state([last_pos.x, last_pos.y, last_heading]), \
               self._normalize_action([expert_heading])
               
    def reset(self):
        super()._reset()
        return self._normalize_state([self.agent_position.x, self.agent_position.y, self.heading])
    
    def reset_deterministically(self, idx):
        super()._reset_deterministically(idx)
        return self._normalize_state([self.agent_position, self.agent_position, self.heading])
    
register(
    id="curve-heading-v0",
    entry_point="deeprl.envs.curve.curve_with_heading:CurveWithHeading",
)           
