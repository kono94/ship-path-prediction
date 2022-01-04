import numpy as np
from gym.envs.registration import register
from gym import spaces

from deeprl.envs.curve.curve_base import CurveBase
import deeprl.common.util as util

class CurveSimpleState(CurveBase):
    """Simple curve environment with a 2-dim state space
    state := (x, y)
    
    and a just one dimensional action space
    action := (heading)
    
    """
    
    def __init__(self, animation_delay=0.1):
        super().__init__(animation_delay=animation_delay)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

    def _step_observation(self):
        return self._normalize_state([self.agent_position.x, self.agent_position.y])
    
    def _expert_output(self, last_pos, heading, speed):
        return self._normalize_state([last_pos.x, last_pos.y]), \
               self._normalize_action([heading])
               
    def reset(self):
        super()._reset()
        return self._normalize_state([self.agent_position.x, self.agent_position.y])
    
    def reset_deterministically(self, idx):
        super()._reset_deterministically(idx)
        return self._normalize_state([self.agent_position.x, self.agent_position.y])
    

    
register(
    id="curve-simple-v0",
    entry_point="deeprl.envs.curve.curve_simple:CurveSimpleState",
)           
