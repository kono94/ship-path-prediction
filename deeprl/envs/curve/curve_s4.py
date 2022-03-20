import numpy as np
from gym.envs.registration import register
from gym import spaces

from deeprl.envs.curve.curve_base import CurveBase


class CurveWithHeadingSpeedDistance(CurveBase):
    """Advanced curve environment with 5-dimensional state space
    state := (x, y, heading, speed, distance)
    
    and 2-dim action
    action := (heading, speed)
    
    Speed in constantly increasing over time during an episode
    """
    def __init__(self, animation_delay=0.1):
        super().__init__(animation_delay=animation_delay)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32
        )
        
    def _step_observation(self):
        return self._normalize_state([self.agent_position.x, self.agent_position.y, self.agent_heading, self.agent_speed, self.distance])
    
    def _expert_output(self, last_pos, last_heading, last_speed, expert_heading, expert_speed):
        raise RuntimeError("S4 does not allow for expert sampling")
    
    def _next_agent_speed(self):
        return self.current_action[1]
    
    def reset(self):
        super()._reset()
        return self._normalize_state([self.agent_position.x, self.agent_position.y, self.heading, self.speed, 0])
    
    def reset_deterministically(self, idx):
        super()._reset_deterministically(idx)
        return self._normalize_state([self.agent_position.x, self.agent_position.y, self.heading, self.speed,0])


register(
    id="curve-heading-speed-distance-v0",
    entry_point="deeprl.envs.curve.curve_s4:CurveWithHeadingSpeedDistance",
)
