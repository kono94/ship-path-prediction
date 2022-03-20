from gym.envs.registration import register
from deeprl.envs.curve.curve_s4 import CurveWithHeadingSpeedDistance

class CurveWithHeadingSpeedTimestep(CurveWithHeadingSpeedDistance):
    """Advanced curve environment with 5-dimensional state space
    state := (x, y, heading, speed, timestep)
    
    and 2-dim action
    action := (heading, speed)
    
    Speed in constantly increasing over time during an episode
    """
    def __init__(self, animation_delay=0.1):
        super().__init__(animation_delay=animation_delay)
        
    def _step_observation(self):
        #return self._normalize_state([self.position.x, self.position.y, self.heading, self.speed, self.step_count])
        return self._normalize_state([self.agent_position.x, self.agent_position.y, self.agent_heading, self.agent_speed, self.step_count])
    
    def _expert_output(self, last_pos, last_heading, last_speed, expert_heading, expert_speed):
          return self._normalize_state([last_pos.x, last_pos.y, last_heading, last_speed, self.step_count]), \
               self._normalize_action([expert_heading, expert_speed])


register(
    id="curve-heading-speed-timestep-v0",
    entry_point="deeprl.envs.curve.curve_s5:CurveWithHeadingSpeedTimestep",
)
