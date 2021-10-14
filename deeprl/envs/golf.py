import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class GolfHiddenHoles(gym.Env):

    def __init__(self, num_episodes=100, num_steps=100, max_speed=3.5):

        self.num_episodes = num_episodes
        self.num_steps = num_steps
        # Ball of unit mass m, gamma_a sets the speed decay rate.
        self.gamma = 0.9
        self.omega = 0.9
        alpha_min, alpha_max = 5, 10
        self.alphas = np.linspace(alpha_min, alpha_max, self.num_episodes)

        self.T = 2*np.pi / self.omega
        self.dt = self.T / self.num_steps
        self.dangle = np.radians(5)
        
        # Spatial boundaries
        self.y_shift = 0.5
        self.min_x = 0.0
        self.max_x = np.sqrt(alpha_max * self.T)
        self.min_y = 0
        self.max_y = 1.0 + self.y_shift
        # Kinematic boundaries
        self.min_speed = 0.0
        self.max_speed = max_speed
        self.min_angle = -np.pi
        self.max_angle = np.pi
        # History of states
        self.history_state = np.empty(shape=(self.num_steps+1,2))
        self.history_action = np.empty(shape=(self.num_steps+1,2))
        self.gamma = 0.9 # for exponential smoothing of DTW

        # Rendering parameters
        self.screen_width = 600
        self.screen_height = 400
        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        scale_x = self.screen_width / world_width
        scale_y = self.screen_height / world_height
        self.scale = np.array([scale_x, scale_y])
        self.ball_rad = 5

        self.viewer = None
        self.reset()

        # Actions: (speed, angle) to throw the ball
        self.action_space = spaces.Box(
            low = np.array([-1, -1], dtype=np.float32),
            high= np.array([1, 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Observations: (x,y) of hidden hole
        self.observation_space = spaces.Box(
            low = np.array([self.min_x, self.min_y], dtype=np.float32),
            high = np.array([self.max_x, self.max_y], dtype=np.float32),
            dtype = np.float32
        )
    
    def rescale_action(self, action):
        tmp = np.ones_like(action)
        spaces = {'low': [self.min_speed, self.min_angle],
                  'high': [self.max_speed, self.max_angle]}

        for i in range(0, len(action)):
            act_k = (spaces['high'][i] - spaces['low'][i])/ 2. 
            act_b = (spaces['high'][i] + spaces['low'][i])/ 2.  
            tmp[i] = act_k * action[i] + act_b
        return tmp

    def step(self, action):
        # step()'s input is an action that is the output from the agent which is -1 to 1
        # so this needs to get scaled back to the values of the environment
        env_action = self.rescale_action(action)

        t = self.step_counter * self.dt
        self.state = self._gen_state(t)
        self.action = self._polar_to_cartesian(env_action)
        # Storing history of observations and agent decisions
        self.history_state[self.step_counter] = self.state
        self.history_action[self.step_counter] = self.action
        # Computing reward
        self.reward = self._get_reward(self.state, self.action)
        done = self._termination(t, self.action)
        self.prev_state = self.history_state[self.step_counter-1]
        self.prev_action = self.action
        self.step_counter += 1 
        self.maDTW = self.gamma * self.maDTW + (1-self.gamma) * self._dtw_distance()
        return self.state, self.reward, done, {}
    
    def reset(self):
        self.alpha = np.random.choice(self.alphas)
        self.prev_state = np.array([0.0, self.max_y])
        self.prev_action = np.array([0.0, self.max_y])
        self.history_state[0] = self.prev_state
        self.history_action[0] =  self.prev_action
        self.step_counter = 1
        self.maDTW = 0
        return self.prev_state
    
    def _gen_state(self, t):
        x = np.sqrt(self.alpha * t)
        y = self.y_shift + np.cos(self.omega * t) * np.exp(-self.gamma * t)
        return np.array([x, y])
    
    def _polar_to_cartesian(self, action):
        # The distance travelled by the ball during dt is v * dt
        dist_trip = action[0] * self.dt
        return np.array([self.prev_action[0] + np.cos(action[1]) * dist_trip, 
                         self.prev_action[1] + np.sin(action[1]) * dist_trip])
    
    def _cartesian_to_polar(self, prev_state, state):
        delta_state = state - prev_state
        speed =  np.linalg.norm(delta_state) / self.dt
        angle = np.arctan2(delta_state[1], delta_state[0])
        return np.array([speed, angle])
    
    def _get_reward(self, state, action):
        #prev_dist_state_action = np.linalg.norm(self.prev_action - self.prev_state)
        #dist_state_action = np.linalg.norm(action - state)
        #return 1 if (dist_state_action < prev_dist_state_action) else -1

        return (0.01 * self.step_counter **0.5) / (2 - self.maDTW + 1e-6)
        #return 1 if self.maDTW < 2 else -1

    def _dtw_distance(self):
        distance, _ = fastdtw(self.history_state[:self.step_counter], 
                              self.history_action[:self.step_counter], 
                              dist=euclidean)
        charact_dist = np.mean([self.max_x - self.min_x, self.max_y - self.min_y])
        return distance / charact_dist

    def _steer_ini(self):
        x_ini, y_ini = 0.0, self.max_y
        x_end, y_end = self._gen_state(self.T)
        angle = np.arctan2(y_end-y_ini, x_end-x_ini)
        speed = 0.5 * self.max_speed
        return self._normalize_action(speed, angle)
    
    def _normalize_state(self, x, y):
        x_sc = (x - self.min_x) / (self.max_x - self.min_x)
        y_sc = (y - self.min_y) / (self.max_y - self.min_y)
        return np.array([x_sc, y_sc])
    
    def _normalize_action(self, speed, angle):
        speed_sc = speed / self.max_speed
        angle_sc = angle / self.max_angle
        return np.array([speed_sc, angle_sc])

    def _termination(self, t, action):
        x_a, y_a = action
        out_of_x_lim = (x_a > self.max_x) or (x_a < self.min_x)
        out_of_y_lim = (y_a > self.max_y) or (y_a < self.min_y)
        all_time_steps = True if np.isclose(t, self.T) else False
        if out_of_x_lim or out_of_y_lim or all_time_steps:
            done = True
        else:
            done = False
        return done
    
    def _warmup(self, batch_size, buffer=None, latent_dim=None, plot_samples=False):
        self.alpha = np.random.choice(self.alphas)
        step_counter = np.random.choice(self.num_steps, (batch_size,), replace=False)

        if buffer: # Add initial correct steering to the batch
            state_ini = self._normalize_state(0.0, self.max_y)
            buffer.record((0, state_ini, self._steer_ini(), 1))
        
        for k in range(batch_size-1):
            state_prev = self._gen_state(step_counter[k] * self.dt)
            state_now = self._gen_state((step_counter[k] + 1) * self.dt)
            delta_state = state_now - state_prev
            speed_true = np.linalg.norm(delta_state) / self.dt
            angle_true = np.arctan2(delta_state[1], delta_state[0])

            # Randomly sample an action with correct speed
            theta_a = np.random.normal(angle_true, self.dangle)
            unscaled_action = np.array([speed_true, theta_a])
            action = self._normalize_action(*unscaled_action)
            approx_prev_state = delta_state - self._polar_to_cartesian(unscaled_action)
            # Get reward if meaningful action
            reward = 1 if abs(theta_a - angle_true) < self.dangle else -1

            if buffer:
                buffer.record((step_counter[k], approx_prev_state, action, reward))
            if plot_samples:
                color = 'g' if reward == 1 else 'r'
                plt.scatter(*state_prev, color=color)
                plt.scatter(*state_now, color=color)
        
        if plot_samples: plt.show()
        if buffer: return buffer
        
    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            hole = rendering.make_circle(self.ball_rad)
            ball = rendering.make_circle(self.ball_rad)
            ball.set_color(1.,0.,0.)
            self.hole_trans = rendering.Transform(translation=(0, 0))
            self.ball_trans = rendering.Transform(translation=(0, 0))
            hole.add_attr(self.hole_trans)
            ball.add_attr(self.ball_trans)
            self.viewer.add_geom(hole)
            self.viewer.add_geom(ball)

        self.viewer.draw_polyline(self.history_state[:self.step_counter] * self.scale[None,:], 
                                  color=(0, 0, 255), linewidth=2)
        self.viewer.draw_polyline(self.history_action[:self.step_counter] * self.scale[None,:], 
                                  color=(255, 0, 0), linewidth=2)
        self.hole_trans.set_translation(*(self.state * self.scale))
        self.ball_trans.set_translation(*(self.action * self.scale))
        return self.viewer.render(return_rgb_array= mode == 'rgb_array')
            
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def plot_tracks(num_epsisodes=100):
    
    omega = 0.9
    gamma = 0.9
    t = np.linspace(0, 10, 200)

    for alpha in [5, 7.18, 10]:
        x = np.sqrt(alpha * t)
        y = np.cos(omega * x**2/alpha) * np.exp(-gamma*x**2/alpha)
        plt.plot(x,y, label=f'$\\alpha$={alpha}')

    for alpha in np.linspace(5, 10, num_epsisodes):
        x = np.sqrt(alpha * t)
        y = np.cos(omega * x**2/alpha) * np.exp(-gamma*x**2/alpha)
        plt.plot(x,y, color='gray', alpha=0.02)

    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()

class CompareDTW(object):

    def __init__(self) -> None:
        self.omega = 0.9
        self.gamma = 0.9
        alpha_min, alpha_max = 5, 10
        self.alpha_base = 7.5
        self.num_steps = 200
        self.T = 2*np.pi / self.omega
        
        # Spatial boundaries
        self.y_shift = 0.5
        self.min_x = 0.0
        self.max_x = np.sqrt(alpha_max * self.T)
        self.min_y = 0
        self.max_y = 1.0 + self.y_shift

        self.t =  np.linspace(0, 10, self.num_steps)
        
    def gen_track(self, alpha):
        x = np.sqrt(alpha * self.t)
        y = np.cos(self.omega* x**2/alpha) * np.exp(-self.gamma*x**2/alpha)
        return np.stack((x,y), axis=1)

    def distance(self, a1, a2):
        xy_1 = self.gen_track(a1)
        xy_2 = self.gen_track(a2)
        dist, _ = fastdtw(xy_1, xy_2, dist=euclidean)
        charact_dist = np.mean([self.max_x - self.min_x, self.max_y - self.min_y])
        return dist / charact_dist


if __name__ == '__main__':
    plot_tracks(num_epsisodes=100)

    # env = GolfHiddenHoles()
    # for _ in range(env.num_steps_per_episode):
    #     action = env.action_space.sample()
    #     obs, reward, _, _ = env.step(action)
    #     print(f'Reward: {reward}')
    #     env.render()
    #     time.sleep(1)

    # env._warmup(32, plot_samples=True)
    
    cmp = CompareDTW()
    print(cmp.distance(5, 7.18))
    print(cmp.distance(7.18, 10))
    print(cmp.distance(5, 10))