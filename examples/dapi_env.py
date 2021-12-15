import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from numpy.testing._private.utils import assert_equal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def gen_seeds(num_seeds, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, (1 << 31) - 1, (num_seeds,))

class Paths(object):

    def __init__(self, timesteps=1, n_paths=1, T=None, seed=None, name=None) -> None:
        self.timesteps = timesteps
        self.n_paths = n_paths
        self.name = name
        self.np_random, _ = seeding.np_random(seed)

        if T is not None: # Otherwise call self.set_T(expr)
            self.T = T
            self.dt = T / timesteps

    def set_shape_parameters(self, is_greek_symb=False, **kwargs):
        # Search for variable parameter defining family
        # This should be var_name, var_name_min, var_name_max
        var_name_min = [k for k in kwargs if ('min' in k)][0]
        var_name_max = [k for k in kwargs if ('max' in k)][0]
        var_name = '_'.join(var_name_min.split('_')[:-1])
        assert (var_name in kwargs) and (var_name == '_'.join(var_name_max.split('_')[:-1]))
        self.var_name = var_name
        # Assign parameters as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Define grid of values for var_name
        self.var_values = np.linspace(getattr(self, var_name_min), 
                                      getattr(self, var_name_max), 
                                      self.n_paths)
        self.is_greek_sym = is_greek_symb
    
    def set_T(self, expr):
        self.T = eval(expr)
        self.dt = self.T / self.timesteps
    
    def define_family(self, x, y):
        self.x = x
        self.y = y

    def select_random(self):
        var_value = self.np_random.choice(self.var_values)
        setattr(self, self.var_name, var_value)

    def position_at_t(self, t):
        x = self.x(t)
        y = self.y(t)
        return np.array([x, y])
    
    def velocity_at_t(self, t):
        delta_pos = self.position_at_t(t + self.dt) - self.position_at_t(t)
        speed = np.linalg.norm(delta_pos, axis=0) / self.dt
        angle = np.arctan2(delta_pos[1], delta_pos[0])
        return np.array([speed, angle])
    
    def update_position(self, current_position, velocity, vectorized=False):
        if vectorized:
            speed, angle = velocity[:,0], velocity[:,1]
            current_x, current_y = current_position[:,0], current_position[:,1]
        else:
            speed, angle = velocity
            current_x, current_y = current_position[0], current_position[1]

        dist_travel = speed * self.dt
        return np.array([current_x + np.cos(angle) * dist_travel, 
                         current_y + np.sin(angle) * dist_travel])
    
    def reflect_on_boundary(self, x, y, backend=np):
        # Specular reflection on boundaries (when needed after position updates)
        x = backend.where(x > self.x_max, 2 * self.x_max - x, x)
        x = backend.where(x < self.x_min, 2 * self.x_min - x, x)
        y = backend.where(y > self.y_max, 2 * self.y_max - y, y)
        y = backend.where(y < self.y_min, 2 * self.y_min - y, y)
        return x, y

    def get_positions(self):
        return self.position_at_t(np.linspace(0, self.T, self.timesteps)).T
    
    def get_velocities(self):
        return self.velocity_at_t(np.linspace(0, self.T, self.timesteps)).T

    def set_boundaries(self, automatic=True, var_winded=False, **kwargs):
        if automatic:
            all_paths = np.empty((self.timesteps, 2, len(self.var_values)))
            all_speeds = np.empty((self.timesteps, 2, len(self.var_values)))
            for k, value in enumerate(self.var_values):
                setattr(self, self.var_name, value)
                all_paths[...,k] = self.get_positions()
                all_speeds[...,k] = self.get_velocities()
            self.x_min, self.y_min = np.min(all_paths, (0,2))
            self.x_max, self.y_max = np.max(all_paths, (0,2))
            self.s_min, self.a_min = np.min(all_speeds, (0,2))
            self.s_max, self.a_max = np.max(all_speeds, (0,2))
            # Compute "diameter" of the family = DTW of boundary curves
            setattr(self, self.var_name, np.min(self.var_values))
            all_xy_min = self.get_positions()
            if var_winded:
                far_end = np.median(self.var_values)  
                coeff = 2.0
            else:
                far_end = np.max(self.var_values)
                coeff = 1.0
            setattr(self, self.var_name, far_end)
            all_xy_mid_or_max = self.get_positions()
            self.dtw_diameter = coeff * fastdtw(all_xy_min, all_xy_mid_or_max, dist=euclidean)[0]
            assert_msg = "Try changing the limits of the parameters so that path_min != path_max"
            assert not np.isclose(self.dtw_diameter, 0), assert_msg
        else:
            self.x_min, self.x_max = kwargs['left'], kwargs['right']
            self.y_min, self.y_max = kwargs['bottom'], kwargs['top']
            self.s_min, self.s_max = kwargs['s_min'], kwargs['s_max']

        # Reset boundaries for angles as [-pi, pi]
        self.a_min, self.a_max = -np.pi, np.pi
    
    def scale_position(self, xy):
        xy_sc = xy - np.asarray([self.x_min, self.y_min])
        xy_sc /= np.asarray([self.x_max - self.x_min, self.y_max - self.y_min])
        return xy_sc
    
    def scale_velocity(self, ve):
        speed, angle = ve
        speed_sc = (speed - self.s_min) / (self.s_max - self.s_min)
        angle_sc = angle / self.a_max
        return np.array([speed_sc, angle_sc])
    
    def unscale_velocity(self, uve):
        speed_sc, angle_sc = uve
        speed = self.s_min + np.clip(speed_sc, 0, 1) * (self.s_max - self.s_min)
        angle = angle_sc * self.a_max
        return np.array([speed, angle])
    
    def dtw_distance_from_paths(self, ref_path, test_path):
        distance = fastdtw(ref_path, test_path, dist=euclidean)[0]
        return distance / self.dtw_diameter
    
    def dtw_distance_from_params(self, param1, param2):
        setattr(self, self.var_name, param1)
        path1 = self.get_positions()
        setattr(self, self.var_name, param2)
        path2 = self.get_positions()
        distance = fastdtw(path1, path2, dist=euclidean)[0]
        return distance
    
    def plot_family(self, ax=None, vars=None):
        var_min = getattr(self, f'{self.var_name}_min')
        var_max = getattr(self, f'{self.var_name}_max')
        var_mid = 0.5 * (var_min + var_max)
        if vars: var_min, var_mid, var_max = vars
        var_name = f'$\\{self.var_name}$' if self.is_greek_sym else f'${self.var_name}$'

        for var_value in [var_min, var_mid, var_max]:
            setattr(self, self.var_name, var_value)
            xy = self.get_positions()
            if ax:
                ax.plot(xy[:,0], xy[:,1], label=f'{var_name}={var_value:.2f}')
            else:
                plt.plot(xy[:,0], xy[:,1], label=f'{var_name}={var_value:.2f}')

        for var_value in self.var_values:
            setattr(self, self.var_name, var_value)
            xy = self.get_positions()
            if ax:
                ax.plot(xy[:,0], xy[:,1], color='gray', alpha=0.03)
            else:
                plt.plot(xy[:,0], xy[:,1], color='gray', alpha=0.03)

        if ax is None:
            plt.legend()
            plt.xlabel(r'$x$', fontsize=13)
            plt.ylabel(r'$y$', fontsize=13)
            plt.show()
        else:
            ax.set_xlabel(r'$x$', fontsize=13)
            ax.set_ylabel(r'$y$', fontsize=13)
            ax.tick_params(axis='both', labelsize=13)

class Circles(Paths):

    def __init__(self, timesteps=1, n_paths=1, T=None, seed=None, R_min=0.5, R_max=1) -> None:
        super().__init__(timesteps=timesteps, n_paths=n_paths, T=T, seed=seed, name='circles')

        self.set_shape_parameters(R=1.0, omega=0.4, R_min=R_min, R_max=R_max)
        self.set_T('2 *np.pi / self.omega')
        self.define_family(
            x = lambda t: self.R * np.cos(self.omega*t), 
            y = lambda t: self.R * np.sin(self.omega*t)
        )
        self.set_boundaries()

class FreeArmLogistic(Paths):

    def __init__(self, timesteps=1, n_paths=1, T=None, seed=None, y_a_min=0.8, y_a_max=1.0) -> None:
        super().__init__(timesteps=timesteps, n_paths=n_paths, T=T, seed=seed, name='free-start')

        self.set_shape_parameters(y_a=1.0, y_a_min=y_a_min, y_a_max=y_a_max)
        self.define_family(
            x = lambda t: t-6, 
            y = lambda t: self.y_a / (1 + np.exp(t-6))
        )
        self.set_boundaries()

class RotateRibbon(Paths):

    def __init__(self, timesteps=1, n_paths=1, T=None, seed=None, R1=1.0, R2=2.0) -> None:
        super().__init__(timesteps=timesteps, n_paths=n_paths, T=T, seed=seed, name='ribbons')
    
        self.set_shape_parameters(alpha=1.0, R1=R1, R2=R2, omega=0.4, alpha_min=-np.pi, alpha_max=np.pi, is_greek_symb=True)
        self.set_T('2 *np.pi / self.omega')
        self.define_family(
            x = lambda t: (self.R1-self.R2*np.cos(self.omega*t/4)) * np.cos(self.omega*t + self.alpha), 
            y = lambda t: (self.R1-self.R2*np.cos(self.omega*t/4)) * np.sin(self.omega*t + self.alpha)
        )
        self.set_boundaries(var_winded=True)

class KinkAndWell(Paths):

    def __init__(self, timesteps=1, n_paths=1, T=None, seed=None) -> None:
        super().__init__(timesteps=timesteps, n_paths=n_paths, T=T, seed=seed, name='fixed-start')

        self.set_shape_parameters(omega=0.9, gamma=0.9, alpha=3, alpha_min=5, alpha_max=10, is_greek_symb=True)
        self.set_T('2 *np.pi / self.omega')
        self.define_family(x=lambda t: np.sqrt(self.alpha * t), 
                        y=lambda t: np.cos(self.omega * t) * np.exp(-self.gamma * t))
        self.set_boundaries()

def built_in_paths(timesteps, num_paths, seed=None):
    registry = {
        'fixed-start': KinkAndWell(timesteps=timesteps, n_paths=num_paths, seed=seed),
        'free-start': FreeArmLogistic(timesteps=timesteps, n_paths=num_paths, T=20, seed=seed),
        'ribbons': RotateRibbon(timesteps=timesteps, n_paths=num_paths, seed=seed),
        'circles': Circles(timesteps=timesteps, n_paths=num_paths, seed=seed),
    }
    return registry

def plot_families():

    fig, axs = plt.subplots(1,2,figsize=(6.4,3.4))
    KinkAndWell(timesteps=200, n_paths=100, seed=0).plot_family(ax=axs[0])
    axs[0].set_title('FixedEnds', fontsize=13)
    FreeArmLogistic(timesteps=200, n_paths=100, T=20, seed=0).plot_family(ax=axs[1])
    axs[1].set_title('FreeEnd', fontsize=13)
    fig.tight_layout() 
    plt.savefig('/home/sola_ed/Documents/paper_dapi/family_kw.pdf')
    plt.show()

    fig, axs = plt.subplots(1,2)
    Circles(timesteps=200, n_paths=100, seed=0).plot_family(ax=axs[0])
    axs[0].set_title('Circles', fontsize=13)
    axs[0].set_aspect('equal', 'box')
    RotateRibbon(timesteps=200, n_paths=100, seed=0).plot_family(ax=axs[1], vars=[0, np.pi/2, np.pi])
    axs[1].set_title('Ribbons', fontsize=13)
    axs[1].set_aspect('equal', 'box') 
    fig.tight_layout() 
    plt.savefig('/home/sola_ed/Documents/paper_dapi/family_cr.pdf')
    plt.show()

class MouseHiddenCheese(gym.Env):

    def __init__(
        self, 
        num_episodes=100, 
        timesteps=200, 
        eps=0.01,
        dtw_smoothing=0.9,
        need_unscaling=False, 
        paths=None, 
        seed=None, 
        var_horizon=False,
        reflective=False,
    ):
        self.num_episodes = num_episodes
        self.timesteps = timesteps
        self.need_unscaling = need_unscaling
        self.var_horizon = var_horizon
        self.reflective = reflective

        default_paths = KinkAndWell(
            timesteps=timesteps, 
            n_paths=500, 
            seed=seed,
        )
        self.paths = paths if paths else default_paths
        self.eps = eps
        self.smoothing = dtw_smoothing
        
        # Actions: (speed, angle) of the mouse seeking the cheese
        self.action_space = spaces.Box(
            low = np.array([self.paths.s_min, self.paths.a_min], dtype=np.float32),
            high= np.array([self.paths.s_max, self.paths.a_max], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Observations: (x,y) of the cheese after mouse moves
        self.observation_space = spaces.Box(
            low = np.array([self.paths.x_min, self.paths.y_min], dtype=np.float32),
            high= np.array([self.paths.x_max, self.paths.y_max], dtype=np.float32),
            dtype = np.float32
        )

        # History of states
        self.history_state = np.empty(shape=(self.timesteps+1,2))
        self.history_xy_action = np.empty(shape=(self.timesteps+1,2))

        # Rendering parameters
        self.screen_width = 600
        self.screen_height = 400
        world_width = self.paths.x_max - self.paths.x_min
        world_height = self.paths.y_max - self.paths.y_min
        scale_x = self.screen_width / world_width
        scale_y = self.screen_height / world_height
        self.scale = np.array([scale_x, scale_y])
        self.origin = np.array([-self.paths.x_min * scale_x, -self.paths.y_min * scale_y])

        self.viewer = None
        self.current_seed = seed
        self.seed(seed)
        self.reset()
    
    def step(self, action):
        t = self.step_counter * self.paths.dt
        self.state = self.paths.position_at_t(t)
        if self.need_unscaling: 
            velocity = self.paths.unscale_velocity(action)
        else:
            velocity = action
        self.xy_action = self.paths.update_position(self.prev_xy_action, velocity)
        # Check if xy_action is out of boundaries of observation space
        done = self._is_the_end(t)
        # Storing history of observations and agent decisions
        self.history_state[self.step_counter] = self.state
        self.history_xy_action[self.step_counter] = self.xy_action
        # Computing the dtw distance betwee observation and prediction
        dtw_dist = self.paths.dtw_distance_from_paths(
            self.history_state[:self.step_counter], 
            self.history_xy_action[:self.step_counter], 
        )
        # Smoothing the dtw distance to extract signal from noisy predictions
        self.dtw_dist *= 1 - self.smoothing
        self.dtw_dist += self.smoothing * dtw_dist
        # Computing the reward based on the smoothed dtw distance
        self.reward = 1 if self.dtw_dist < self.eps else -1
        # Update variables
        self.prev_state = self.history_state[self.step_counter-1]
        self.prev_xy_action = self.xy_action
        self.step_counter += 1 
        return self.state, self.reward, done, {}
    
    def seed(self, seed):
        self.paths.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.paths.select_random()
        self.prev_state = self.paths.position_at_t(0)
        self.prev_xy_action = self.paths.position_at_t(0)
        self.history_state[0] = self.prev_state
        self.history_xy_action[0] = self.prev_xy_action
        self.step_counter = 1
        self.dtw_dist = 0
        return self.prev_state
    
    def _is_the_end(self, t, tol=1e-6):
        x_a, y_a = self.xy_action
        out_of_x_lim = (x_a > self.paths.x_max + tol) or (x_a < self.paths.x_min - tol)
        out_of_y_lim = (y_a > self.paths.y_max + tol) or (y_a < self.paths.y_min - tol)
        all_time_steps = True if self.step_counter == self.timesteps else False

        if (out_of_x_lim or out_of_y_lim) and not all_time_steps:
            if self.var_horizon:
                done = True
            else:
                done = False
                if self.reflective:
                    self.xy_action = self.paths.reflect_on_boundary(*self.xy_action)
                else:
                    self.xy_action = self.prev_xy_action
        elif all_time_steps:
            done = True
        else:
            done = False
        return done
    
    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            cheese = rendering.make_circle(5)
            mouse = rendering.make_circle(5)
            mouse.set_color(1.,0.,0.)
            self.cheese_trans = rendering.Transform()
            self.mouse_trans = rendering.Transform()
            cheese.add_attr(self.cheese_trans)
            mouse.add_attr(self.mouse_trans)
            self.viewer.add_geom(cheese)
            self.viewer.add_geom(mouse)

        self.viewer.draw_polyline(self.origin + self.history_state[:self.step_counter] * self.scale[None,:], 
                                  color=(0, 0, 255), linewidth=2)
        self.viewer.draw_polyline(self.origin + self.history_xy_action[:self.step_counter] * self.scale[None,:], 
                                  color=(255, 0, 0), linewidth=2)

        self.cheese_trans.set_translation(*(self.origin + self.state * self.scale))
        self.mouse_trans.set_translation(*(self.origin + self.xy_action * self.scale))
        return self.viewer.render(return_rgb_array= mode == 'rgb_array')
            
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':

    # plot_families()

    p = RotateRibbon(timesteps=200, n_paths=500, seed=0)
    # p = Circles(timesteps=200, n_paths=100, seed=0)
    # p = FreeArmLogistic(timesteps=200, n_paths=100, T=20, seed=0)
    # p = KinkAndWell(timesteps=200, n_paths=100, seed=0)

    import time
    env = MouseHiddenCheese(paths=p, var_horizon=False)
    for t in range(env.timesteps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f't: {t}/{env.timesteps}, Reward : {reward}')
        env.render()
        time.sleep(0.1)
        if done: break