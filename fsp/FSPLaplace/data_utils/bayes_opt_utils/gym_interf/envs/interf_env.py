import gym
from matplotlib import pyplot as plt
from math import *
from cmath import *
import numpy as np
import time as tm
from scipy import optimize

from .calc_image_cpp import calc_image as calc_image_cpp
from .utils import reflect, project, rotate_x, dist, angle_between, visibility_for_telescopes
from .domain_randomizer import DomainRandomizer
from .exp_state_provider import ExpStateProvider


class InterfEnv(gym.Env):
    n_points = 64
    n_frames = 16
    n_actions = 6

    # mirror screw step l / L, (ratio of delta screw length to vertical distance)
    one_mirror_step = 0.52 * 1e-6
    far_mirror_max_screw_value = one_mirror_step * 5000
    near_mirror_max_screw_value = one_mirror_step * 2500

    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (0, 1)

    observation_space = gym.spaces.Box(low=0, high=255, shape=(n_frames, n_points, n_points), dtype=np.uint8)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)

    # in mm
    lamb = 6.35 * 1e-4
    omega = 1

    dist_between_telescopes = 500
    one_lens_step = 0.7 * 1e-3
    lens_mount_max_screw_value = 6000 * one_lens_step

    # initial normals
    mirror1_x_rotation_angle = 3 * pi / 4
    mirror2_x_rotation_angle = -pi / 4

    # camera matrix size (in mm)
    camera_size = 3.57

    done_visibility = 0.9999

    def __init__(self, a=200, b=700, c=100, f1=50, f2=50, beam_radius=0.714):
        # size of interferometer (in mm)
        self.a = a
        self.b = b
        self.c = c

        # focuses of lenses (in mm) of first telescope
        self.f1 = f1
        # focuses of lenses (in mm) of second telescope
        self.f2 = f2

        self.radius = beam_radius

        self._visibility = visibility_for_telescopes

        self.mirror1_screw_x = 0
        self.mirror1_screw_y = 0
        self.mirror2_screw_x = 0
        self.mirror2_screw_y = 0

        self.state = None
        self.n_steps = None
        self.info = None
        self.visib = None
        self.dist = None
        self.angle = None
        self.noise_coef = 0
        self.backward_frames = 4
        self.phase_std = 0
        self.max_steps = 100

        self.beam1_mask = None
        self.beam2_mask = None

        self.beam1_rotation = 0
        self.beam2_rotation = 0
        self.beam1_sigmax = 1
        self.beam1_sigmay = 1
        self.beam2_sigmax = 1
        self.beam2_sigmay = 1

        self._calc_reward = self._calc_reward_visib_minus_1
        self._calc_image = calc_image_cpp
        self._image_randomizer = DomainRandomizer('data')
        self._use_beam_masks = False

        self._exp_state_provider = ExpStateProvider('saved_states')
        self._exp_state_provider.get_state()
        self._use_exp_data = False

        # image min & max coords
        self.x_min = -InterfEnv.camera_size / 2
        self.x_max = InterfEnv.camera_size / 2
        self.y_min = -InterfEnv.camera_size / 2
        self.y_max = InterfEnv.camera_size / 2

        # distance between lenses
        # reduced_lens_dist = ((lens_dist - f1 - f2) / lens_mount_max_screw_value - 0.5) / 2
        self.reduced_lens_dist1 = None
        self.reduced_lens_dist2 = None

    def set_radius(self, value):
        self.radius = value
        
    def set_max_steps(self, value):
        self.max_steps = value

    def shift_camera_position(self, delta_x, delta_y):
        self.x_min = (-0.5 + delta_x) * InterfEnv.camera_size
        self.x_max = (0.5 + delta_x) * InterfEnv.camera_size
        self.y_min = (-0.5 + delta_y) * InterfEnv.camera_size
        self.y_max = (0.5 + delta_y) * InterfEnv.camera_size

    def set_phase_std(self, value):
        self.phase_std = value

    def set_beam_rotation(self, value):
        self.beam1_rotation = value
        self.beam2_rotation = value

    def set_beam_ellipticity(self, value):
        self.beam1_sigmax = 1.0 / np.sqrt(value)
        self.beam1_sigmay = 1.0 * np.sqrt(value)
        self.beam2_sigmax = self.beam1_sigmax
        self.beam2_sigmay = self.beam1_sigmay

    def set_calc_reward(self, method):
        if method == 'visib_minus_1':
            self._calc_reward = self._calc_reward_visib_minus_1
        elif method == 'delta_visib':
            self._calc_reward = self._calc_reward_delta_visib
        else:
            assert False, 'unknown reward_calc == {} optnions are "visib_minus1", "delta_visib"'.format(method)

    def set_calc_image(self, device):
        if device == 'cpu':
            self._calc_image = calc_image_cpp
        elif device == 'gpu':
            from .calc_image_cuda import calc_image as calc_image_gpu
            self._calc_image = calc_image_gpu
        else:
            assert False, 'unknown device == {} optnions are "cpu", "gpu"'.format(device)

    def set_backward_frames(self, val):
        self.backward_frames = val

    def add_noise(self, noise_coef):
        self.noise_coef = noise_coef

    def use_beam_masks(self, enabled):
        self._use_beam_masks = enabled

    def use_exp_data(self, enabled):
        self._use_exp_data = enabled

    def get_keys_to_action(self):
        return {
            (ord('w'),): 0,
            (ord('s'),): 1,
            (ord('a'),): 2,
            (ord('d'),): 3,
            (ord('i'),): 4,
            (ord('k'),): 5,
            (ord('j'),): 6,
            (ord('l'),): 7,
            (ord('n'),): 8,
            (ord('m'),): 9,
            (ord('v'),): 10,
            (ord('b'),): 11
        }

    def seed(self, seed=None):
        self.action_space.seed(seed)

    def step(self, actions):
        """

        :param action: (mirror_name, axis, delta_angle)
        :return: (state, reward, done, info)
        """

        self.n_steps += 1

        for action_id, action_value in enumerate(actions):
            self._take_action(action_id, action_value)

        center1, wave_vector1, center2, wave_vector2, self.angle = self._calc_centers_and_wave_vectors()
        proj_1, proj_2, self.dist = self._calc_projection_distance(center1, wave_vector1, center2, wave_vector2)
        self.info['proj_1'] = proj_1
        self.info['proj_2'] = proj_2
        self.info['dist'] = self.dist

        self.state, tot_intens = self._calc_state(center1, wave_vector1, proj_1, center2, wave_vector2, proj_2)
        reward = self._calc_reward(tot_intens)

        return self.state, reward, self.game_over(), self.info

    def reset(self, actions=None):
        self.n_steps = 0
        self.info = {}

        self.beam1_mask = self._image_randomizer.get_mask()
        self.beam2_mask = self._image_randomizer.get_mask()

        self.reduced_lens_dist1 = 0
        self.reduced_lens_dist2 = 0

        self.mirror1_screw_x = 0
        self.mirror1_screw_y = 0
        self.mirror2_screw_x = 0
        self.mirror2_screw_y = 0

        if actions is None:
            actions = InterfEnv.action_space.sample()

        for action_id, action_value in enumerate(actions):
            self._take_action(action_id, action_value)

        c1, k1, c2, k2, self.angle = self._calc_centers_and_wave_vectors()
        proj_1, proj_2, self.dist = self._calc_projection_distance(c1, k1, c2, k2)
        self.state, tot_intens = self._calc_state(c1, k1, proj_1, c2, k2, proj_2)

        # should be called after self._calc_state()
        self.visib = self._calc_visib(tot_intens)

        return self.state

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return self.state
        elif mode == 'human':
            plt.imshow(self.state[0], vmin=0, vmax=4)
            plt.ion()
            plt.pause(1)
            plt.show()
        else:
            return None

    def get_approx_visib(self):

        center1, wave_vector1, center2, wave_vector2, self.angle = self._calc_centers_and_wave_vectors()
        proj_1, proj_2, _ = self._calc_projection_distance(center1, wave_vector1, center2, wave_vector2)

        radius_bottom, curvature_radius = self._calc_beam_propagation(self.reduced_lens_dist1, self.reduced_lens_dist2)
        beam2_amplitude = self.radius / radius_bottom

        has_interf = True  # band_width > 4 * cell_size

        pixel_size = (self.y_max - self.y_min) / InterfEnv.n_points

        _, tot_intens = self._calc_image(
            self.x_min, self.y_min, InterfEnv.n_points, InterfEnv.n_points, pixel_size,
            wave_vector1, center1, self.radius, self.beam1_mask, 3.57, 4*64, self.beam1_sigmax, self.beam1_sigmay, 1.0,
            self.beam1_rotation,
            wave_vector2, center2, radius_bottom, self.beam2_mask, 3.57, 4*64, self.beam2_sigmax, self.beam2_sigmay,
            beam2_amplitude, self.beam2_rotation,
            curvature_radius, 10*(InterfEnv.n_frames - self.backward_frames), 10*self.backward_frames, InterfEnv.lamb,
            InterfEnv.omega,
            noise_coef=self.noise_coef,
            ampl_std=0,
            phase_std=self.phase_std,
            use_beam_masks=self._use_beam_masks,
            has_interf=has_interf)

        def visib(vmin, vmax):
            return (vmax - vmin) / (vmax + vmin)

        return visib(float(min(tot_intens)), float(max(tot_intens)))


    def _calc_beam_propagation(self, lens_dist1, lens_dist2):
        lens_dist1 = lens_dist1 * InterfEnv.lens_mount_max_screw_value
        lens_dist2 = lens_dist2 * InterfEnv.lens_mount_max_screw_value

        if lens_dist1 == 0:
            lens_dist1 = 1e-6

        if lens_dist2 == 0:
            lens_dist2 = 1e-6

        def free_space(length):
            return np.array([[1, length], [0, 1]])

        def lens(focal_length):
            return np.array([[1, 0], [-1 / focal_length, 1]])

        dist_between_lenses1 = 2 * self.f1 + lens_dist1
        dist_between_lenses2 = 2 * self.f2 + lens_dist2
        dist_between_tel = self.dist_between_telescopes - lens_dist1

        dist_to_camera = self.c + self.a + self.b - dist_between_lenses1 - dist_between_lenses2 - dist_between_tel
        abcd_matrix = \
            free_space(dist_to_camera) @ \
            lens(self.f2) @ \
            free_space(dist_between_lenses2) @ \
            lens(self.f2) @ \
            free_space(dist_between_tel) @ \
            lens(self.f1) @ \
            free_space(dist_between_lenses1) @ \
            lens(self.f1)
        inv_q = -1j * self.lamb / (np.pi * self.radius ** 2)
        inv_q_prime = (abcd_matrix[1][0] + abcd_matrix[1][1] * inv_q) / (abcd_matrix[0][0] + abcd_matrix[0][1] * inv_q)

        curvature_radius = 1 / (np.real(inv_q_prime) + 1e-9)
        beam_radius = np.sqrt(-self.lamb / np.imag(inv_q_prime) / np.pi)
        return beam_radius, curvature_radius

    def _take_action(self, action, normalized_step_length):
        """
        0 - do nothing
        [1, 2, 3, 4] - mirror1
        [5, 6, 7, 8] - mirror2
        :param action:
        :return:
        """

        if action == 0:
            self.mirror1_screw_x = np.clip(self.mirror1_screw_x + normalized_step_length, -1, 1)
        elif action == 1:
            self.mirror1_screw_y = np.clip(self.mirror1_screw_y + normalized_step_length, -1, 1)
        elif action == 2:
            self.mirror2_screw_x = np.clip(self.mirror2_screw_x + normalized_step_length, -1, 1)
        elif action == 3:
            self.mirror2_screw_y = np.clip(self.mirror2_screw_y + normalized_step_length, -1, 1)
        elif action == 4:
            self.reduced_lens_dist1 = np.clip(self.reduced_lens_dist1 + normalized_step_length, -1, 1)
        elif action == 5:
            self.reduced_lens_dist2 = np.clip(self.reduced_lens_dist2 + normalized_step_length, -1, 1)
        else:
            assert False, 'unknown action = {}'.format(action)

    def _calc_centers_and_wave_vectors(self):
        assert abs(self.mirror1_screw_x) <= 1, self.mirror1_screw_x
        assert abs(self.mirror1_screw_y) <= 1, self.mirror1_screw_y
        assert abs(self.mirror2_screw_x) <= 1, self.mirror2_screw_x
        assert abs(self.mirror2_screw_y) <= 1, self.mirror2_screw_y

        mirror1_screw_x_value = self.mirror1_screw_x * InterfEnv.far_mirror_max_screw_value
        mirror1_screw_y_value = self.mirror1_screw_y * InterfEnv.far_mirror_max_screw_value
        mirror1_x_component = - mirror1_screw_x_value / np.sqrt(mirror1_screw_x_value ** 2 + 1)
        mirror1_y_component = - mirror1_screw_y_value / np.sqrt(mirror1_screw_y_value ** 2 + 1)
        mirror1_z_component = np.sqrt(1 - mirror1_x_component ** 2 - mirror1_y_component ** 2)
        mirror1_normal = np.array(
            [mirror1_x_component, mirror1_y_component, mirror1_z_component],
            dtype=np.float64
        )
        mirror1_normal = rotate_x(mirror1_normal, InterfEnv.mirror1_x_rotation_angle)

        mirror2_screw_x_value = self.mirror2_screw_x * InterfEnv.near_mirror_max_screw_value
        mirror2_screw_y_value = self.mirror2_screw_y * InterfEnv.near_mirror_max_screw_value
        mirror2_x_component = - mirror2_screw_x_value / np.sqrt(mirror2_screw_x_value ** 2 + 1)
        mirror2_y_component = - mirror2_screw_y_value / np.sqrt(mirror2_screw_y_value ** 2 + 1)
        mirror2_z_component = np.sqrt(1 - mirror2_x_component ** 2 - mirror2_y_component ** 2)
        mirror2_normal = np.array(
            [mirror2_x_component, mirror2_y_component, mirror2_z_component],
            dtype=np.float64
        )
        mirror2_normal = rotate_x(mirror2_normal, InterfEnv.mirror2_x_rotation_angle)

        self.info['mirror1_normal'] = mirror1_normal
        self.info['mirror2_normal'] = mirror2_normal

        wave_vector1 = np.array([0, 0, 1], dtype=np.float64)
        center1 = np.array([0, 0, -self.c], dtype=np.float64)

        center2 = np.array([0, -self.a, -(self.b + self.c)], dtype=np.float64)
        wave_vector2 = np.array([0, 0, 1], dtype=np.float64)

        # reflect wave vector by first mirror
        center2 = project(center2, wave_vector2, mirror1_normal, np.array([0, -self.a, -self.c]))
        wave_vector2 = reflect(wave_vector2, mirror1_normal)
        self.info['reflect_with_mirror1'] = 'center = {}, k = {}'.format(center2, wave_vector2)

        # reflect wave vector by second mirror
        center2 = project(center2, wave_vector2, mirror2_normal, np.array([0, 0, -self.c]))
        wave_vector2 = reflect(wave_vector2, mirror2_normal)
        self.info['reflect_with_mirror2'] = 'center = {}, k = {}'.format(center2, wave_vector2)

        self.info['kvector'] = wave_vector2

        angle = angle_between(wave_vector1, wave_vector2)
        self.info['angle_between_beams'] = angle

        return center1, wave_vector1, center2, wave_vector2, angle

    def _calc_projection_distance(self, center1, wave_vector1, center2, wave_vector2):
        projection_plane_normal = np.array([0, 0, 1])
        projection_plane_center = np.array([0, 0, 0])

        proj_1 = project(center1, wave_vector1, projection_plane_normal, projection_plane_center)
        proj_2 = project(center2, wave_vector2, projection_plane_normal, projection_plane_center)
        distance = dist(proj_1, proj_2)

        return proj_1, proj_2, distance

    def _calc_reward_visib_minus_1(self, tot_intens):
        self.visib = self._calc_visib(tot_intens)
        self.info['visib'] = self.visib
        return self.visib - 1.

    def _calc_reward_delta_visib(self, tot_intens):
        prev_visib = self.visib
        self.visib = self._calc_visib(tot_intens)
        self.info['visib'] = self.visib
        return self.visib - prev_visib

    def _calc_visib(self, tot_intens):
        def visib(vmin, vmax):
            return (vmax - vmin) / (vmax + vmin)

        imin, imax = min(tot_intens), max(tot_intens)
        self.info['fit_time'] = 0
        self.info['imin'] = imin
        self.info['imax'] = imax

        return visib(float(min(tot_intens)), float(max(tot_intens)))

        def fit_func(x, a, b, phi):
            return a + b * np.cos(x + phi)

        try:
            tstart = tm.time()
            params, params_covariance = optimize.curve_fit(
                fit_func, np.linspace(0, 2 * pi, InterfEnv.n_frames),
                tot_intens,
                p0=[np.mean(tot_intens), np.max(tot_intens) - np.mean(tot_intens), 0])
            tend = tm.time()

            self.info['fit_time'] = tend - tstart

            a_param = params[0]
            b_param = abs(params[1])

            fmax = a_param + b_param
            fmin = max(a_param - b_param, 0)

            return visib(fmin, fmax)
        except RuntimeError:
            return visib(float(min(tot_intens)), float(max(tot_intens)))

    def game_over(self):
        return self.visib > InterfEnv.done_visibility or \
               self.n_steps >= self.max_steps

    def _calc_state(self, center1, wave_vector1, proj_1, center2, wave_vector2, proj_2):
        if self._use_exp_data:
            state, tot_intens, handles = self._exp_state_provider.get_state()
            self.mirror1_screw_x, self.mirror1_screw_y, self.mirror2_screw_x, self.mirror2_screw_y = handles
            print('-handles', -handles / 5000)
            self.info['state_calc_time'] = 0
            return state, tot_intens

        state_calc_time = 0

        tstart = tm.time()

        # band_width_x = InterfEnv.lamb / abs(wave_vector2[0])
        # band_width_y = InterfEnv.lamb / abs(wave_vector2[1])
        # band_width = min(band_width_x, band_width_y)
        # cell_size = (self.x_max - self.x_min) / InterfEnv.n_points

        radius_bottom, curvature_radius = self._calc_beam_propagation(self.reduced_lens_dist1, self.reduced_lens_dist2)
        beam2_amplitude = self.radius / radius_bottom
        self.info['r_curvature'] = curvature_radius
        self.info['reduced_lens_dist1'] = self.reduced_lens_dist1
        self.info['reduced_lens_dist2'] = self.reduced_lens_dist2
        self.info['radius_bottom'] = radius_bottom

        kvector = wave_vector2 * 2 * np.pi / self.lamb
        self.info['visib_device'] = self._visibility(
            self.radius, radius_bottom, curvature_radius,
            proj_2[0], proj_2[1], kvector[0], kvector[1], self.lamb)

        has_interf = True #band_width > 4 * cell_size

        # print('band_width / (4 * cells_size)', band_width / (2 * cell_size))

        # print('band_width_x = {}, band_width_y = {}, cell_size = {}, interf = {}'.format(
        #     band_width_x, band_width_y, cell_size, has_interf)
        # )

        pixel_size = (self.y_max - self.y_min) / InterfEnv.n_points

        state = self._calc_image(
            self.x_min, self.y_min, InterfEnv.n_points, InterfEnv.n_points, pixel_size,
            wave_vector1, center1, self.radius, self.beam1_mask, 3.57, 64, self.beam1_sigmax, self.beam1_sigmay, 1.0, self.beam1_rotation,
            wave_vector2, center2, radius_bottom, self.beam2_mask, 3.57, 64, self.beam2_sigmax, self.beam2_sigmay, beam2_amplitude, self.beam2_rotation,
            curvature_radius, InterfEnv.n_frames - self.backward_frames, self.backward_frames, InterfEnv.lamb, InterfEnv.omega,
            noise_coef=self.noise_coef,
            ampl_std=0.2,
            phase_std=self.phase_std,
            use_beam_masks=self._use_beam_masks,
            has_interf=has_interf)

        tend = tm.time()

        state_calc_time += tend - tstart

        self.info['state_calc_time'] = state_calc_time

        return state
