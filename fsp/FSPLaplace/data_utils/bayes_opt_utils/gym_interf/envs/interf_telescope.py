import gym
import numpy as np

from .interf_env import InterfEnv
from .utils import visibility_for_telescopes


class InterfTelescope(InterfEnv):
    n_actions = 5
    action_space = gym.spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)

    def __init__(self):
        super().__init__(a=200, b=300, c=100, beam_radius=0.714)
        self._visibility = visibility_for_telescopes

    def step(self, actions):
        return super().step([*actions, 0])

    def reset(self, actions=None):
        if actions is None:
            actions = InterfTelescope.action_space.sample()
        return super().reset([*actions, 0])

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
            (ord('m'),): 9
        }

    def _calc_beam_propagation(self, lens_dist, lens_dist2):
        lens_dist = lens_dist * InterfEnv.lens_mount_max_screw_value

        if lens_dist == 0:
            lens_dist = 1e-6

        dist_between_lenses = 2 * self.f1 + lens_dist
        dist_to_camera = self.c + self.a + self.b - dist_between_lenses
        curvature_radius_eq = dist_to_camera - self.f1 ** 2 / lens_dist - self.f1

        beam_radius_eq = np.abs(
            lens_dist *
            (dist_to_camera / (self.f1 * self.f1) - 1.0 / self.f1)
            - self.f1 / self.f1
        ) * self.radius

        # TODO explain difference between
        #  beam_radius / beam_radius_eq and curvature_radius / curvature_radius_eq
        return beam_radius_eq, curvature_radius_eq
