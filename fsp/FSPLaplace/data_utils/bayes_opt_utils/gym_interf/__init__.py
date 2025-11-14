from gym.envs.registration import register
from .envs import InterfEnv, InterfNoLenses, InterfTelescope

register(id='interf-v1',
         entry_point='gym_interf.envs:InterfNoLenses')

register(id='interf-v2',
         entry_point='gym_interf.envs:InterfTelescope')

register(id='interf-v3',
         entry_point='gym_interf.envs:InterfEnv')

