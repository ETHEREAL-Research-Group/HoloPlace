import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import logging
logger = logging.getLogger()


class RPMEnv(gym.Env):
  metadata = {'render_modes': ['human']}
  reward_range = (-float(1), float(1))
  def __init__(self, logger, rng=None):
    super().__init__()
    self.observation_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
    self.action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
    self.logger = logger
    self.rng = np.random.default_rng(0) if rng is None else rng

  def get_matrix(self, flat: np.array):
    pos = flat[:3]
    rot = flat[3:]
    rot_matrix = R.from_quat(rot).as_matrix()
    res = np.zeros(shape=(4, 4), dtype=np.float32)
    res[:3, :3] = rot_matrix
    res[:3, 3] = pos
    res[3, 3] = 1
    return res

  def destruct_matrix(self, matrix: np.array):
    res = np.zeros(shape=(7,), dtype=np.float32)
    res[:3] = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3]).as_quat()
    res[3:] = rot
    return res

  def reset(self):
    self._state = (self.rng.random(7).astype('f') - 0.5) * 2
    self._info = self.get_matrix(self._state)
    self.counter = 0
    return self._state

  def step(self, action):
    self.counter += 1
    act_matrix = self.get_matrix(action)
    try:
      inversed = np.linalg.inv(act_matrix)
      self._info = np.matmul(inversed, self._info)
      self._state = self.destruct_matrix(self._info)
      return self._state, 0.0, False, {}
    except:
      self.logger.error('singular matrix')
      return self._state, 0.0, True, {}


def get_env(logger):
  if logger is None:
    logger = logging.getLogger()
  return Monitor(RPMEnv(logger))


def get_venv(subproc=0, rng=None, logger=None) -> SubprocVecEnv:
  if logger is None:
    logger = logging.getLogger()
  if subproc > 0:
    return SubprocVecEnv([lambda: Monitor(RPMEnv(logger, rng))]*subproc)
  else:
    return DummyVecEnv([lambda: Monitor(RPMEnv(logger, rng))])


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()
  logger.info('testing `get_env()`...')
  logger.info(get_env(logger))
  logger.info('testing `get_venv(4)`...')
  logger.info(get_venv(4, logger))
