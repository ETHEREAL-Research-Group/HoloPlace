import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import logging
logger = logging.getLogger()

observation_space = spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)


class RPMEnv(gym.Env):
  metadata = {'render_modes': ['human']}
  reward_range = (-float(1), float(1))

  def __init__(self, logger, rng=None, done_threshold=float('inf')):
    super().__init__()
    self.observation_space = observation_space
    self.action_space = action_space
    self.logger = logger
    self.rng = np.random.default_rng(0) if rng is None else rng
    self.done_threshold = done_threshold

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
    self._state = (self.rng.random(
        self.observation_space.shape[0]).astype('f') - 0.5) * 2
    self._info1 = self.get_matrix(self._state[:7])
    self._info2 = self.get_matrix(self._state[7:])
    self.counter = 0
    return self._state

  def step(self, action):
    self.counter += 1
    act_matrix = self.get_matrix(action)
    try:
      inversed = np.linalg.inv(act_matrix)
      self._info1 = np.matmul(inversed, self._info1)
      self._info2 = np.matmul(inversed, self._info2)
      self._state[:7] = self.destruct_matrix(self._info1)
      self._state[7:] = self.destruct_matrix(self._info2)
      return self._state, 0.0, self.counter > self.done_threshold, {}
    except:
      self.logger.error('singular matrix')
      return self._state, 0.0, True, {}


def get_env(logger=None, rng=None, done_threshold=float('inf')):
  if logger is None:
    logger = logging.getLogger()
  return Monitor(RPMEnv(logger, rng, done_threshold))


def get_venv(subproc=0, rng=None, logger=None, done_threshold=float('inf')) -> SubprocVecEnv:
  if logger is None:
    logger = logging.getLogger()
  if subproc > 0:
    return SubprocVecEnv([lambda: Monitor(RPMEnv(logger, rng, done_threshold))]*subproc)
  else:
    return DummyVecEnv([lambda: Monitor(RPMEnv(logger, rng, done_threshold))])


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()
  logger.info('testing `get_env()`...')
  logger.info(get_env(logger))
  from multiprocessing import cpu_count
  logger.info(f'testing `get_venv({cpu_count()})`...')
  logger.info(get_venv(cpu_count(), logger))
