import logging
logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from utils.read_data import read_data
from utils.get_env import get_venv
from utils.export_onnx import export_model


if __name__ == "__main__":
  logger = logging.getLogger()
  data = read_data()
  rng = np.random.default_rng(0)
  venv = get_venv(2, rng, logger)

  try:
    learner = PPO(env=venv, policy=MlpPolicy)
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
    )
    gail_trainer = GAIL(
        demonstrations=data,
        demo_batch_size=50,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net
    )
    gail_trainer.train(gail_trainer.gen_train_timesteps * 500)
    export_model(gail_trainer.policy, venv.observation_space)
  finally:
    logger.info('closing venv...')
    venv.close()
