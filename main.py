if __name__ == "__main__":
  # on linux, by default it's fork -- we change it to spawn for some reason
  from multiprocessing import set_start_method, cpu_count
  set_start_method('spawn')
  import logging
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  import numpy as np
  from stable_baselines3 import PPO
  from stable_baselines3.ppo import MlpPolicy
  # from imitation.algorithms.adversarial.gail import GAIL
  from imitation.algorithms.bc import BC
  from imitation.rewards.reward_nets import BasicRewardNet
  from utils.read_data import read_data
  from utils.get_env import get_venv
  from utils.export_onnx import export_model

  logger = logging.getLogger()
  logger.info('reading the data...')
  data = read_data()
  rng = np.random.default_rng(0)
  logger.info(f'creating venv with cpu count {cpu_count()}')
  venv = get_venv(cpu_count(), rng, logger)

  try:
    logger.info('creating the learner, policy, and trainer...')
    learner = PPO(env=venv, policy=MlpPolicy)
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
    )
    # gail_trainer = GAIL(
    #     demonstrations=data,
    #     demo_batch_size=512,
    #     venv=venv,
    #     gen_algo=learner,
    #     reward_net=reward_net
    # )
    bc_trainer = BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=data,
        rng=rng,
        batch_size=512
    )
    logger.info('begin training...')
    # gail_trainer.train(gail_trainer.gen_train_timesteps * 200)
    bc_trainer.train(n_epochs=1000)
    logger.info('exporting to onnx...')
    # export_model(gail_trainer.policy, venv.observation_space, 'test3.onnx')
    export_model(bc_trainer.policy, venv.observation_space, 'bad.onnx')
  finally:
    logger.info('closing venv...')
    venv.close()
