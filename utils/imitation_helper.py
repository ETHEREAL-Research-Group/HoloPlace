import logging
logger = logging.getLogger()


def custom_trainer(data, output_path, num_epochs=10000):
  import torch as th
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import DataLoader
  batch_size = 500
  input_size = 7
  output_size = 7
  patience = float('inf')
  min_delta = 1e-7
  device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
  logger.info('creating the model...')
  model = nn.Sequential(
      nn.Linear(input_size, 64),
      nn.Tanh(),
      # nn.Dropout(0.05),
      nn.Linear(64, 64),
      nn.Tanh(),
      # nn.Dropout(0.05),
      nn.Linear(64, 64),
      nn.Tanh(),
      # nn.Dropout(0.05),
      nn.Linear(64, 64),
      nn.Tanh(),
      # nn.Dropout(0.05),
      nn.Linear(64, output_size),
  ).to(device)

  # Define the loss function and optimizer
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
  scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=1000)

  train, val = data

  train = DataLoader(train, batch_size=batch_size, shuffle=False)
  val = DataLoader(val, batch_size=batch_size, shuffle=False)
  best_val_loss = float('inf')
  counter = 0
  for epoch in range(num_epochs):
    model.train()
    # pbar = tqdm(desc=f'Training Epoch {epoch}', total=len(train))
    train_loss = None
    for _id_batch, (x_batch, y_batch) in enumerate(train):
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      train_loss = loss.item()
      # pbar.set_postfix({
      #     'Train Loss': train_loss
      # })

      # pbar.update(1)

    model.eval()
    val_losses = []
    for _id_batch, (x_batch, y_batch) in enumerate(val):
      outputs = model(x_batch)
      val_loss = criterion(y_batch, outputs)
      val_losses.append(val_loss.item())
    mean_val_loss = sum(val_losses)/len(val_losses)

    # pbar.set_postfix({
    #     'Train Loss': train_loss,
    #     'Avg Val Loss': mean_val_loss,
    #     'learning rate': optimizer.param_groups[0]['lr'],
    #     'Best': 'True' if mean_val_loss < best_val_loss else 'False'
    # })
    # pbar.close()

    logger.info(
        f'Epoch {epoch:>5d}/{num_epochs} {(epoch*100)//num_epochs:>3d}%: Train Loss={train_loss:.4e}, Avg Val Loss={mean_val_loss:.4e}, Learning Rate={optimizer.param_groups[0]["lr"]:.4e}, Best={"True" if mean_val_loss < best_val_loss else "False"}')

    if mean_val_loss < best_val_loss:
      best_val_loss = mean_val_loss
      counter = 0
      th.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'val_loss': mean_val_loss,
      }, f'{output_path[:-4]}pt')
    elif mean_val_loss > (best_val_loss + min_delta):
      counter += 1

    if counter > patience:
      logger.info('stopping training early...')
      break

    scheduler.step()
  try:
    pass
    # logger.info('loading the best model...')
    # checkpoint = th.load(f'{output_path[:-4]}pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
  except:
    logger.error(f'{output_path[:-4]}pt could not be found...')
  dummy_input = th.randn(input_size,)
  logger.info('exporting to onnx...')
  th.onnx.export(
      model,
      th.ones(dummy_input.shape, dtype=th.float32,
              device=device),
      output_path,
      opset_version=9,
      input_names=['input'],
      output_names=['output']
  )
  logger.info('training finished')


def imitation_trainer(data, output_path, method='gail', num_epochs=200):
  from utils.export_onnx import export_model
  from utils.get_env import get_venv
  from imitation.rewards.reward_nets import BasicRewardNet
  from imitation.algorithms.adversarial.gail import GAIL
  from imitation.algorithms.bc import BC
  from stable_baselines3.ppo import MlpPolicy
  from stable_baselines3 import PPO
  import numpy as np
  from multiprocessing import cpu_count

  logger = logging.getLogger()
  logger.info('reading the data...')
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
    trainer = None
    if method == 'gail':
      trainer = GAIL(
          demonstrations=data,
          demo_batch_size=512,
          venv=venv,
          gen_algo=learner,
          reward_net=reward_net
      )
      num_epochs = trainer.gen_train_timesteps * num_epochs
    else:
      trainer = BC(
          observation_space=venv.observation_space,
          action_space=venv.action_space,
          demonstrations=data,
          rng=rng,
          batch_size=512
      )
    logger.info('begin training...')
    trainer.train(num_epochs)
    logger.info('exporting to onnx...')
    export_model(trainer.policy, venv.observation_space, output_path)
    logger.info('training finished')
  finally:
    logger.info('closing venv...')
    venv.close()


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()
  # Testing custom nn:
  from read_data import read_data
  data, custom_data = read_data()
  custom_trainer(custom_data, 'output/custom_test.onnx', 200)
  # Testing gail
  imitation_trainer(data, 'output/gail_test.onnx', 20)