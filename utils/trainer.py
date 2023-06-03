import threading
import torch as th
import logging
from random import shuffle
logger = logging.getLogger()

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


def position_distance(points1, points2):
  positions1 = points1[:, :3]
  positions2 = points2[:, :3]
  res = th.norm(positions1 - positions2, dim=1, p=2)
  return res.mean()


def normalize_quaternions(quaternions):
  # Normalize the quaternions to unit length
  quaternions_norm = th.norm(quaternions, dim=1, keepdim=True)
  return quaternions / quaternions_norm


def quaternion_distance(points1, points2):
  quaternions1 = points1[:, 3:]
  quaternions2 = points2[:, 3:]
  dot_product = th.sum(normalize_quaternions(quaternions1)
                       * normalize_quaternions(quaternions2), dim=1)
  res = 1 - dot_product**2
  return res.mean()


def chamfer_distance(points1, points2):
  # Expand dimensions to perform element-wise operations
  # points1_expanded = points1.unsqueeze(1)  # (num_points1, 1, 7)
  # points2_expanded = points2.unsqueeze(0)  # (1, num_points2, 7)

  # # Calculate Euclidean distances
  # # (batch_size, num_points1, num_points2)
  # distances = th.norm(points1_expanded - points2_expanded, dim=-1)

  distances = th.cdist(points1, points2, p=2)

  # Calculate minimum distance from points1 to points2
  min_distances_1to2, _ = th.min(
      distances, dim=-1)  # (batch_size, num_points1)

  # Calculate minimum distance from points2 to points1
  min_distances_2to1, _ = th.min(
      distances, dim=-2)  # (batch_size, num_points2)

  # Calculate Chamfer distance loss
  chamfer_loss = th.mean(min_distances_1to2) + th.mean(min_distances_2to1)

  return chamfer_loss


def get_model():
  import torch.nn as nn

  class QuatNormLayer(nn.Module):
    def __init__(self):
      super(QuatNormLayer, self).__init__()

    # def forward(self, x):
    #   # Implement the desired functionality of the custom layer
    #   x_copy = th.Tensor(x)
    #   x_copy[:, 3:] = x[:,3:] / th.norm(x[:, 3:], dim=1, keepdim=True)
    #   return x_copy

    def forward(self, x):
      # Split the input tensor into position and quaternion parts
      if (len(x.shape) == 1):
        position = x[:3]
        quaternion = x[3:]
      else:
        position = x[:, :3]
        quaternion = x[:, 3:]

      # Normalize the quaternion part
      quaternion_norm = th.norm(quaternion, p=2, dim=-1, keepdim=True)
      normalize_quaternions = quaternion / quaternion_norm
      # quaternion = nn.functional.normalize(quaternion, p=2, dim=-1)

      # Concatenate the position and normalized quaternion parts
      normalized_x = th.cat([position, normalize_quaternions], dim=-1)

      return normalized_x

  input_size = 14
  output_size = 7
  logger.info('creating the model...')
  model = nn.Sequential(
      nn.Linear(input_size, 256),
      nn.Tanh(),
      nn.BatchNorm1d(256),
      nn.Dropout(0.1),

      nn.Linear(256, 256),
      nn.Tanh(),
      nn.BatchNorm1d(256),
      nn.Dropout(0.1),

      nn.Linear(256, 256),
      nn.Tanh(),
      nn.BatchNorm1d(256),
      nn.Dropout(0.1),

      nn.Linear(256, 256),
      nn.Tanh(),
      nn.BatchNorm1d(256),
      nn.Dropout(0.1),

      nn.Linear(256, output_size),
      QuatNormLayer()
  ).to(device)

  return model


def prep_data(data):
  batch_size = 512

  from torch.utils.data import DataLoader
  train, val = data

  train = DataLoader(train, batch_size=batch_size, shuffle=True)
  val = DataLoader(val, batch_size=batch_size, shuffle=True)
  return train, val


def static_vars(**kwargs):
  def decorate(func):
    for k in kwargs:
      setattr(func, k, kwargs[k])
    return func
  return decorate


@static_vars(lock=threading.Lock())
def export(model, path):
  logger.info(f'exporting model to {path}')
  export.lock.acquire()  # type: ignore
  th.onnx.export(
      model,
      th.ones([1, 14], dtype=th.float32, device=device),
      path,
      opset_version=9,
      input_names=['input'],
      output_names=['output']
  )
  export.lock.release()  # type: ignore


def train_model(data, output_path, num_epochs=4096, loss_fn='chamfer', mode='flatten', tensor_prefix='default', batches_per_epoch=1):
  import torch.optim as optim
  from torch.utils.tensorboard.writer import SummaryWriter
  from pathlib import Path
  import os
  tensor_path = Path(output_path).parent.joinpath('summary', tensor_prefix)
  os.makedirs(tensor_path, exist_ok=True)
  writer = SummaryWriter(tensor_path)

  patience = 2048+1
  min_delta = 0

  model = get_model()

  # criterion = nn.MSELoss()
  if loss_fn == 'chamfer':
    criterion = chamfer_distance
  elif loss_fn == 'mse':
    criterion = th.nn.MSELoss()
  else:
    raise Exception('invalid argument for loss')

  optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
  scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=128)

  if mode == 'flatten':
    data = prep_data(data)

  train, val = data

  best_val_loss = float('inf')
  counter = 0

  for epoch in range(num_epochs):
    model.train()
    # pbar = tqdm(desc=f'Training Epoch {epoch}', total=len(train))
    train_losses = []
    for _ in range(batches_per_epoch):
      if mode == 'ep':
        shuffle(train)
      for _, (x_batch, y_batch) in enumerate(train):
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        optimizer.step()
        train_losses.append(loss.item())

      # pbar.update(1)
    mean_train_loss = sum(train_losses)/len(train_losses)
    writer.add_scalar(f'Loss({loss_fn})/Train', mean_train_loss, epoch)
    model.eval()
    val_losses = []
    for _, (x_batch, y_batch) in enumerate(val):
      outputs = model(x_batch)
      val_loss = criterion(outputs, y_batch)
      val_losses.append(val_loss)
    mean_val_loss = sum(val_losses) / \
        len(val_losses) if len(val_losses) > 0 else float('inf')
    writer.add_scalar(f'Loss({loss_fn})/Val', mean_val_loss, epoch)

    # pbar.set_postfix({
    #     'Train Loss': train_loss,
    #     'Avg Val Loss': mean_val_loss,
    #     'learning rate': optimizer.param_groups[0]['lr'],
    #     'Best': 'True' if mean_val_loss < best_val_loss else 'False'
    # })
    # pbar.close()

    best = mean_val_loss < (best_val_loss + min_delta)

    logger.info(
        f'epoch {epoch:>5d}/{num_epochs} {(epoch*100)//num_epochs:>3d}%: avg_train_loss={mean_train_loss:.4e}, avg_val_loss={mean_val_loss:.4e}, lr={optimizer.param_groups[0]["lr"]:.1e}, best={best}')

    if best:
      best_val_loss = mean_val_loss
      counter = 0
      th.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'val_loss': mean_val_loss
      }, f'{output_path[:-4]}pt')
    else:
      counter += 1

    if epoch % 500 == 0:
      export(model, output_path[:-5] + f'-{epoch}.onnx')

    if counter > patience:
      logger.info('stopping training early...')
      break
    scheduler.step()

  try:
    pass
    logger.info('loading the best model...')
    checkpoint = th.load(f'{output_path[:-4]}pt')
    model.load_state_dict(checkpoint['model_state_dict'])
  except:
    logger.error(f'{output_path[:-4]}pt could not be found...')

  export(model, output_path)
  logger.info('training finished')
  writer.close()


if __name__ == '__main__':
  logging.basicConfig(
      format='%(threadName)s %(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()
  from data import read_data
  from copy import deepcopy
  import threading
  # Testing custom nn:
  logger.info('testing `train_model` with mode=eq and loss=chamfer')
  data = read_data('data/data.csv', 'data/events.csv', flatten=False)
  t1 = threading.Thread(target=train_model, args=(
      deepcopy(data), 'output/test_ep.onnx', 2**15), kwargs={'mode': 'ep', 'tensor_prefix': 'ep'})
  t1.start()

  # train_model(data, 'output/test.onnx', 10, tensor_prefix='ep')
  logger.info('testing `train_model` with mode=flatten and loss=chamfer')
  flat_data = read_data('data/data.csv', 'data/events.csv', flatten=True)
  t2 = threading.Thread(target=train_model, args=(
      deepcopy(flat_data), 'output/test_flat.onnx', 2**15), kwargs={'mode': 'flatten', 'tensor_prefix': 'flat'})
  t2.start()

  # logger.info('testing `train_model` with mode=ep and loss=mse')
  # t3 = threading.Thread(target=train_model, args=(
  #     deepcopy(data), 'output/test_mse_ep.onnx', 2**15), kwargs={'mode': 'ep', 'loss_fn': 'mse', 'tensor_prefix': 'mse-ep'})
  # t3.start()

  # logger.info('testing `train_model` with mode=flat and loss=mse')
  # t4 = threading.Thread(target=train_model, args=(
  #     deepcopy(flat_data), 'output/test_mse_flat.onnx', 2**15), kwargs={'mode': 'flatten', 'loss_fn': 'mse', 'tensor_prefix': 'mse-flat'})
  # t4.start()
  # train_model(data, 'output/test.onnx', 10,
  #             mode='flatten', tensor_prefix='flatten')
  # t1.join()
  t2.join()
  # t3.join()
  # t4.join()
