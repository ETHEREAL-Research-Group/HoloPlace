import threading
import torch as th
import logging
import math
import scipy.stats as st
# import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
logger = logging.getLogger()

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


def position_distance(points1, points2):
  positions1 = points1[:, :3]
  positions2 = points2[:, :3]
  res = th.norm(positions1 - positions2, dim=1, p=2, keepdim=True)
  if (res.isnan().any()):
    logger.error('HM.. something went wrong')
  return th.mean(res)


epsilon = th.finfo(th.float32).eps


def normalize_quaternions(quaternions: th.Tensor):
  # Normalize the quaternions to unit length
  quaternions_norm = th.norm(quaternions, dim=1, p=2, keepdim=True)
  # zero_mask = quaternions_norm == 0.0
  # quaternions_norm += epsilon * zero_mask
  # so that we don't get division by zero
  normalized = quaternions / (quaternions_norm+epsilon)
  # identity_quat = th.tensor(
  #     [[0.0, 0.0, 0.0, 1.0]]).expand(normalized.shape[0], -1)
  # nan_mask = th.isnan(normalized)
  # if (nan_mask.any()):
  #   logger.warning('all zero quaternions...')
  # normalized[nan_mask] = identity_quat[nan_mask]

  # if (th.isnan(normalized).any()):
  #   logger.error('all zero quaternions...')
  #   raise Exception()

  return normalized


def wrap_angle(theta):
  """
  Helper method: Wrap the angles of the input tensor to lie between -pi and pi.
  Odd multiples of pi are wrapped to +pi (as opposed to -pi).
  """
  pi_tensor = th.ones_like(theta) * math.pi
  result = ((theta + pi_tensor) % (2 * pi_tensor)) - pi_tensor
  result[result.eq(-pi_tensor)] = math.pi

  return result


def geodesic_loss(points1, points2):
  q1 = normalize_quaternions(points1[:, -4:])
  q2 = normalize_quaternions(points2[:, -4:])
  dot_product = th.clip(th.sum(q1 * q2, dim=-1), min=-1, max=1)
  angle = th.abs(th.acos(2 * dot_product**2 - 1))
  return th.mean(angle)

  # diff = q_mul(q_conjugate(quaternions1), quaternions2)
  # Rotation.from_quat(quaternions1).
  # x = th.abs(wrap_angle(th.Tensor(Rotation.from_quat(
  #     diff.detach().numpy()).as_euler('xyz', degrees=False))))

  # return x.mean()


def q_mul(q1, q2):
  """
  Multiply quaternion q1 with q2.
  Expects two equally-sized tensors of shape [*, 4], where * denotes any number of dimensions.
  Returns q1*q2 as a tensor of shape [*, 4].
  """
  assert q1.shape[-1] == 4
  assert q2.shape[-1] == 4
  original_shape = q1.shape

  # Compute outer product
  terms = th.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
  w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
  x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
  y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
  z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

  return th.stack((w, x, y, z), dim=1).view(original_shape)


def q_conjugate(q):
  """
  Returns the complex conjugate of the input quaternion tensor of shape [*, 4].
  """
  assert q.shape[-1] == 4

  # multiplication coefficients per element
  conj = th.tensor([1, -1, -1, -1], device=q.device)
  return q * conj.expand_as(q)


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

    def forward(self, x):
      position = x[:, :3]
      quaternion = x[:, -4:]

      # Normalize the quaternion part
      quaternion_norm = th.norm(quaternion, p=2, dim=-1, keepdim=True)
      normalized = quaternion / (quaternion_norm+epsilon)
      normalized_x = th.cat([position, normalized], dim=-1)
      return normalized_x

  input_size = 21
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
  train, val, test = data

  train = DataLoader(train, batch_size=batch_size, shuffle=True)
  val = DataLoader(val, batch_size=batch_size, shuffle=True)
  test = DataLoader(test, batch_size=1, shuffle=True)
  return train, val, test


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
      th.ones([1, 21], dtype=th.float32, device=device),
      path,
      opset_version=9,
      input_names=['input'],
      output_names=['output']
  )
  export.lock.release()  # type: ignore


def train_model(data, output_path, num_epochs=4096, loss_fn='mse', mode='flatten', tensor_prefix='default', batches_per_epoch=1):
  import torch.optim as optim
  from torch.utils.tensorboard.writer import SummaryWriter
  from pathlib import Path
  import os
  tensor_path = Path(output_path).parent.joinpath('summary', tensor_prefix)
  os.makedirs(tensor_path, exist_ok=True)
  writer = SummaryWriter(tensor_path)

  patience = 512+1
  if loss_fn == 'mse':
    min_delta = 5e-6
  else:
    min_delta = 5e-3

  model = get_model()

  # criterion = th.nn.MSELoss()
  if loss_fn == 'chamfer':
    criterion = chamfer_distance
  elif loss_fn == 'mse':
    criterion = th.nn.MSELoss()
  else:
    raise Exception('invalid argument for loss')

  optimizer = optim.Adam(model.parameters(), lr=1e-3,
                         weight_decay=1e-5)  # , eps=1e-5)
  scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=128)

  if mode == 'flatten':
    data = prep_data(data)

  train, val, test = data

  best_loss = float('inf')
  counter = 0

  for epoch in range(num_epochs):
    model.train()
    # pbar = tqdm(desc=f'Training Epoch {epoch}', total=len(train))
    train_losses = []
    rot_diffs = []
    pos_diffs = []
    for _ in range(batches_per_epoch):
      if mode == 'ep':
        shuffle(train)
      for _, (x_batch, y_batch) in enumerate(train):
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        train_losses.append(loss.item())
        rot_diff = geodesic_loss(outputs, y_batch)
        pos_diff = position_distance(outputs, y_batch)
        rot_diffs.append(rot_diff.item())
        pos_diffs.append(pos_diff.item())

        optimizer.step()
      # pbar.update(1)
    mean_train_loss = sum(train_losses)/len(train_losses)
    mean_rot_diff = sum(rot_diffs)/len(rot_diffs)
    mean_pos_diff = sum(pos_diffs)/len(pos_diffs)
    writer.add_scalar(f'Loss/Train', mean_train_loss, epoch)
    writer.add_scalar(f'Rot_Loss/Train', mean_rot_diff, epoch)
    writer.add_scalar(f'Pos_loss/Train', mean_pos_diff, epoch)
    model.eval()
    val_losses = []
    val_rot_diffs = []
    val_pos_diffs = []
    for _, (x_batch, y_batch) in enumerate(val):
      outputs = model(x_batch)
      val_loss = criterion(outputs, y_batch)
      val_losses.append(val_loss.item())

      rot_diff = geodesic_loss(outputs, y_batch)
      val_rot_diffs.append(rot_diff.item())

      pos_diff = position_distance(outputs, y_batch)
      val_pos_diffs.append(pos_diff.item())
    mean_val_loss = sum(val_losses) / len(val_losses)
    mean_val_rot_diff = sum(val_rot_diffs) / len(val_rot_diffs) if len(val_rot_diffs) > 0 else float('inf')
    mean_val_pos_diff = sum(
        val_pos_diffs) / len(val_pos_diffs) if len(val_pos_diffs) > 0 else float('inf')
    writer.add_scalar(f'Loss/Val', mean_val_loss, epoch)
    writer.add_scalar(f'Rot_Loss/Val', mean_val_rot_diff, epoch)
    writer.add_scalar(f'Pos_loss/Val', mean_val_pos_diff, epoch)

    # pbar.set_postfix({
    #     'Train Loss': train_loss,
    #     'Avg Val Loss': mean_val_loss,
    #     'learning rate': optimizer.param_groups[0]['lr'],
    #     'Best': 'True' if mean_val_loss < best_val_loss else 'False'
    # })
    # pbar.close()

    best = (mean_val_loss + min_delta) < best_loss
    if best:
      counter = 0

    logger.info(
        f'epoch {epoch:>5d}/{num_epochs} {(epoch*100)//num_epochs:>3d}%: pos_loss={mean_pos_diff:.4e}, rot_loss={mean_rot_diff:.4e}, val_pos_loss={mean_val_pos_diff:.4e}, val_rot_loss={mean_val_rot_diff:.4e}, train_loss={mean_train_loss:.4e}, val_loss={mean_val_loss:.4e}, lr={optimizer.param_groups[0]["lr"]:.1e}, best={best}')

    if mean_val_loss < best_loss:
      logger.info('saving the best model so far...')
      best_loss = mean_val_loss
      th.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'val_loss': mean_val_loss
      }, f'{output_path[:-4]}pt')
    else:
      counter += 1

    if epoch % 100 == 0:
      export(model, output_path[:-5] + f'-{epoch}.onnx')

    if counter > patience:
      logger.info('stopping training early...')
      break
    scheduler.step()

  try:
    logger.info('loading the best model...')
    checkpoint = th.load(f'{output_path[:-4]}pt')
    model.load_state_dict(checkpoint['model_state_dict'])
  except:
    logger.error(f'{output_path[:-4]}pt could not be found...')

  logger.info('testing the model...')

  model.eval()
  test_losses = []
  test_rot_diffs = []
  test_pos_diffs = []
  i = 0

  true_list = []
  pred_list = []

  for x, y in test:
    pred = model(x)
    true_list.append(y.detach().cpu().numpy()[0])
    pred_list.append(pred.detach().cpu().numpy()[0])

    test_loss = criterion(pred, y)
    
    i += 1
    test_losses.append(test_loss.item())

    rot_diff = geodesic_loss(pred, y)
    test_rot_diffs.append(rot_diff.item())

    pos_diff = position_distance(pred, y)
    test_pos_diffs.append(pos_diff.item())

  np.save(f'{output_path[:-10]}true.npy', np.array(true_list))
  np.save(f'{output_path[:-10]}pred.npy', np.array(pred_list))

  rot_intervals = st.t.interval(0.95, len(test_rot_diffs)-1, loc=th.mean(
      th.Tensor(test_rot_diffs)), scale=st.sem(th.Tensor(test_rot_diffs)))
  mean_test_rot_diff = sum(
      test_rot_diffs) / len(test_rot_diffs) if len(test_rot_diffs) > 0 else float('inf')
  loss_intervals = st.t.interval(0.95, len(
      test_losses)-1, loc=th.mean(th.Tensor(test_losses)), scale=st.sem(th.Tensor(test_losses)))
  mean_test_loss = sum(test_losses) / \
      len(test_losses) if len(test_losses) > 0 else float('inf')
  pos_intervals = st.t.interval(0.95, len(test_pos_diffs)-1, loc=th.mean(
      th.Tensor(test_pos_diffs)), scale=st.sem(th.Tensor(test_pos_diffs)))
  mean_test_pos_diff = sum(
      test_pos_diffs) / len(test_pos_diffs) if len(test_pos_diffs) > 0 else float('inf')
  # writer.add_embedding(th.Tensor(test_losses).expand(len(test_losses), -1), tag="Test/Loss")
  # writer.add_embedding(th.Tensor(test_rot_diffs).expand(len(test_losses), -1), tag="Test/Rot_Loss")
  # writer.add_embedding(th.Tensor(test_pos_diffs).expand(len(test_losses), -1), tag="Test/Pos_Loss")

  # writer.add_scalar(f'Test/Loss', mean_test_loss)
  # writer.add_scalar(f'Test/Rot_Loss', mean_test_rot_diff)
  # writer.add_scalar(f'Test/Pos_loss', mean_test_pos_diff)

  # writer.add_scalar(f'Test/Loss_CI', loss_intervals[1] - mean_test_loss)
  # writer.add_scalar(f'Test/Rot_Loss_CI', rot_intervals[1] - mean_test_rot_diff)
  # writer.add_scalar(f'Test/Pos_loss_CI', pos_intervals[1] - mean_test_pos_diff)

  # fig = plt.figure(dpi=300)
  # ax = plt.subplot(111)

  # ax.set_xticks([1, 2, 3])
  # ax.set_xticklabels(['Pos', 'Rot', 'Loss'])
  # ax.errorbar([1, 2, 3], [mean_test_pos_diff, mean_test_rot_diff, mean_test_loss], yerr=[pos_intervals[1]-mean_test_pos_diff,
  #             rot_intervals[1]-mean_test_rot_diff, loss_intervals[1]-mean_test_loss], marker="o", capsize=2, markersize=4, ls='none')

  # writer.add_figure('Fig/Test', fig)

  export(model, output_path)

  end_result = {'test_loss': {'m': mean_test_loss, 'ci': loss_intervals[1] - mean_test_loss}, 'test_rot_loss': {
      'm': mean_test_rot_diff, 'ci': rot_intervals[1] - mean_test_rot_diff}, 'test_pos_loss': {'m': mean_test_pos_diff, 'ci': pos_intervals[1] - mean_test_pos_diff}}
  logger.info('end result:')
  logger.info(end_result)
  logger.info('training finished')
  writer.close()
  return end_result


if __name__ == '__main__':
  logging.basicConfig(
      format='%(threadName)s %(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()
  from data import read_data
  from copy import deepcopy
  import threading
  # Testing custom nn:
  # logger.info('testing `train_model` with mode=flatten and loss=chamfer')
  data = read_data('data/data.csv', 'data/events.csv', flatten=True)
  # t1 = threading.Thread(target=train_model, args=(
  #     deepcopy(data), 'output/test_chamfer.onnx', 2**15), kwargs={'mode': 'flatten', 'loss_fn': 'chamfer', 'tensor_prefix': 'chamfer'})
  # t1.start()

  # train_model(data, 'output/test.onnx', 10, tensor_prefix='ep')
  logger.info('testing `train_model` with mode=flatten and loss=mse')
  # flat_data = read_data('data/data.csv', 'data/events.csv', flatten=True)
  t2 = threading.Thread(target=train_model, args=(
      deepcopy(data), 'output/test_mse.onnx', 1), kwargs={'mode': 'flatten', 'loss_fn': 'mse', 'tensor_prefix': 'mse'})
  t2.start()

  # t1.join()
  t2.join()
