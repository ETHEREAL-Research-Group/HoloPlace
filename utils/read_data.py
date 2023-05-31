import pandas as pd
from ast import literal_eval
import pytz
from imitation.data.types import Trajectory
import numpy as np
from random import shuffle
import torch as th
import logging
import json
import sys
# import datetime
from torch.utils.data import TensorDataset

logger = logging.getLogger()


def flatten_seq(seq):
  x = th.cat(tuple([x for (x, _y) in seq]))
  y = th.cat(tuple([y for (_x, y) in seq]))
  return TensorDataset(x, y)


def get_mean(data_path='data/data.csv', event_path='data/events.csv'):
  data = pd.read_csv(data_path)
  events = pd.read_csv(event_path)

  mask = events['event'] == 'Right IndexTip'
  events = events[mask]

  idx_list = []

  for _, row in events.iterrows():
    ts = row.timestamp
    closest = data.iloc[(data['timestamp'] - ts).abs().argsort()[:1]]
    idx_list.append(closest.index.values[0])

  data = data.iloc[idx_list]
  data.drop(['timestamp', 'act_pos', 'act_rot'], axis=1, inplace=True)
  data[data.columns] = data[data.columns].applymap(
      literal_eval, na_action='ignore')
  dataset = pd.DataFrame()

  for col in data.columns:
    col_data = data[col]
    dataset[f'{col}_x'] = col_data.apply(lambda x: x if not x == x else x[0])
    dataset[f'{col}_y'] = col_data.apply(lambda x: x if not x == x else x[1])
    dataset[f'{col}_z'] = col_data.apply(lambda x: x if not x == x else x[2])
    if col.endswith('rot'):
      dataset[f'{col}_w'] = col_data.apply(lambda x: x if not x == x else x[3])

  result = dict(dataset.mean())

  # do not change print to logger.info here!
  print(json.dump(result, sys.stderr))
  return result

# from scipy.spatial.transform import Rotation
# def get_matrix(flat: np.array):
#   pos = flat[:3]
#   rot = flat[3:]
#   rot_matrix = Rotation.from_quat(rot).as_matrix()
#   res = np.zeros(shape=(4, 4), dtype=np.float32)
#   res[:3, :3] = rot_matrix
#   res[:3, 3] = pos
#   res[3, 3] = 1
#   return res
# def destruct_matrix(matrix: np.array):
#   res = np.zeros(shape=(7,), dtype=np.float32)
#   res[:3] = matrix[:3, 3]
#   rot = Rotation.from_matrix(matrix[:3, :3]).as_quat()
#   res[3:] = rot
#   return res

# def transform(x):
#   hand = get_matrix(x[:7])
#   target = get_matrix(x[7:])
#   transformed = destruct_matrix(np.matmul(np.linalg.inv(target), hand))
#   return transformed

def read_data(data_path='data/data.csv', event_path='data/events.csv', custom_flatten=True, obs_size=14):
  data = pd.read_csv(data_path)

  events = pd.read_csv(event_path)
  # details = pd.read_csv(details_path, usecols=['timestamp', 'rif_pos', 'rif_rot', 'tar_pos','tar_rot'])
  # details['datetime'] = pd.to_datetime(
  #     details['timestamp'], unit='ms', utc=True).dt.tz_convert(pytz.timezone('US/Mountain'))
  # details.set_index(['datetime'], inplace=True)
  # details.drop(['timestamp'], axis=1, inplace=True)
  # details[details.columns] = details[details.columns].applymap(
  #     literal_eval, na_action='ignore')
  # details_ds = pd.DataFrame()

  # for col in details.columns:
  #   col_data = details[col]
  #   details_ds[f'{col}_x'] = col_data.apply(lambda x: x if not x == x else x[0]).interpolate('time', limit_direction='both')
  #   details_ds[f'{col}_y'] = col_data.apply(lambda x: x if not x == x else x[1]).interpolate('time', limit_direction='both')
  #   details_ds[f'{col}_z'] = col_data.apply(lambda x: x if not x == x else x[2]).interpolate('time', limit_direction='both')
  #   if col.endswith('rot'):
  #     details_ds[f'{col}_w'] = col_data.apply(lambda x: x if not x == x else x[3]).interpolate('time', limit_direction='both')
  events['datetime'] = pd.to_datetime(
      events['timestamp'], unit='ms', utc=True).dt.tz_convert(pytz.timezone('US/Mountain'))

  data['datetime'] = pd.to_datetime(
      data['timestamp'], unit='ms', utc=True).dt.tz_convert(pytz.timezone('US/Mountain'))
  data.set_index(['datetime'], inplace=True)
  data.drop(['timestamp'], axis=1, inplace=True)
  data[data.columns] = data[data.columns].applymap(
      literal_eval, na_action='ignore')
  dataset = pd.DataFrame()

  for col in data.columns:
    col_data = data[col]
    dataset[f'{col}_x'] = col_data.apply(lambda x: x if not x == x else x[0])
    dataset[f'{col}_y'] = col_data.apply(lambda x: x if not x == x else x[1])
    dataset[f'{col}_z'] = col_data.apply(lambda x: x if not x == x else x[2])
    if col.endswith('rot'):
      dataset[f'{col}_w'] = col_data.apply(lambda x: x if not x == x else x[3])

  prev_idx = dataset.index[0]
  batches = []
  custom_batches = []
  first_time = True
  device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
  for i in dataset[dataset['act_pos_x'].isna()].index:
    event_mask = (events['datetime'] >= prev_idx) & (
        events['datetime'] < i) & (events['event'] == 'Right IndexTip')
    if (len(events[event_mask]) == 0):
      logger.warning(
          'could not find the last touch by right index finger. skipping this episode...')
      first_time = False
      prev_idx = i
      continue
    last_touch = events[event_mask].iloc[-1].datetime
    if first_time:
      mask = (dataset.index >= prev_idx) & (dataset.index < i) & (
          dataset.index <= last_touch)
      first_time = False
    else:
      mask = (dataset.index > prev_idx) & (dataset.index < i) & (
          dataset.index <= last_touch)
    prev_idx = i
    temp = dataset[mask].copy()
    # details_mask = (details_ds.index >= temp.index[0] - datetime.timedelta(milliseconds=10)) & (details_ds.index <= temp.index[-1] + datetime.timedelta(milliseconds=10))
    # details_temp = details_ds[details_mask].copy()
    # try:
    #   details_temp = details_temp.values[:len(temp)]
    # except:
    #   logger.warning('sorry, because of poor coding, we have to skip this episode...')
    #   continue

    # details_temp = np.apply_along_axis(transform, 1, details_temp)
    # if True:
    #   obs = np.append(temp.values[:, :7], details_temp, axis=1).astype(np.float32)
    # else:
    #   obs = temp.values[:, :7].astype(np.float32)
    obs = temp.values[:, :obs_size].astype(np.float32)
    acs = temp.values[:, obs_size:].astype(np.float32)
    if len(acs[:-1, :]) == 0:
      logger.warning('need at least one action. skipping this episode...')
      continue
    # if len(acs[:-1, :]) <= 40:
    #   logger.warning('episode is too short to be included in the training. skipping this episode...')
    #   continue

    batches.append(Trajectory(obs, acs[:-1, :], None, True))
    custom_batches.append(
        (th.Tensor(obs[:-1]).to(device), th.Tensor(obs[1:,:7]).to(device)))

  temp = list(zip(batches, custom_batches))
  shuffle(temp)
  batches, custom_batches = zip(*temp)

  custom_train = custom_batches[0:int(len(custom_batches)*0.99)]
  custom_val = custom_batches[int(len(custom_batches)*0.99):]
  if custom_flatten:
    custom_train = flatten_seq(custom_train)
    custom_val = flatten_seq(custom_val)

  train = batches[0:int(len(batches)*0.99)]
  val = batches[int(len(batches)*0.99):]

  return (train, val), (custom_train, custom_val)


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()
  data, custom_data = read_data()
  train, val = data
  custom_train, custom_val = custom_data
  # get_mean()
