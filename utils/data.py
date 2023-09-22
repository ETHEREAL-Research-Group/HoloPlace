import pandas as pd
from ast import literal_eval
import pytz
import numpy as np
from random import shuffle
import torch as th
import logging
import json
import sys
from torch.utils.data import TensorDataset


def flatten_seq(seq, make_ds=False):
  x = th.cat(tuple([x for (x, _y) in seq]))
  y = th.cat(tuple([y for (_x, y) in seq]))
  if make_ds:
    return TensorDataset(x, y)
  return x, y


def get_mean(data_path, event_path):
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
  data.drop(['timestamp', 'act_pos', 'act_rot', 'rif_pos',
            'rif_rot', 'rpa_pos', 'rpa_rot'], axis=1, inplace=True)
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
  return th.Tensor(np.expand_dims(dataset.mean().values, 0))


def normalize_quaternions(quaternions):
  # Normalize the quaternions to unit length
  quaternions_norm = th.norm(quaternions, dim=1, p=2, keepdim=True)

  return quaternions / quaternions_norm


def read_data(data_path, event_path, obs_size=21, acs_size=7, flatten=True, shuffle_tensors=True, split_ratio=[0.7, 0.15, 0.15]):
  logger = logging.getLogger()
  data = pd.read_csv(data_path)

  events = pd.read_csv(event_path)
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
    obs = temp.values[:, :obs_size].astype(np.float32)
    acs = temp.values[:, -acs_size:].astype(np.float32)
    if len(acs[:-1, :]) < 1:
      logger.warning(
          'need at least 1 datapoints for each episode. skipping this episode...')
      continue

    batches.append((th.Tensor(obs[:-1]).to(device),
                   th.Tensor(obs[1:, :acs_size]).to(device)))
  if len(batches) == 0:
    logger.error(
        'the batch list is empty. this is probably due to the system not detecting collision events...')
    # raise Exception('ERR_DATA')
    logger.warning(
        'the system will continue to train the model but the usability aspect may be affected. setting the minimum datapoint in an episod to 150 and excluding the last 100...')
    prev_idx = dataset.index[0]
    print(prev_idx)
    batches = []
    first_time = True
    for i in dataset[dataset['act_pos_x'].isna()].index:
      if first_time:
        mask = (dataset.index >= prev_idx) & (dataset.index < i)
        first_time = False
      else:
        mask = (dataset.index > prev_idx) & (dataset.index < i)
      temp = dataset[mask].copy()
      obs = temp.values[:, :obs_size].astype(np.float32)
      acs = temp.values[:, -acs_size:].astype(np.float32)
      if len(acs[:-1, :]) < 150:
        logger.warning(
            'need at least 150 datapoints for each episode. skipping this episode...')
        continue
      obs = obs[:-100]
      batches.append(
          (th.Tensor(obs[:-1]).to(device), th.Tensor(obs[1:, :acs_size]).to(device)))

  shuffle(batches)
  if flatten:
    x, y = flatten_seq(batches)
    y[:, -4:] = normalize_quaternions(y[:, -4:])
    size = len(x)
    if shuffle_tensors:
      indices = th.randperm(size)
      x = x[indices]
      y = y[indices]
    train_idx = int(size*split_ratio[0])
    val_idx = int(size*(split_ratio[0] + split_ratio[1]))
    train = TensorDataset(x[0:train_idx], y[0:train_idx])
    val = TensorDataset(x[train_idx:val_idx], y[train_idx:val_idx])
    test = TensorDataset(x[val_idx:], y[val_idx:])
  else:
    size = len(batches)
    train_idx = int(size*split_ratio[0])
    val_idx = int(size*(split_ratio[0] + split_ratio[1]))
    train = batches[0:train_idx]
    val = batches[train_idx:val_idx]
    test = batches[val_idx:]

  logger.info(
      f'train samples: {len(train)} -- val samples: {len(val)} -- test samples: {len(test)}')
  return (train, val, test)


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()
  logger.info('testing `read_data()`...')
  read_data('data/data.csv', 'data/events.csv', flatten=True)
  logger.info('testing `get_mean()`...')
  get_mean('data/data.csv', 'data/events.csv')
