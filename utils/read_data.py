import pandas as pd
from ast import literal_eval
import pytz
from imitation.data.types import Trajectory  # , TransitionsMinimal
import numpy as np
from random import shuffle
import torch as th
import logging
from torch.utils.data import TensorDataset

logger = logging.getLogger()


def flatten_seq(seq):
  x = th.cat(tuple([x for (x, _y) in seq]))
  y = th.cat(tuple([y for (_x, y) in seq]))
  return TensorDataset(x, y)


def read_data(path='data/data.csv', torch_campatible=False, event_path='data/events.csv', flatten=True):
  data = pd.read_csv(path)

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
  torch_batches = []
  first_time = True
  device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
  for i in dataset[dataset['act_pos_x'].isna()].index:
    event_mask = (events['datetime'] >= prev_idx) & (
        events['datetime'] < i) & (events['event'] == 'Right IndexTip')
    if (len(events[event_mask]) == 0):
      logger.error(
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
    obs = temp.values[:, :7].astype(np.float32)
    acs = temp.values[:, 7:].astype(np.float32)
    if (not torch_campatible):
      batches.append(Trajectory(obs, acs[:-1, :], None, False))
    torch_batches.append(
        (th.Tensor(obs).to(device), th.Tensor(acs).to(device)))

  if not torch_campatible:
    shuffle(batches)
    return batches[0:int(len(batches)*0.8)]
  else:
    shuffle(torch_batches)
    train = torch_batches[0:int(len(torch_batches)*0.8)]
    val = torch_batches[int(len(torch_batches)*0.8):]
    if flatten:
      train = flatten_seq(train)
      val = flatten_seq(val)
    return train, val


if __name__ == '__main__':
  train, val = read_data(torch_campatible=True)
