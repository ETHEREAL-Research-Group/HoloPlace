import pandas as pd
from ast import literal_eval
import pytz
from imitation.data.types import Trajectory#, TransitionsMinimal
import numpy as np
from random import shuffle
import torch as th


def read_data(path='data/obs_acs.csv', torch_campatible=False, event_path='data/events.csv'):
  data = pd.read_csv(path)

  events = pd.read_csv(event_path)
  events['datetime'] = pd.to_datetime(
      events['timestamp'], unit='ms', utc=True).dt.tz_convert(pytz.timezone('US/Mountain'))
  last_timestamps = []
  for i in events[events['event'] == 'target_lost'].index:
    last_timestamps.append(events.iloc[i - 1].datetime)

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
  device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
  for idx, i in enumerate(dataset[dataset['act_pos_x'].isna()].index):
    if len(batches) == 0:
      mask = (dataset.index >= prev_idx) & (dataset.index <= i) & (dataset.index <= last_timestamps[idx])
    else:
      mask = (dataset.index > prev_idx) & (dataset.index <= i) & (dataset.index <= last_timestamps[idx])
    prev_idx = i
    temp = dataset[mask].copy()#.resample('0.1S').mean().fillna(method='pad')
    obs = temp.values[:, :7].astype(np.float32)
    acs = temp.values[:, 7:].astype(np.float32)
    batches.append(Trajectory(obs, acs[:-1, :], None, False))
    torch_batches.append((th.Tensor(obs).to(device), th.Tensor(acs).to(device)))
    # if torch_batches is None:
    #   torch_batches = {'obs': obs, 'acs': acs}
    # else:
    #   torch_batches['obs'] = np.append(torch_batches['obs'], obs, axis=0)
    #   torch_batches['acs'] = np.append(torch_batches['acs'], acs, axis=0)
    # torch_batches.append(TransitionsMinimal(
    #     obs[:-1, :], acs, np.zeros(shape=(acs.shape[0],))))

  if not torch_campatible:
    return batches
  else:
    shuffle(torch_batches)
    return torch_batches[0:int(len(torch_batches)*0.8)], torch_batches[int(len(torch_batches)*0.8):]


if __name__ == '__main__':
  print(read_data(torch_campatible=True))
