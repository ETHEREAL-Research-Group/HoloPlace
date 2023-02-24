import pandas as pd
from ast import literal_eval
import pytz
from imitation.data.types import Trajectory, TransitionsMinimal
import numpy as np


def read_data(path='data/obs_acs.csv', torch_campatible=False):
  data = pd.read_csv(path)
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
  for i in dataset[dataset['act_pos_x'].isna()].index:
    if len(batches) == 0:
      mask = (dataset.index >= prev_idx) & (dataset.index <= i)
    else:
      mask = (dataset.index > prev_idx) & (dataset.index <= i)
    prev_idx = i
    obs = dataset[mask].values[:, :7].astype(np.float32)
    acs = dataset[mask].values[:, 7:].astype(np.float32)[:-1, :]
    batches.append(Trajectory(obs, acs, None, False))
    torch_batches.append(TransitionsMinimal(
        obs[:-1, :], acs, np.zeros(shape=(acs.shape[0],))))
  if not torch_campatible:
    return batches
  else:
    return torch_batches


if __name__ == '__main__':
  print(read_data('data/obs_acs.csv'))
