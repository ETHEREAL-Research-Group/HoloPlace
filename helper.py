# This is a bad code! test_result is copied from the output of main.py
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pprint import PrettyPrinter
import numpy as np
from utils.trainer import get_acc
from utils.data import get_mean, get_tar_pos_stat, get_data_collection_time
test_results = {
    '1c54d7': {'test_loss': {'m': 2.9486621556274913e-05, 'ci': 1.0279208371984175e-05}, 'test_rot_loss': {'m': 0.014038697968470598, 'ci': 0.0006726985606622729}, 'test_pos_loss': {'m': 0.0076351204747589375, 'ci': 0.0005014724489653434}},
    'b3f9f8': {'test_loss': {'m': 1.429120877163771e-05, 'ci': 1.980530229201228e-06}, 'test_rot_loss': {'m': 0.011921223749461371, 'ci': 0.0004426182237028519}, 'test_pos_loss': {'m': 0.005746433086299776, 'ci': 0.00024531490776337136}},
    '40ad1b': {'test_loss': {'m': 0.0001952544713123738, 'ci': 0.0001564013936305512}, 'test_rot_loss': {'m': 0.02695531434253144, 'ci': 0.0021217403826505297}, 'test_pos_loss': {'m': 0.013738264544561804, 'ci': 0.0016305528919508382}},
    'a19cfd': {'test_loss': {'m': 5.5401393605683955e-05, 'ci': 4.68973738200008e-06}, 'test_rot_loss': {'m': 0.0247298132835163, 'ci': 0.001282338329322421}, 'test_pos_loss': {'m': 0.012217275367325378, 'ci': 0.0005732285786760102}},
    '8e5234': {'test_loss': {'m': 5.02122310436957e-05, 'ci': 6.6656690200476924e-06}, 'test_rot_loss': {'m': 0.02296169465388487, 'ci': 0.0016139952414629295}, 'test_pos_loss': {'m': 0.010731750455096148, 'ci': 0.0006850072651040807}}}

pp = PrettyPrinter(indent=4)
dir_list = ['1c54d7', 'b3f9f8', '40ad1b', 'a19cfd', '8e5234']
acc_results = {}
mean_pos_err = 0
mean_pos_err_naive = 0
mean_rot_err = 0
mean_rot_err_naive = 0
mean_data_collection_time = 0
for i, dir in enumerate(dir_list):
  print(f'processing {dir}...')
  naive_mean = get_mean(f'./data/{dir}/data.csv', f'./data/{dir}/events.csv')
  true = np.load(f'./data/{dir}/output/true.npy')
  acc = get_acc(true, naive_mean, test_results[dir])
  print('naive results:')
  pp.pprint(acc['naive'])
  mean_pos_err_naive += acc['naive']['pos_loss']['m']
  mean_rot_err_naive += acc['naive']['rot_loss']['m']
  print('BC results')
  pp.pprint(acc['BC'])
  mean_pos_err += acc['BC']['pos_loss']['m']
  mean_rot_err += acc['BC']['rot_loss']['m']
  print('tar pos stat:')
  tar_pos_stat, tar_pos_dataset = get_tar_pos_stat(f'./data/{dir}')
  pp.pprint(tar_pos_stat)
  collection_time = get_data_collection_time(f'./data/{dir}/events.csv')/1000
  mean_data_collection_time += collection_time
  print(f'data collection time = {collection_time:.2f}')

  mpl.rcParams['font.size'] = 9
  fig = plt.figure(figsize=(4, 3), dpi=300)
  true = np.load(f'./data/{dir}/output/true.npy')
  pred = np.load(f'./data/{dir}/output/pred.npy')

  all_data = np.concatenate((true, pred, naive_mean), axis=0)

  # Perform t-SNE
  tsne = TSNE(n_components=2, random_state=0)
  all_data_2d = tsne.fit_transform(all_data)

  # Separate the transformed true and predicted data
  true_data_2d = all_data_2d[:true.shape[0]-1]
  pred_data_2d = all_data_2d[true.shape[0]:-1]
  naive_data_2d = all_data_2d[-1:]

  # Sampling frequency
  freq = 50
  freq = (50 // freq)
  true_data_2d = true_data_2d[::freq, :]
  pred_data_2d = pred_data_2d[::freq, :]

  # Scatter Plot the data
  plt.scatter(true_data_2d[:, 0], true_data_2d[:, 1],
              color='b', marker='.', alpha=0.8, label='Expert')
  plt.scatter(pred_data_2d[:, 0], pred_data_2d[:, 1], edgecolors='r',
              marker='s', alpha=0.6, facecolors='none', label='BC')
  plt.scatter(naive_data_2d[:, 0], naive_data_2d[:, 1],
              color='g', s=150, marker='*', label='Baseline')

  plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

  plt.legend()
  plt.xticks([])
  plt.yticks([])
  plt.savefig(f'{i+1}-tsne.png', transparent=True)

  fig = plt.figure(dpi=300)
  ax = fig.add_subplot(111, projection='3d', box_aspect=[1,1,1])
  true = tar_pos_dataset.values

  freq = 50
  freq = (50 // freq)
  true = true[::freq]
  ax.scatter(true[:, 0]*100, true[:, 2]*100, true[:, 1]*100, edgecolors='r', facecolors='none', label='Target', alpha=0.6)
  # ax.legend()
  ax.set_xlabel('X (cm)')
  ax.set_ylabel('Z (cm)')
  ax.set_zlabel('Y (cm)')
  x_min = min(true[:, 0]*100)
  x_max = max(true[:, 0]*100)
  x_mid = (x_min + x_max)/2
  x_range = 372
  x_lower_bound = round(x_mid - (x_range/2))
  x_higher_bound = round(x_mid + (x_range/2))

  y_min = min(true[:, 2]*100)
  y_max = max(true[:, 2]*100)
  y_mid = (y_min + y_max)/2
  y_range = 395
  y_lower_bound = round(y_mid - (y_range/2))
  y_higher_bound = round(y_mid + (y_range/2))

  z_min = min(true[:, 1]*100)
  z_max = max(true[:, 1]*100)
  z_mid = (z_min + z_max)/2
  z_range = 130
  z_lower_bound = round(z_mid - (z_range/2))
  z_higher_bound = round(z_mid + (z_range/2))

  # Limit ticks to min, mid and max
  ax.set_xlim([x_lower_bound, x_higher_bound])
  ax.set_xticks([x_lower_bound, round(x_mid), x_higher_bound])

  ax.set_yticks([y_lower_bound, round(y_mid), y_higher_bound])
  ax.set_ylim([y_lower_bound, y_higher_bound])
  
  ax.set_zticks([z_lower_bound, round(z_mid), z_higher_bound])
  ax.set_zlim([z_lower_bound, z_higher_bound])
  plt.savefig(f'{i+1}-movement.png', transparent=True)

print(f'mean pos error = {mean_pos_err/len(dir_list)}')
print(f'mean rot error = {mean_rot_err/len(dir_list)}')
print(f'mean pos error naive = {mean_pos_err_naive/len(dir_list)}')
print(f'mean rot error naive = {mean_rot_err_naive/len(dir_list)}')
print(
    f'mean data collection time = {mean_data_collection_time/len(dir_list):.2f}')
