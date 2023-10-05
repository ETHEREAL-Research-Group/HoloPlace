test_results = {
  '1c54d7': {'test_loss': {'m': 2.9486621556274913e-05, 'ci': 1.0279208371984175e-05}, 'test_rot_loss': {'m': 0.014038697968470598, 'ci': 0.0006726985606622729}, 'test_pos_loss': {'m': 0.0076351204747589375, 'ci': 0.0005014724489653434}}, 
  'b3f9f8': {'test_loss': {'m': 1.429120877163771e-05, 'ci': 1.980530229201228e-06}, 'test_rot_loss': {'m': 0.011921223749461371, 'ci': 0.0004426182237028519}, 'test_pos_loss': {'m': 0.005746433086299776, 'ci': 0.00024531490776337136}}, 
  '40ad1b': {'test_loss': {'m': 0.0001952544713123738, 'ci': 0.0001564013936305512}, 'test_rot_loss': {'m': 0.02695531434253144, 'ci': 0.0021217403826505297}, 'test_pos_loss': {'m': 0.013738264544561804, 'ci': 0.0016305528919508382}}, 
  'a19cfd': {'test_loss': {'m': 5.5401393605683955e-05, 'ci': 4.68973738200008e-06}, 'test_rot_loss': {'m': 0.0247298132835163, 'ci': 0.001282338329322421}, 'test_pos_loss': {'m': 0.012217275367325378, 'ci': 0.0005732285786760102}}, 
  '8e5234': {'test_loss': {'m': 5.02122310436957e-05, 'ci': 6.6656690200476924e-06}, 'test_rot_loss': {'m': 0.02296169465388487, 'ci': 0.0016139952414629295}, 'test_pos_loss': {'m': 0.010731750455096148, 'ci': 0.0006850072651040807}}}
from utils.data import get_mean, get_tar_pos_stat, get_data_collection_time
from utils.trainer import get_acc
import numpy as np
from pprint import PrettyPrinter
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
  pp.pprint(get_tar_pos_stat(f'./data/{dir}'))
  collection_time = get_data_collection_time(f'./data/{dir}/events.csv')/1000
  mean_data_collection_time += collection_time
  print(f'data collection time = {collection_time:.2f}')

print(f'mean pos error = {mean_pos_err/len(dir_list)}')
print(f'mean rot error = {mean_rot_err/len(dir_list)}')
print(f'mean pos error naive = {mean_pos_err_naive/len(dir_list)}')
print(f'mean rot error naive = {mean_rot_err_naive/len(dir_list)}')
print(f'mean data collection time = {mean_data_collection_time/len(dir_list):.2f}')