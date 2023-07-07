# This is for if you have data in the data folder
if __name__ == "__main__":
  import logging
  import os
  import matplotlib.pyplot as plt
  from time import time as epoch_time
  # import threading
  logging.basicConfig(
      format='%(threadName)s %(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()

  from utils.data import read_data
  from utils.trainer import train_model

  base_path = './data'
  dir_list = os.listdir(base_path)
  logger.info(f'users: {dir_list}')
  # threads = []
  test_results = {}
  elapsed_times = []
  for dir in dir_list:
    logger.info(f'processing user {dir}')
    data_path = os.path.join(base_path, dir, 'data.csv')
    events_path = os.path.join(base_path, dir, 'events.csv')
    output_path = os.path.join(base_path, dir, 'output', 'model.onnx')

    start_time = epoch_time()
    data = read_data(data_path, events_path)
    test_results[dir] = train_model(data, output_path, 4096)
    elapsed_time = epoch_time() - start_time
    logger.info(test_results[dir])
    logger.info(f'elapsed time for user {dir} was {elapsed_time}')
    elapsed_times.append(elapsed_time)

    # t = threading.Thread(target=train_model, args=(data, output_path, 2**2), name=dir)
    # threads.append(t)
    # t.start()

  # for t in threads:
  #   t.join()

  logger.info(elapsed_times)
  logger.info(test_results)

  logger.info(
      f'average training time was {sum(elapsed_times)/len(elapsed_times)}')

  fig = plt.figure(figsize=(10, 5), dpi=300)
  ax = plt.subplot(111)

  ax.set_xticks([i for i in range(len(dir_list))])
  ax.set_xticklabels([i for i in dir_list])
  ax.set_ylabel('Geodesic Loss in Radians')
  ax.set_xlabel('Subject')
  ax.errorbar([i for i in range(len(dir_list))], [test_results[i]['test_rot_loss']['m'] for i in dir_list], yerr=[
              test_results[i]['test_rot_loss']['ci'] for i in dir_list], marker="o", capsize=2, markersize=4, ls='none')
  plt.savefig('rot_loss.png', transparent=True)

  fig = plt.figure(figsize=(10, 5), dpi=300)
  ax = plt.subplot(111)

  ax.set_xticks([i for i in range(len(dir_list))])
  ax.set_xticklabels([i for i in dir_list])
  ax.set_ylabel('L2 Loss in m')
  ax.set_xlabel('Subject')
  ax.errorbar([i for i in range(len(dir_list))], [test_results[i]['test_pos_loss']['m'] for i in dir_list], yerr=[
              test_results[i]['test_pos_loss']['ci'] for i in dir_list], marker="o", capsize=2, markersize=4, ls='none')
  plt.savefig('pos_loss.png', transparent=True)
