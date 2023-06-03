if __name__ == "__main__":
  # on linux, by default it's fork -- we change it to spawn for some reason
  from multiprocessing import set_start_method
  set_start_method('spawn')
  import logging
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()

  import argparse
  from utils.data import read_data, get_mean
  from utils.trainer import train_model

  parser = argparse.ArgumentParser()
  parser.add_argument('data_path', default='data/data.csv', const=1, nargs='?')
  parser.add_argument(
      'events_path', default='data/events.csv', const=1, nargs='?')
  parser.add_argument(
      'output_path', default='output/model.onnx', const=1, nargs='?')
  parser.add_argument(
      'method', default='custom', const=1, nargs='?')  # options = custom, naive
  parser.add_argument(
      'epochs', default='200', const=1, nargs='?')
  parser.add_argument(
      'mode', default='flatten', const=1, nargs='?')  # options = flatten, naive
  args = parser.parse_args()

  if args.method == 'custom':
    logger.info('reading the data...')
    data = read_data(args.data_path, args.events_path, flatten=args.mode == 'flatten')
    train_model(data, args.output_path, int(args.epochs), mode=args.mode)
  elif args.method == 'naive':
    get_mean(args.data_path, args.events_path)
  else:
    raise Exception('method not supported')
