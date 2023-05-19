if __name__ == "__main__":
  # on linux, by default it's fork -- we change it to spawn for some reason
  from multiprocessing import set_start_method
  set_start_method('spawn')
  import logging
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  logger = logging.getLogger()

  import argparse
  from utils.read_data import read_data
  from utils.imitation_helper import imitation_trainer, custom_trainer

  parser = argparse.ArgumentParser()
  parser.add_argument('data_path', default='data/data.csv', const=1, nargs='?')
  parser.add_argument(
      'events_path', default='data/events.csv', const=1, nargs='?')
  parser.add_argument(
      'output_path', default='output/model.onnx', const=1, nargs='?')
  parser.add_argument(
      'method', default='gail', const=1, nargs='?')  # options = custom, gail, bc
  parser.add_argument(
      'epochs', default='200', const=1, nargs='?')
  args = parser.parse_args()

  logger.info('reading the data...')
  data, custom_data = read_data(args.data_path, args.events_path)

  if args.method in ['gail', 'bc']:
    train, _val = data
    imitation_trainer(train, args.output_path, args.method, int(args.epochs))
  elif args.method == 'custom':
    custom_trainer(custom_data, args.output_path, int(args.epochs)*50)
  else:
    raise Exception('method not supported')
