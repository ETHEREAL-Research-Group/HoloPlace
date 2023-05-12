if __name__ == '__main__':
  import logging
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
  from multiprocessing import set_start_method#, cpu_count
  set_start_method('spawn')
  import torch as th
  from tqdm import tqdm
  import torch.nn as nn
  import torch.optim as optim
  from utils.read_data import read_data
  from torch.utils.data import DataLoader, TensorDataset, Subset #,random_split
  
  logger = logging.getLogger()
  # Define hyperparameters
  batch_size = 500
  num_epochs = 20000
  input_size = 7
  output_size = 7
  patience = float('inf')
  min_delta = 1e-7
  device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
  logger.info('creating the model...')
  model = nn.Sequential(
      nn.Linear(input_size, 64),
      nn.Tanh(),
      # nn.Dropout(0.05),
      nn.Linear(64, 64),
      nn.Tanh(),
      # nn.Dropout(0.05),
      nn.Linear(64, 64),
      nn.Tanh(),
      # nn.Dropout(0.05),
      nn.Linear(64, 64),
      nn.Tanh(),
      # nn.Dropout(0.05),
      nn.Linear(64, output_size),
  ).to(device)

  # Define the loss function and optimizer
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
  scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=30)

  # Generate some sample data
  logger.info('reading the data...')
  data = read_data(torch_campatible=True)
  data_in = th.Tensor(data['obs']).to(device)
  data_out = th.Tensor(data['acs']).to(device)

  dataset = TensorDataset(data_in, data_out)
  # train, val = random_split(dataset, [0.8, 0.2], th.Generator().manual_seed(42))
  train = Subset(dataset, range(int(len(dataset) * 0.95)))
  val = Subset(dataset, range(int(len(dataset) * 0.95), len(dataset)))

  trainLoader = DataLoader(train, batch_size=batch_size, shuffle=False)
  valLoader = DataLoader(val, batch_size=batch_size, shuffle=False)

  # Train the model
  best_val_loss = float('inf')
  counter = 0
  for epoch in tqdm(range(num_epochs)):
    model.train()
    pbar = tqdm(desc=f'Training Epoch {epoch}', total=len(trainLoader))
    train_loss = None
    for id_batch, (x_batch, y_batch) in enumerate(trainLoader):
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      train_loss = loss.item()
      pbar.set_postfix({
          'Train Loss': train_loss
      })
      
      pbar.update(1)

    model.eval()
    val_losses = []
    for id_batch, (x_batch, y_batch) in enumerate(valLoader):
      outputs = model(x_batch)
      val_loss = criterion(y_batch, outputs)
      val_losses.append(val_loss.item())
    mean_val_loss = sum(val_losses)/len(val_losses)

    pbar.set_postfix({
        'Train Loss': train_loss,
        'Avg Val Loss': mean_val_loss,
        'learning rate': optimizer.param_groups[0]['lr'],
        'Best': 'True' if mean_val_loss < best_val_loss else 'False'
    })
    pbar.close()

    if mean_val_loss < best_val_loss:
      best_val_loss = mean_val_loss
      counter = 0
      th.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'val_loss': mean_val_loss,
      }, 'model.pt')
    elif mean_val_loss > (best_val_loss + min_delta):
      counter += 1

    if counter > patience:
      logger.info('stopping training early...')
      break

    scheduler.step()

  # logger.info('loading the best model...')
  # checkpoint = th.load('model.pt')
  # model.load_state_dict(checkpoint['model_state_dict'])

  dummy_input = th.randn(input_size,)
  th.onnx.export(
      model,
      th.ones(dummy_input.shape, dtype=th.float32,
              device=device),
      'trial.onnx',
      opset_version=9,
      input_names=['input'],
      output_names=['output']
  )
