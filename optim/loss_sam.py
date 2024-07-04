import torch

import models.bn_utils

def classify(model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  for data, target in train_loader:
    loader_has_batches = True
    # first forward-backward pass
    output = model(data.to(DEVICE))
    loss = torch.nn.functional.nll_loss(output, target[:, 0].to(DEVICE))
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # second forward-backward pass
    models.bn_utils.disable_running_stats(model)
    output = model(data.to(DEVICE))
    torch.nn.functional.nll_loss(output, target[:, 0].to(DEVICE)).backward()
    optimizer.second_step(zero_grad=True)
    models.bn_utils.enable_running_stats(model)
  assert loader_has_batches
  return loss.item()

def next_token(model, optimizer, train_loader, offset=1):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  for data, target in train_loader:
    loader_has_batches = True
    data = data.to(DEVICE)
    output = model(data)
    mask = 1 * torch.amax(data != 0, axis=2, keepdim=True)
    output = output * mask
    data = data * mask
    if offset == 0:
      loss = torch.nn.functional.mse_loss(output, data)
    else:
      loss = torch.nn.functional.mse_loss(output[:, :-offset, :], data[:, offset:, :])
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # second forward-backward pass
    models.bn_utils.disable_running_stats(model)
    output = model(data.to(DEVICE))
    if offset == 0:
      torch.nn.functional.mse_loss(output, data.to(DEVICE)).backward()
    else:
      torch.nn.functional.mse_loss(output[:, :-offset, :], data[:, offset:, :].to(DEVICE)).backward()
    optimizer.second_step(zero_grad=True)
    models.bn_utils.enable_running_stats(model)
  assert loader_has_batches
  return loss.item()