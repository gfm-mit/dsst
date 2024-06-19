import torch

import gtorch.models.bn_utils

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
    gtorch.models.bn_utils.disable_running_stats(model)
    output = model(data.to(DEVICE))
    torch.nn.functional.nll_loss(output, target[:, 0].to(DEVICE)).backward()
    optimizer.second_step(zero_grad=True)
    gtorch.models.bn_utils.enable_running_stats(model)
  assert loader_has_batches
  return loss.item()

def next_token(model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  for data, target in train_loader:
    loader_has_batches = True
    output = model(data.to(DEVICE))
    loss = torch.nn.functional.mse_loss(output[:, :-1, :], data[:, 1:, :].to(DEVICE))
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # second forward-backward pass
    gtorch.models.bn_utils.disable_running_stats(model)
    output = model(data.to(DEVICE))
    torch.nn.functional.mse_loss(output[:, :-1, :], data[:, 1:, :].to(DEVICE)).backward()
    optimizer.second_step(zero_grad=True)
    gtorch.models.bn_utils.enable_running_stats(model)
  assert loader_has_batches
  return loss.item()