import torch
import numpy as np

import optim.loss_sam

def get_task_loss(model, optimizer, train_loader, task, offset=1):
  if task == "next_token":
    if optimizer.__module__ == "pytorch_optimizer.optimizer.sam":
      loss = optim.loss_sam.next_token(model, optimizer, train_loader, offset=offset)
    else:
      loss = next_token(model, optimizer, train_loader, offset=offset)
    description = 'Last Batch RMSE={:.2f}'.format(np.sqrt(loss))
  else:
    if optimizer.__module__ == "pytorch_optimizer.optimizer.sam":
      loss = optim.loss_sam.classify(model, optimizer, train_loader)
    else:
      loss = classify(model, optimizer, train_loader)
    
    with np.errstate(over='ignore'):
      exp_loss = np.exp(loss)
    description = 'Last Batch Perplexity={:.2f}'.format(exp_loss)
  if optimizer.__module__ == "pytorch_optimizer.optimizer.prodigy":
    loss = optimizer.param_groups[0]["d"]
    description = 'D={:.2e}  {}'.format(loss, description)
  elif hasattr(optimizer, "base_optimizer") and optimizer.base_optimizer.__module__ == "pytorch_optimizer.optimizer.prodigy":
    loss = optimizer.base_optimizer.param_groups[0]["d"]
    description = 'D={:.2e}  {}'.format(loss, description)
  return loss, description

def balance_class_weights(target):
    w = torch.nn.functional.one_hot(target).float().mean(axis=0)
    w = 1 / w
    w = torch.clip(w, 0, 10)
    w = w / w.sum()
    return w

def classify(model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  for data, target in train_loader:
    loader_has_batches = True
    def get_grad():
      output = model(data.to(DEVICE))
      loss = torch.nn.functional.nll_loss(output, target[:, 0].to(DEVICE))
      loss.backward()
      return loss
    loss = optimizer.step(get_grad)
  assert loader_has_batches
  return loss.item()

def next_token(model, optimizer, train_loader, offset=1):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  for data, target in train_loader:
    loader_has_batches = True
    data = data.to(DEVICE)
    def get_grad(data=data):
      output = model(data)
      assert output.shape == data.shape, f"{output.shape=} {data.shape=}"
      mask = 1 * torch.amax(data != 0, axis=2, keepdim=True)
      output = output * mask
      data = data * mask
      if offset == 0:
        loss = torch.nn.functional.mse_loss(output, data)
      else:
        loss = torch.nn.functional.mse_loss(output[:, :-offset, :], data[:, offset:, :])
      loss.backward()
      return loss
    loss = optimizer.step(get_grad)
  assert loader_has_batches
  return loss.item()