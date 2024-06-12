import torch
import numpy as np

def get_task_loss(epoch, model, optimizer, train_loader, task):
  if task == "next_token":
    loss = next_token(epoch, model, optimizer, train_loader)
    description = 'Train Epoch: {} \tLast Batch MSE: {:.2f}'.format(epoch, loss)
    return loss, description
  else:
    loss = classify(epoch, model, optimizer, train_loader)
    description = 'Train Epoch: {} \tLast Batch Perplexity: {:.2f}'.format(epoch, np.exp(loss))
    return loss, description

def balance_class_weights(target):
    w = torch.nn.functional.one_hot(target).float().mean(axis=0)
    w = 1 / w
    w = torch.clip(w, 0, 10)
    w = w / w.sum()
    return w

def classify(epoch, model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  for data, target in train_loader:
    loader_has_batches = True
    optimizer.zero_grad()
    output = model(data.to(DEVICE))
    #class_weights = balance_class_weights(target)[0, :]
    #loss = torch.nn.functional.nll_loss(output, target[:, 0].to(DEVICE), weight=class_weights)
    loss = torch.nn.functional.nll_loss(output, target[:, 0].to(DEVICE))
    loss.backward()
    optimizer.step()
  assert loader_has_batches
  return loss.item()

def next_token(epoch, model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  for data, target in train_loader:
    loader_has_batches = True
    optimizer.zero_grad()
    output = model(data.to(DEVICE))
    loss = torch.nn.functional.mse_loss(output[:, :-1, :], data[:, 1:, :].to(DEVICE))
    loss.backward()
    optimizer.step()
  assert loader_has_batches
  return loss.item()