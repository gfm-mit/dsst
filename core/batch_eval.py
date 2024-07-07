import numpy as np
import torch

def next_token(model, val_loader, verbose=False, offset=1):
  DEVICE = next(model.parameters()).device
  logits = []
  targets = []
  groups = []
  model.eval()
  with torch.no_grad():
    for (data, target, *g) in val_loader:
      output = model(data.to(DEVICE)).to('cpu')
      output = output.detach().numpy()
      data = data.detach().to('cpu').numpy()
      if offset > 0:
        output = output[:, :-offset, :]
        data = data[:, offset:, :]
      # hacks
      rmse = np.sqrt(np.mean((output - data)**2, axis=1, keepdims=True))
      zeros = np.zeros([data.shape[0], 1, data.shape[2]])
      logits += [rmse]
      targets += [zeros]
      if len(g) > 0:
        groups += g
  logits = np.concatenate(logits)
  targets = np.concatenate(targets)
  if len(groups) > 0:
    groups = np.concatenate(groups)
    return logits, targets, groups
  return logits, targets

def binary_classifier(model, loader):
  logits = []
  targets = []
  groups = []
  DEVICE = next(model.parameters()).device
  with torch.no_grad():
    for _, (data, target, *g) in enumerate(loader):
      output = model(data.to(DEVICE)).detach().to('cpu').numpy()
      label = target.detach().to('cpu').numpy()
      logits += [output[:, 1]]
      targets += [label[:, 0]]
      if len(g) > 0:
        groups += g
  logits = np.concatenate(logits)
  targets = np.concatenate(targets)
  if len(groups) > 0:
    groups = np.concatenate(groups)
    return logits, targets, groups
  return logits, targets