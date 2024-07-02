import numpy as np
import torch

def next_token(model, val_loader, verbose=False, offset=1):
  DEVICE = next(model.parameters()).device
  results = []
  groups = []
  model.eval()
  with torch.no_grad():
    for (data, target, *g) in val_loader:
      output = model(data.to(DEVICE)).to('cpu')
      if offset == 0:
        results += [(
            output.detach().numpy(),
            data.detach().to('cpu').numpy(),
        )]
      else:
        results += [(
            output.detach().numpy()[:, :-offset, :],
            data.detach().to('cpu').numpy()[:, offset:, :],
        )]
      if len(g) > 0:
        groups += g
  predicted, data = zip(*results)
  predicted = np.concatenate(predicted)
  data = np.concatenate(data)
  if len(groups) > 0:
    groups = np.concatenate(groups)
    return predicted, data, groups
  return predicted, data

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