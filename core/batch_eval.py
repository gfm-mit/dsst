import numpy as np
import torch

def next_token(model, val_loader, verbose=False, offset=1):
  DEVICE = next(model.parameters()).device
  results = []
  model.eval()
  with torch.no_grad():
    for data, target in val_loader:
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
  predicted, data = zip(*results)
  predicted = np.concatenate(predicted)
  data = np.concatenate(data)
  return predicted, data

def binary_classifier(model, val_loader):
  DEVICE = next(model.parameters()).device
  results = []
  model.eval()
  with torch.no_grad():
    for data, target in val_loader:
      output = model(data.to(DEVICE)).to('cpu')
      results += [(
          output.detach().numpy(),
          target.detach().to('cpu').numpy(),
      )]
  logits, targets = zip(*results)
  logits = np.concatenate(logits)
  # only needed for accuracy
  #predictions = np.argmax(logits, axis=1)
  targets = np.concatenate(targets)
  return logits, targets

def binary_classifier_with_groups(model, loader):
  logits = []
  targets = []
  groups = []
  DEVICE = next(model.parameters()).device
  with torch.no_grad():
    for idx, (data, target, g) in enumerate(loader):
      #print(target)
      output = model(data.to(DEVICE)).to('cpu')
      logits += [output.detach().numpy()[:, 1]]
      targets += [target.detach().to('cpu').numpy()[:, 0]]
      groups += [g]
      if idx % 100 == 0 and idx > 0:
        print(f"metrics.get_combined_roc()[{idx}]")
  logits = np.concatenate(logits)
  targets = np.concatenate(targets)
  groups = np.concatenate(groups)
  return logits, targets, groups